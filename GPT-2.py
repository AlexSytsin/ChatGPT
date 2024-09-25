from  dataclasses import dataclass
import torch
import math
import time
import inspect
import torch.nn as nn
import numpy as np
import os
from torch.nn import functional as F
import tiktoken
from hellaswag import render_example, iterate_examples
#HF_HUB_DISABLE_SYMLINKS_WARNING = True
# --------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.n_head = config.n_head

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)

        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.register_buffer('bias', torch.tril(torch.ones(1,1, self.block_size, self.block_size)))


    def forward(self, x):
        B, T, C = x.shape # (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B,T,self.n_head,self.n_embd // self.n_head).transpose(1,2)  # (B, n_heads, T, head_size)
        v = v.view(B,T,self.n_head,self.n_embd // self.n_head).transpose(1,2)  # (B, n_heads, T, head_size)
        k = k.view(B,T,self.n_head,self.n_embd // self.n_head).transpose(1,2)  # (B, n_heads, T, head_size)

        # att = q @ k.transpose(2,3) * k.size(-1) ** -0.5 # (B, n_heads, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B, n_heads, T, T)
        # att = F.softmax(att, dim=-1)  # (B, n_heads T, T)
        # out = att @ v  # (B, n_heads, T, T) @ (B, n_heads, T, head_size) -> (B, n_heads, T, head_size)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1,2).reshape(B,T,C)
        out = self.c_proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing between embeddng and last layer
        self.transformer.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f" Block_size smaller than seq len {T}"
        # token and position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # T, n_embd
        tok_emb = self.transformer.wte(idx) # B, T, n_embd
        x = tok_emb + pos_emb
        # forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)
        # final LayerNorm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # B, T, vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn : p for pn,p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params' : nodecay_params, 'weight_decay' : 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = use_fused)
        return optimizer


# --------------------------------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T,  split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# micro batches
total_batch_size = 4 * 64 # нужно поставить 524288
B = 4 # нужно поставить 16
T = 64 # нужно поставить 1024
assert total_batch_size % (B * T) == 0, "Batch size isn't divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size}")
print(f"Gradient accumulation steps: {grad_accum_steps}")
train_loader = DataLoaderLite(B=B,T=T, split="train")
val_loader = DataLoaderLite(B=B,T=T, split="val")
torch.set_float32_matmul_precision('high')


enc = tiktoken.get_encoding("gpt2")
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
use_compile = False
if use_compile: # dkfsdfksdkfspodfkpewok
    model = torch.compile(model)   #Обязаательно нужно скомпилировать

max_lr = 6e-4 # can set higher upto x3
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    # warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # stable lr
    if it > max_steps:
        return min_lr

    # cosine decay weight
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# optimizing
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device=device)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()

    last_step = step == max_steps - 1

    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)

                # with torch.autocast(device_type=device, dtype=torch.bfloat16):  # Доооооооообавь строку
                logits, loss = model(x, y)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        if step > 0 and (step % 5000 == 0 or last_step):
            # optionally write model checkpoints
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }
            # you might also want to add optimizer.state_dict() and
            # rng seeds etc., if you wanted to more exactly resume training
            torch.save(checkpoint, checkpoint_path)


    """# once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            print(0)
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                # with torch.autocast(device_type=device, dtype=torch.bfloat16): # добавь
                logits, loss = model(tokens)

                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
            print(i)
        acc_norm = num_correct_norm / num_total

        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")"""




    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 5
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                # with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(xgen)  # (B, T, vocab_size)


                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample {i}: {decoded}")





    model.train()
    optimizer.zero_grad()
    loss_acum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):  # Доооооооообавь строку
        logits, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_acum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    #torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = (tokens_processed) / (t1- t0)
    print(f"step {step:4d}, loss: {loss_acum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, time: {dt:.2f} ms, tok/sec: {tokens_per_sec}" )
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_acum.item():.6f}\n")



