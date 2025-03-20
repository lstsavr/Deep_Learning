import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap

batch_size = 64  
block_size = 256  
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 256  
num_heads = 8 
head_size = n_embd // num_heads
n_layer = 8 
learning_rate = 0.0005  
max_iters = 600  #
eval_interval = max_iters // 12  
eval_iters = 250  # 
dropout_value = 0.3  
wrap_width = 50 

torch.manual_seed(325) '''The random seed is used to control the initial state of the random number 
generator, ensuring that the generated random number sequence is the same every time the program runs. 
'''

file_name = "西游记.txt"

with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))  
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda lst: "".join(itos[i] for i in lst)

data = torch.tensor(encode(text), dtype=torch.long)

train_size = int(0.8 * len(data))
valid_size = int(0.1 * len(data))
test_size = len(data) - train_size - valid_size

train_data = data[:train_size]
valid_data = data[train_size:train_size + valid_size]
test_data = data[train_size + valid_size:]

print(f" file ({file_name}) 读取完成")
print(f" train data: {train_size} 个字符")
print(f" valid data: {valid_size} 个字符")
print(f" test data: {test_size} 个字符")

def get_batch(split):
    dataset = train_data if split == 'train' else valid_data if split == 'valid' else test_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()  
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train() 
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, head_size)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout_value),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embd = self.token_embedding(idx)
        pos_embd = self.position_embedding(torch.arange(T, device=device))
        x = token_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
            return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  
            probs = F.softmax(logits, dim=-1)  
            next_idx = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat((idx, next_idx), dim=1)  
        return idx[:, -max_new_tokens:]  

def main():
    print(f"训练文件: {file_name}")

    model = LanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in range(max_iters):
        xb, yb = get_batch("train")
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        if i % eval_interval == 0 or i == max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {i}: Train Loss {losses['train']:.4f}, Valid Loss {losses['valid']:.4f}")

if __name__ == "__main__":
    main()
    
max_new_tokens = 6000

start_idx = random.randint(0, len(test_data) - block_size - max_new_tokens)

context = torch.zeros((1, block_size), dtype=torch.long, device=device)
context[0, :] = test_data[start_idx: start_idx + block_size]

context_str = decode(context[0].tolist())
wrapped_context_str = textwrap.fill(context_str, width=wrap_width)

real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)
real_next_tokens[0, :] = test_data[start_idx + block_size: start_idx + block_size + max_new_tokens]

real_next_tokens_str = decode(real_next_tokens[0].tolist())
wrapped_real_next_tokens_str = textwrap.fill(real_next_tokens_str, width=wrap_width)

generated_tokens = model.generate(context, max_new_tokens)

generated_str = decode(generated_tokens[0].tolist())
wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)

# 打印结果
print("\n西游记的原始的上下文:\n")
print(wrapped_context_str)

print("\n这是模型生成的文本:\n")
print(wrapped_generated_str)

print("\n这是西游记的真实文本:\n")
print(wrapped_real_next_tokens_str)


