import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap
#torch.nn,nn is neural network😊 and functional include function like loss and activate

batch_size = 56
block_size = 224
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 128
num_heads = 10
head_size = n_emd//num_heads
n_layer = 6
learning_rate=0.0003
max_iters=500
eval_interval =int(max_iters/10)
eval_iters =200
dropout_value = 0.2

wrap_width = 50

torch.manual_seed(325) '''The random seed is used to control the initial state of the random number 
generator, ensuring that the generated random number sequence is the same every time the program runs. 
'''
file_name ="西游记.txt"

with open(file_name,'r', encoding='uft-8') as f:
    text = f.read()
chars = sorted(list(set(text))) #delete the duplicate characters,and then convert to list, and then sorted
vocal_size = len(chars)

stoi ={ch:i for i,ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda str1: [stoi[c]] for c in str1]
decode = lambda list1:"".join([itos[i]] for i in list1])

data = torch.tensor(encode(text), dtype=torch.long)

train_size = int(0.8 * len(data))
valid_size = int(0.1 * len(data))

train_data = data[:train_size]
valid_data = data[train_size:train_size + valid_size]
test_data = data[train_size + valid_size:]
print(f"文件(西游记.txt)读取完成")

def batch(split):
    if split == 'train':
        data = train_data
    elif split == 'valid':
        data = valid_data
    else split == 'test':
        data = test_data
    ix = torch.randint(len(data)- block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y
#---------------------------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
        model.train()    
        return out
#---------------------------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self,head_size): 
        super().__init__()
        self.key = nn.linear(n_embd,head_size, bias=False)
        self.query = nn.linear(n_embd,head_size, bias=False)
        self.value = nn.linear(n_embd,head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1)* k.shape[-1]**-0.5
        wei = wei.masked_fill(self.trill ==0, float("-inf"))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,head__size)
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout_value)

def forward(self, x)
    out = torch.cat([h(x) for h in self.heads],dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
    self.net = nn.Sequential(
        nn.Linear(n_embd, n_embd*4),
        nn.ReLU(),
        nn.Linear(n_embd*4, n_embd),
        nn.Dropout(dropout_value),
    )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self. n_embd, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self. ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
#---------------------------------------------------------------------------------------------
class languagemodel(nn.module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocal_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(ln_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocal_size)
     
    def forward(self, idx, targets = None):
        B, T = idx.shape
        token_embd = self.token_embedding_table(idx)
        position_idx = torch.arange(T,device = device)
        position_embd = self.position_embedding_table(position_idx)
        x = token_embd + position_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
                
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, token_sequ, max_new_tokens):
        for _ in range(max_new_tokens):
            tokens_input = token_sequ[:, -block_size:]
            logits, loss = self.forward(tokens_input)
            logits = logits[:, -1 :]
            probs = F.softmax(logits. dim=-1)
            token_next = torch.multinomial(probs, num_samples=1)
            token_sequ = torch.cat((token_sequ, token_next), dim=1)
        new_tokens = token_sequ[:,-max_new_tokens:]
        return new_tokens
#----------------------------------------------------------------------------------------------
def main()
    print(f"train content:{file_name}")
    model = languagemodel()
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters())/1e6,'M parameters')
#
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i%eval_interval ==0 or i == max_iters-1:
        losses = estimate_loss(model)
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb,yb = get_batch("train")
    logits,loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
max_new_tokens = 600
start_idx = random.randint(0, len(valid——data)-block_size-max_new_tokens)
#
context = torch.zeros((1, block_size), dtype=torch.long, device=device)
context[0,:] = valid_data[start_idx: start_idx+block_size]
context_str = decode(context[0]].tolist())
wrapped_context_str = textwrap.fill(context_str, width=wrap_width)
#
real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)
real_next_tokens[0,:] = valid_data[start_idx+block_size: start_idx+block_size+max_new_tokens]
real_next_tokens_str = decode(real_next_tokens[0]].tolist())
wrapped_real_next_tokens_str = textwrap.fill(real_next_tokens_str, width=wrap_width)
#
generated_tokens = model.generate(context, max_new_tokens)
generated_str = decode(generated_tokens[0].tolist())
wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)

print(wrapped_context_str)
print(wrapped_generated_str)
print(wrapped_real_next_tokens_str)
#--------------------------------------------------------------------------------------------------
main()
head = Head(5)
print(head.tril)
wei = torch.ones(block_size,block_size)
print(wei)
wei = wei.masked_fill(head.trill ==0, float("-inf"))
print(wei)
wei = F.softmax(wei, dim = -1)
print(wei)
