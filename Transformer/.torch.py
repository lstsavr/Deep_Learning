import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap
#torch.nn,nn is neural network😊 and functional include function like loss and activate

batch_size = 3
block_size = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 3

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
class languagemodel(nn.module):
    def__init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocal_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.network = nn.Linear(n_embd,vocal_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        token_embd = self.token_embedding_table(idx)
        position_idx = torch.arange(T)
        position_embd = self.position_embedding_table(position_idx)
        x = token_embd + position_embd
        logits = self.network(x)
                
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
model = languagemodel()
model = model.to(device)

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
