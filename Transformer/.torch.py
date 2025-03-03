import torch
import torch.nn as nn
from torch.nn import functional as F
#torch.nn,nn is neural network😊 and functional include function like loss and activate

batch_size = 8
block_size = 5

torch.manual_seed(325) '''The random seed is used to control the initial state of the random number 
generator, ensuring that the generated random number sequence is the same every time the program runs. 
'''
file_name ="text"

with open(file_name,'r', encoding='uft-8') as f:
    text = f.read()
chars = sorted(list(set(text))) #delete the duplicate characters,and then convert to list, and then sorted
stoi ={ch:i for i,ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor(encode(text), dtype=torch.long)

train_size = int(0.8 * len(data))
valid_size = int(0.1 * len(data))

train_data = data[:train_size]
valid_data = data[train_size:train_size + valid_size]
test_data = data[train_size + valid_size:]

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

batch("train")