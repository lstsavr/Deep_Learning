import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap


# hyperparameters 超级参数
batch_size = 64 # batch是一次喂给模型的数据子集，batch size是每个batch中含有的样本数量
block_size = 256 # block size就是每个样本中token的数量
max_iters = 1000 # 训练循环次数（默认5000）
eval_interval = 500 # evaluate interval,训练多少个batch评估一下模型的效果，通过计算loss函数的方式
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # evaluate iters,评估效果的过程中（纯推理），抽样多少个测试数据用来计算loss函数
n_embd = 384 # embedding向量的维数
n_head = 6 # 多头注意力block里有几个头
n_layer = 6 # 有几个block（层）
dropout = 0.2 # 训练中，随机20%的节点会设置为0,以减少过拟合增强模型的通用性
wrap_width = 50
# ------------

torch.manual_seed(1337) # 随机种子
file_name = "txt"

# ----数据预处理---------------
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) # set()创建一个无序无重复的集合，然后转化为一个list，再按sorted进行排序
vocab_size = len(chars) # 一共有多少个字母

#构建字符和整数index之间的双向映射字典，把字符转换为模型能训练的token ID，把模型输出的数字转化为字符
#字典推导式：{key:value for item in iterable}这里的enumerate()可以返回chars里的索引和值，i,ch默认是遍历chars里的索引和值

stoi = { ch:i for i,ch in enumerate(chars) } # 字符到序号（symbol to index）的对应字典
itos = { i:ch for i,ch in enumerate(chars) } # 序号到字符（index to symbol）的对应字典

#lambda是一个匿名函数，快速定义一个只包含一行表达式的小函数
#列表推导式 [stoi[c] for c in s],遍历stoi，找出对应编号的序号，然后形成一个列表
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# 数据分组处理
data = torch.tensor(encode(text), dtype=torch.long)  #先把文本转换为token ID列表，然后再转换为张量，用整数（index）代表每一个字符
n = int(0.9*len(data))  #用int是可能乘法会导致有小数
train_data = data[:n]
val_data = data[n:]

print(f"File {file_name} has been read and processed.")


#-----定义函数与模型------------

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #这个randint指的是，生成的值的范围是0-len(data) - block_size，形状是(batch_size,)
  #因为后面还会加block size，所以x = data[99743 : 99999]和y = data[99744 : 100000]是极限。随机确定每个batch item的起始点
    x = torch.stack([data[i:i+block_size] for i in ix]) # 读取输入序列并叠起来组成batch，这里的for循环是指，ix里的所有值都会被遍历，作为起点，从data中取值出来。最终这些值组成一个列表
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # 比x往后移一位并叠起来，目的是给定前面的token，预测下一个
    x, y = x.to(device), y.to(device)
    return x, y # x，y 存储的数据都是整数，是每个字符的序号（index）

@torch.no_grad() # 不做梯度计算的decorator,作用域为整个函数
def estimate_loss(model):#评估模型在训练集和验证集上的平均损失，不做反向传播，仅用于查看当前模型表现
    out = {} #创造一个空字典，用来记录两个loss的平均值，out['train']：训练集上的平均 loss，out['val']：验证集上的平均 loss
    model.eval() # 把模型转化为evaluate模式（默认模式是train，防止Dropout影响评估结果）
    for split in ['train', 'val']:#会轮流用，第一次看train，第二次看val的。
        losses = torch.zeros(eval_iters) # 建立一个初始值为0的容器，用于储存loss值，每取一个batch，计算一次loss，就把结果塞进这个数组里面, eval iters是取的次数
        for k in range(eval_iters):
            X, Y = get_batch(split) # split是一个字符串，用来控制get_batch()函数的行为
            logits, loss = model(X, Y) # model的输入值一个是index（以每个字符的序号表示的序列），一个是target
            losses[k] = loss.item()# losses就像个列表，k就像索引，把每次的loss都存进去了.item将这个loss变成普通的python浮点数
        out[split] = losses.mean() # out是含有两个元素的字典，一个是train，一个是val，每个元素对应一个loss的平均值
    model.train() # 再转化为训练模式（如果之前没有转为evaluate模式，则不需要这一步，因为模型建立后默认为训练模式）
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # 输入为n_embd维，输出为head_size维
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 
        # 这个地方的语法，tril指的是挡住右上方的，使得右上方的值为0
        # 建立下三角mask矩阵，此处定义了self.tril
        # register_buffer的作用是使这个矩阵成为模型的一部分（而不仅仅是一个变量），移动到显卡的时候也能随着移动
        # 而且register_buffer是不可训练的，它只是一个常量，在训练中不被优化
        # 此处的register_buffer定义了self.tril作为模型的一个不可训练的结构

        self.dropout = nn.Dropout(dropout) # # 训练中，一部分随机节点会设置为0,以减少过拟合增强模型的通用性

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape # B,T,C 分别代表Batch size, Time steps (token个数), Channels (or embedding size)
        k = self.key(x)   # (B,T,hs) 此处的x是embedding向量格式，脚注①
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
      # k.shape[-1]**-0.5是缩放因子，归一化用的，防止点积值太大，导致softmax变得极端
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)，tensor.masked_fill(mask, value)把对于mask来说是true的位置换成value.
        #[:T, :T]是取一个T * T大小的矩阵
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # 可调用对象（因为类里面有一个forward()方法）
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs) 脚注③
        return out # (B, T, hs)
    #_init_()是构造函数，创建模型的结构，forward是向前传播,是模型运行的逻辑，每次调用模型都执行这个forward

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      # 创建 num_heads 个独立的注意力头，每个头都用 Head(head_size) 初始化，这里写Head，就是创建了一个Head对象，并且传入
      #head_size作为参数，初始化这个头。创建num_heads 个 Head(head_size) 实例，组成一个列表
      #nn.ModuleList([...])是个特殊容器，用于装很多模块，例如self.proj = nn.Linear(head_size * num_heads, n_embd) 
        self.proj = nn.Linear(head_size * num_heads, n_embd) # 把连接起来的“多头”运算结果再投射回词嵌入空间，因为是在最后一个维度上拼接的，所以这样转化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):# x依旧是batch size，block size，n_embd的大小
        # head把n_embd维数据转化成n_head个head_size维，然后再连接成n_embd维（n_embd = n_head X head_size）
        out = torch.cat([h(x) for h in self.heads], dim=-1)#在最后一个维度上拼接
      #这是个列表推导式，遍历了每个头，把输入x传进去，得到每个head的输出，然后把这些输出全部拼起来
        out = self.dropout(self.proj(out))#把out送进线性层里处理一下，
        return out # (B, T, n_embd)

class FeedFoward(nn.Module):#前馈神经网络层

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( #sequential就是按顺序执行这几个layer，然后return变换后的输入值）
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) # (B, T, n_embd)

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # 把词嵌入向量线性变换为多个head，平行处理注意力 脚注④
        self.sa = MultiHeadAttention(n_head, head_size) # sa = self attention
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)# LayerNorm是层归一化
  # 对每个 token 的特征向量 进行归一化（标准化），使它的均值为 0，方差为 1，这样可以让模型训练更稳定，更容易收敛
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # 将x进行layernorm归一化后，送进多头注意力那个class，+x是残差表示
      #self-attention 会重新编码 token 向量，但原始 x 中的信息也很有价值，所以我们直接加回来，保证原始信息也能保留
        x = x + self.ffwd(self.ln2(x))#再归一化，送进前馈神经网络，做非线性处理
        return x # (B, T, n_embd)

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 词嵌入，word embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Positional embedding 位置嵌入
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
      #*是解包，这个列表是生成n_layer个block，把每一项单独传进去给sequential
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    
        self.apply(self._init_weights)#对模型的子模块都执行_init_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # 此处的idx为token（词典里的序号）格式
        B, T = idx.shape # T = block_size, 在get_batch()函数里，每条序列的长度就是由block_size定的

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # 对传入的idx进行词嵌入，[B, T, n_embd]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # 位置嵌入，[T, n_embd]
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C) # 几个block顺序处理，每个里面都包含“多头”注意力
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 摊平，flatten
            targets = targets.view(B*T) # targets也摊平
            loss = F.cross_entropy(logits, targets) # logits 和 targets的shape不一样，这是交叉熵计算loss

        return logits, loss

    # 生成文本
    def generate(self, token_sequ, max_new_tokens):
        # token_sequ is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):#每次预测下一个token，重复max_new_tokens次
            # crop token_sequ to the last block_size tokens（不能长于block size）
            tokens_input = token_sequ[:, -block_size:] # 只保留序列最后的block size作为输入
            logits, loss = self.forward(tokens_input) # logits, (B,T,vocab_size)
            logits = logits[:, -1, :] # becomes (B, vocab_size)，切片，每个batch只取最后一个token
            probs = F.softmax(logits, dim=-1) # 计算概率
            token_next = torch.multinomial(probs, num_samples=1) # (B, 1) 以分布值为概率随机选择，multinomial相比argmax更加灵活
            token_sequ = torch.cat((token_sequ, token_next), dim=1) # (B, T+1)，拼接到上文后面
        new_tokens = token_sequ[:, -max_new_tokens:] # 生成完毕之后，返回最后max_new_tokens 个 token
        return new_tokens

#---main()函数--------------------------

def main():
    print(f"训练内容：{file_name}")
    model = GPTLanguageModel() # 实例化一个模型
    model = model.to(device) # 移到GPU， m 和 model 可以混用
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer 设定优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #model.parameters（）是他要优化的参数

    # 训练循环
    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            #eval_interval 是你设定的评估频率（比如每 500 步评估一次）.estimate_loss(model)：会在 train/val 数据上计算平均 loss，这一步是为了查看模型的训练效果，是否在降低
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data， 根据随机起始点在训练数据中提取一个batch
        xb, yb = get_batch('train') # xb， yb 中的数据时序长度都是block_size，每个token都用一个整数表示

        # evaluate the loss
        logits, loss = model(xb, yb) # 向前传播，计算loss
        optimizer.zero_grad(set_to_none=True) # 梯度重置
        loss.backward() # 计算损失函数，反向传播
        optimizer.step() # 优化一步，参数更新

    print ("Training complete 训练结束，下面开始生成内容：")
    # generate from the model
    max_new_tokens = 500
    start_idx = random.randint(0, len(val_data)-block_size) # val_data 是一维tensor

    context = torch.zeros((1, block_size), dtype=torch.long, device=device) #(B, T) T = block_size
    real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)

    context[0, :] = val_data[start_idx: start_idx+block_size] # context的batch里有多条数据，这里只对第0条赋值，其余仍都是0
    context_str = decode(context[0].tolist()) # [0],只取把它吃里的第0条数据， 把context由二阶变为一阶张量
    wrapped_context_str = textwrap.fill(context_str, width=wrap_width)

    real_next_tokens[0, :] = val_data[start_idx+block_size: start_idx+block_size+max_new_tokens] # 截取、赋值
    real_next_str = decode(real_next_tokens[0].tolist()) # [0] 把context由二阶变为一阶张量
    wrapped_real_next_str = textwrap.fill(real_next_str, width=wrap_width)

    generated_tokens = model.generate(context, max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist())
    wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)

    print("context：")
    print(wrapped_context_str)
    print("generate:")
    print(wrapped_generated_str)
    print("Real next content:")
    print(wrapped_real_next_str)

#------执行---------------
main()

