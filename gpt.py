import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64
block_size = 32 #256
n_embd = 64     #384
n_head = 4  #6
n_layer = 3 #6
learning_rate = 3e-4
max_iters=5000
eval_interval = 500
eval_iters = 50 #200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.0   #0.2
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

#encode
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = (int)(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


#load data
def get_batch(split):
    #generates a small batch of inputs x and output y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)       # B, T, head_size
        k = self.key(x)         # B, T, head_size
        wei = q @ k.transpose(-2, -1) * k.shape[-1] **-0.5      #B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v   #B, T, C
        return out
    
    
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.head = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.head], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.head = MultiHead(num_heads, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        #communication
        x = x + self.head(self.ln1(x))
        #computation
        x = x + self.ff(self.ln2(x))
    
        return x


class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)
        self.lf = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)   #B, T, C
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))  #T, C
        x = tok_emb + pos_emb # B, T, n_embd
        x = self.blocks(x)
        x = self.lf(x)
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx



model = GPTModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, 1000)[0].tolist()))


















# with open('input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# stoi = {ch:i for i, ch in enumerate(chars)}
# itos = {i:ch for i, ch in enumerate(chars)}
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: "".join([itos[i] for i in l])

# data = torch.tensor(encode(text), dtype=torch.long)
# n = (int)(0.9*len(data))
# train_data = data[:n]
# val_data = data[n:]


# #load data
# def get_batch(split):
#     #generates a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data)-block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1: i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y

# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out


# class Head(nn.Module):
#     def __init__(self, head_size):
#         super().__init__()
#         self.key = nn.Linear(n_embd, head_size, bias = False)
#         self.value = nn.Linear(n_embd, head_size, bias=False)
#         self.query = nn.Linear(n_embd, head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         #todo 
#         B,T,C = x.shape
#         k = self.key(x)
#         q = self.query(x)
#         v = self.value(x)
#         wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T). each token interacting with every
#         #other token to calculate affinity
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
#         wei = F.softmax(wei, dim=-1) # (B, T, T)
#         wei = self.dropout(wei)
#         out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)  
#         return out
    
# class MultiHeadAttention(nn.Module):
#     def __init__(self, num_heads, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = nn.Linear(head_size*num_heads, n_embd)
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.dropout(self.proj(out))
#         return out
        

# class Feedforward(nn.Module):
#     def __init__(self, n_embd):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_embd, 4 * n_embd),
#             nn.ReLU(),
#             nn.Linear(4*n_embd, n_embd),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)
        
# class Block(nn.Module):
#     #transformer block. communication followed by computation
#     def __init__(self, n_embd, num_heads):
#         super().__init__()
#         head_size = n_embd // n_head
#         self.sa_head = MultiHeadAttention(num_heads, head_size)
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.ffwd = Feedforward(n_embd)
#         self.ln2 = nn.LayerNorm(n_embd)
#         return
#     def forward(self, x):        #todo
#         x = x+ self.sa_head(self.ln1(x))
#         x = x+ self.ffwd(self.ln2(x))
        
#         return x
        
# class GPTModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #token reads off the logits for the next token from a lookup table
#         self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
#         self.position_embedding_table = nn.Embedding(block_size, n_embd)
#         self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
#         self.ln_f = nn.LayerNorm(n_embd)
#         self.lm_head = nn.Linear(n_embd, vocab_size)

#     def forward(self, idx, targets=None):
#         B,T = idx.shape
        
#         tok_emb = self.token_embedding_table(idx)     #B, T, C
#         pos_emb = self.position_embedding_table(torch.arange(T, device=device))   #T, C
#         x = tok_emb + pos_emb #B, T, C
#         x = self.blocks(x)
#         x = self.ln_f(x)
        
#         logits = self.lm_head(x)
        
#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logits.shape
#             logits = logits.view(B*T, C)
#             targets = targets.view(B*T)
#             loss = F.cross_entropy(logits, targets)
            
#         return logits, loss
    
#     def generate(self, idx, max_new_tokens):
#         #idx is (B, T) array of indices in the current context
#         for _ in range(max_new_tokens):
#             #crop idx to the last block_size tokens
#             idx_cond = idx[:, -block_size:]
#             #get the predictions
#             logits, loss = self(idx_cond)
#             #focus only on the last time step
#             logits = logits[:, -1, :] # becomes (B, C)
#             #apply softmax to get probabilities
#             probs = F.softmax(logits, dim=-1) # (B, C)
#             #sample from the distribution
#             idx_next = torch.multinomial(probs, num_samples=1)
#             idx = torch.cat((idx, idx_next), dim=1)
#         return idx
        

# model = GPTModel()
# m = model.to(device)

# # print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
# for iter in range(max_iters):
#     if iter % eval_interval == 0 or iter == max_iters -1:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
#     xb, yb = get_batch('train')
#     logits, loss = model(xb,yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
    
    
# context = torch.zeros((1,1), dtype=torch.long, device=device)
# with open('output.txt', 'w') as f:
#     print(decode(m.generate(context, max_new_tokens=500)[0].tolist()), file=f)