#
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F


batch_size=32
block_size = 8
torch.manual_seed(0)
learning_rate = 3e-4
eval_interval = 500
eval_iters = 200
n_embd = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(r'D:\Backup\Desktop\programs\nanoGPT\input.txt', 'r', encoding = "utf-8") as file:
    text = file.read()

#

vocab = len(sorted(set(text)))


#
#create tokens for each char
tokens = {char: idx for idx, char in enumerate(set(text))}
snekot = {idx: char for idx, char in enumerate(set(text))}
encode = lambda s: [tokens[c] for c in s]
decode = lambda a: ''.join([snekot[i] for i in a])

data = torch.tensor(encode(text), dtype=torch.long)
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
#
block_size = 10
train_data[:block_size+1]
#this simulates a basic ngram proposition, where 50 implies 18. 50 and 18 implies 18. 50, 18, and 18 implies 63. etc.
#this trains the transformer to be used to context of n=1 all the way up to n=block_size.

#
#blocks/chunks are groups of tokens
#batches are groups of blocks/chunks

#

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0,len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def loss_eval():
    out = {}
    model.eval()#different modes for training and eval
    for split in ["train", "val"]:
        losses = torch.zeros(eval_interval, device=device)
        for i in range(eval_interval):
            x,y = get_batch(split)
            logits, loss = m(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

#
xb,yb = get_batch('train')


#
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        wei = query @ key.transpose(-2, -1)* C**(-0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        value = self.value(x)
        out = wei @ value
        return out

class multiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd) #transforms the heads into a single tensor to be outputted
    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),#adding a layer for residuals
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = multiHead(n_head, head_size)
        self.ffwd = FFN(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x





class Bigram(nn.Module):
    def __init__(self,  vocab):
        super().__init__()
        self.token_embedding_table= nn.Embedding(vocab, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)#hold positional value of each token
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(3)])
        self.lm_head = nn.Linear(n_embd, vocab)#creates an intermediary layer to convert token embeddings to the logits
        

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd= self.token_embedding_table(idx)#creates a [batch, chunk, n_embd] tensor
        #logits represent the next scores for each embedding
        position_embedding = self.position_embedding_table(torch.arange(T, device=device))#creates a [chunk, n_embd] tensor
        x = tok_embd + position_embedding#adds the positional embedding to the token embedding
        x = self.blocks(x)
        logits = self.lm_head(x)#creates a [batch, chunk, vocab_size] tensor
       
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)#reshaping for the loss fn
            targets = targets.view(-1)#reshaping for the loss fn
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max):
        #idx is the current token
        for _ in range(max):
            #get predictions
            idx_c = idx[:,-block_size:]# crop to block size
            logits, loss = self(idx_c)
            logits = logits[:,-1,:]#get the timestep to form (B,C) tensor
            probs= F.softmax(logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next), dim=1)
        return idx
model = Bigram(vocab)
m = model.to(device)
logits, loss = m(xb, yb)


#
#testing


#
#training
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

#

for i in range(5000):

    if i%eval_interval == 0:
        loss = loss_eval()
        print(loss['train'], loss['val'])

    x,y = get_batch('train')
    logits, loss = m(x, y)
    optimizer.zero_grad(set_to_none=True)#set grads to 0
    loss.backward()#do backward prop
    optimizer.step()#update params based on grad values


#
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, 200)[0].tolist()))

#



