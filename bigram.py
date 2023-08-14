#
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F


batch_size=32
block_size = 10
torch.manual_seed(0)
learning_rate = 1e-3
eval_interval = 300
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
print(xb.shape,'\n',xb)
print(yb.shape,'\n',yb)


#

class Bigram(nn.Module):
    def __init__(self,  vocab):
        super().__init__()
        self.token_embedding_table= nn.Embedding(vocab, vocab)
    def forward(self, idx, targets=None):
        logits= self.token_embedding_table(idx)#creates a [batch, chunk, vocab] tensor
            #logits represent the next scores for each embedding
       
        
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
            logits, loss = self(idx)
            logits = logits[:,-1,:]#get the timestep to form (B,C) tensor
            probs= F.softmax(logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next), dim=1)
        return idx
model = Bigram(vocab)
m = model.to(device)
logits, loss = m(xb, yb)
print(logits.shape)


#
#testing


#
#training
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

#

for i in range(10000):

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
print(decode(m.generate(context, 100)[0].tolist()))

#



