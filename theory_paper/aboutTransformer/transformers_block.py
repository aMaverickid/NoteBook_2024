import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):

        super().__init__()

        assert k % heads == 0, 'the embedding dimension needs \
            to be divisible by the number of heads.'

        self.k, self.heads = k, heads

        # These compute the queries, keys and values for all
        # heads
        self.tokeys    = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues  = nn.Linear(k, k, bias=False)

        # This will be applied after the multi-head self-attention operation.
        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):

        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x)
        keys    = self.tokeys(x)
        values  = self.tovalues(x)

        s = k // h

        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # Get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # -- dot has size (b*h, t, t) containing raw weights

        dot = dot / (k ** (1/2))

        # normalize
        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = torch.bmm(dot, values)
        # - out now is [b*h, t, s]
        out = out.view(b, h, t, s)
        # print('temp2=', temp2)
        # print('temp2=', temp2.transpose(1,2))
        # print('temp2=', temp2.transpose(1,2).contiguous())
        out = out.transpose(1,2).contiguous().view(b, t, s*h)

        out = self.unifyheads(out)
        
        # print('out =', out)

# t, k = 3, 2
# mymodel = SelfAttention(k, heads=2)
# x = torch.randn(2, t, k)
# mymodel.forward(x)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
    
        self.attention = SelfAttention(k, heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.feedforward = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.feedforward(x)

        return self.norm2(fedforward + x)