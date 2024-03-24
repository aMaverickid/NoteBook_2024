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
    
class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super.__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the
		# heavy lifting

        # wrong version ? Why ?
        # def transformers(x):
        #     for _ in range(depth)
        #         new_transform = TransformerBlock(k, heads)
        #         x = new_transform.forward(x)
        
        # self.tblocks = transformers

        # right version
        tblock = []
        for i in range(depth):
            tblock.append(TransformerBlock(k, heads))
        self.tblocks = nn.Sequential(*tblock)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)
        

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
                !!! question: What's that? one-hot encoding? --> 看Embedding()的实现代码发现是就只是 indices 输入
                # A: emebdding 的输入是一个词汇表的整数索引！
                # 本例中的预处理:对输入文本进行build_vocab操作, (根据单词出现频率)建立了一个词汇表, 将一串文本转化成一串 indices
                # 然后将其映射到连续的低维向量
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        # generate position embeddings
        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :,:].expand(b, t, k)

        x = tokens + positions

        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class
        # probabilities

        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
        