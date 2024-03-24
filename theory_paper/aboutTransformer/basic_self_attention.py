import torch
import torch.nn.functional as F

""" 
How to express the input & ouput sequence?
- input sequence: [batch_size b, seq_len t, dimension k]
"""

"""
What is the inner product?
- inner product: input x input^T
- inner product: [b, t, k] x [b, k, t] = [b, t, t]
- a Matrix W!
"""

"""
So how can we get the inner product, preferably in a fast way?
- torch.bmm: batch matrix multiplication
"""

# assume we have some tensor x with size (b, t, k)
x = torch.randn(3, 5, 7)

raw_weights = torch.bmm(x, x.transpose(1, 2))
# - torch.bmm is a batched matrix multiplication. It
#   applies matrix multiplication over batches of
#   matrices.

weights = F.softmax(raw_weights, dim=2)

y = torch.bmm(weights, x)

# print
print(y)
