import torch
import torch.nn as nn

class SparseAttention(nn.Module):
    def __init__(self, dim, window_size=64, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads)
        q, k, v = qkv.unbind(2)
        
        # 分块处理
        x_windows = x.view(B, N // self.window_size, self.window_size, C)
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) / (C ** 0.5)
        
        # 创建局部掩码
        mask = torch.ones_like(attn)
        mask = torch.triu(mask, diagonal=self.window_size//2) 
        mask += torch.tril(mask, diagonal=-self.window_size//2)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        return self.proj(out)




