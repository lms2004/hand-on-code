import torch
import torch.nn as nn
import torch.nn.functional as F
class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, groups=2, dropout=0.1):
        super().__init__()
        # assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        # assert num_heads % groups == 0, "num_heads must be divisible by groups" 
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.groups = groups
        self.head_dim = embed_dim // num_heads
        self.group_heads = num_heads // groups  # 每个组的头数
        # 注意KV的投影维度是groups * head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.groups * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.groups * self.head_dim) 
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, key_padding_mask=None):
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()
        # 投影QKV
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch, seq_len, groups*head_dim)
        v = self.v_proj(x)  # (batch, seq_len, groups*head_dim)
        # 调整维度，为注意力计算做准备
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.groups, self.head_dim).permute(0, 2, 3, 1)  # (batch, groups, head_dim, seq_len)
        v = v.view(batch_size, seq_len, self.groups, self.head_dim).transpose(1, 2)  # (batch, groups, seq_len, head_dim)
        # 扩展KV以匹配组的头数
        k = k.unsqueeze(2).expand(-1, -1, self.group_heads, -1, -1).contiguous()
        k = k.view(batch_size, self.num_heads, self.head_dim, seq_len)
        v = v.unsqueeze(2).expand(-1, -1, self.group_heads, -1, -1).contiguous()
        v = v.view(batch_size, self.num_heads, seq_len, self.head_dim)
        # 计算注意力分数
        attn_scores = torch.matmul(q, k)  # (batch, num_heads, seq_len, seq_len)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            # 处理padding mask (batch, seq_len)
            mask = key_padding_mask.view(batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 加权求和
        output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)