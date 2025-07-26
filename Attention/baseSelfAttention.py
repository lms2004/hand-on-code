from math import sqrt

import torch
import torch.nn as nn
import math

class MySelfAttention(nn.Module):
    def __init__(self, dim_embedding, dim_qk, dim_v):
        super(MySelfAttention, self).__init__()
        self.dim_embedding = dim_embedding
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self._norm_fact = 1 / sqrt(dim_qk)
        

        self.linear_q = nn.Linear(dim_embedding, dim_qk, bias=False)
        self.linear_k = nn.Linear(dim_embedding, dim_qk, bias=False)
        self.linear_v = nn.Linear(dim_embedding, dim_v, bias=False)
    def forward(self, x):
        # x: batch, sequence_length, dim_embedding
        # 根据文本获得相应的维度
        
        batch, n, dim_embedding = x.shape
        assert dim_embedding == self.dim_embedding
        
        # nn.Linear 自动生成 W，b
        # 1. Q = X * W_Q + b
        # 2. K = X * W_K + b
        # 3. V = X * W_V + b 
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # Q * K^T / sqrt(dim_embedding) -> transpose 处理 batch 中 转置
        score = torch.bmm(q, k.transpose(1, 2))* self._norm_fact

        # Softmax( Q * K^T / sqrt(dim_embedding)) * V
        score = torch.softmax(score, dim=-1)  # batch, sequence_length, sequence_length
        att = torch.bmm(score, v)
        return att

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
 
        #定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)
 
    def forward(self, x):
        # x: batch, n, dim_q
        #根据文本获得相应的维度
 
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q
 
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        #q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        #归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        #attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)
        return att

# Inputs to the attention module
batch_size = 3    # 每次选取3句话
dim_embedding = 6    # input_size
sequence_length = 4    #每句话固定有4个单词(单词之间计算注意力)
dim_V = 8    # V 向量的长度(V向量长度可以与Q and K不一样)
dim_QK = 7    #Q and K向量的长度(dim_embedding经过Wq、Wk变换后QK向量长度变为dim_QK)

x_gen = torch.randn(batch_size, sequence_length, dim_embedding)
attention = SelfAttention(dim_embedding, dim_QK, dim_V)
myattention = MySelfAttention(dim_embedding, dim_QK, dim_V)

att = attention(x_gen)
myatt = myattention(x_gen)

print("att shape:", att.shape)
print("myatt shape:", myatt.shape)


