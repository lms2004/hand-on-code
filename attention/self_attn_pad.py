from math import sqrt

import torch
import torch.nn as nn
import math

import torch
import torch.nn.functional as F


def collate_batch(batch, max_padding=128, pad_id=-1, eos_id=3):
    """
    处理输入batch，添加填充符号并生成mask。
    :param batch: 句子对的列表
    :param max_padding: 句子最大长度
    :param pad_id: 填充符号ID
    :param eos_id: 句子结束符号ID
    :return: 填充后的目标句子和源句子的mask
    """
    tgt_list = []
    
    # 插入 每个句子中的起始和结束标识    
    for _tgt in batch:
        processed_tgt = torch.cat([
            torch.tensor([eos_id]),  # 句子开始符号
            torch.tensor(_tgt, dtype=torch.int64),  # 目标句子
            torch.tensor([eos_id])   # 句子结束符号
        ])
        # F.pad 在 processed_tgt 后面填充 0，直到长度达到 max_padding
        processed_tgt = F.pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id)
        tgt_list.append(processed_tgt)
    
    # 堆叠成一个batch_tensor (list -> tensor[batch_size, seq_len])
    tgt_tensor = torch.stack(tgt_list)
    
    # tgt_tensor != pad_id -> mask -> 升维（-1 -> [batch_size, seq_len, 1]）
    tgt_mask = (tgt_tensor != pad_id).unsqueeze(-1)  # [batch_size, 1, seq_len]

    return tgt_tensor, tgt_mask

class MySelfAttentionWithpad(nn.Module):
    def __init__(self, dim_embedding, dim_qk, dim_v, mask=None):
        super(MySelfAttentionWithpad, self).__init__()
        self.dim_embedding = dim_embedding
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.mask = mask
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
        if self.mask is not None:
            # Apply mask to the score
            # mask: batch, 1, sequence_length, sequence_length
            score = score.masked_fill(self.mask == 0, float('-inf'))

        # Softmax( Q * K^T / sqrt(dim_embedding)) * V
        score = torch.softmax(score, dim=-1)  # batch, sequence_length, sequence_length
        att = torch.bmm(score, v)
        return att


def test_padding_mask_behavior():
    # Example sentences (randomized lengths and padding for illustration)
    batch = [
        [1, 2, 3, 4],  # Sentence 1
        [5, 6],         # Sentence 2
        [7, 8, 9, 10, 11],  # Sentence 3
    ]
    
    max_padding = 10  # Maximum padding length for this test
    pad_id = -1
    eos_id = 3
    
    # Step 1: Prepare the batch using collate_batch
    batch_tensor, mask_tensor = collate_batch(batch, max_padding, pad_id, eos_id)
    
    print("Batch Tensor:")
    print(batch_tensor)
    print("\nMask Tensor:")
    print(mask_tensor)
    
    # Step 2: Create an attention model
    dim_embedding = 5  # Arbitrary choice for the embedding dimension
    dim_qk = 4         # Arbitrary choice for query/key dimension
    dim_v = 4          # Arbitrary choice for value dimension
    
    attention_model = MySelfAttentionWithpad(dim_embedding, dim_qk, dim_v, mask_tensor)
    
    # Step 3: Generate random embeddings for the input (matching batch size and dim_embedding)
    batch_size, seq_len = batch_tensor.shape
    random_embeddings = torch.randn(batch_size, seq_len, dim_embedding)
    
    # Step 4: Forward pass through the attention layer
    attention_output = attention_model(random_embeddings)
    
    print("\nAttention Output:")
    print(attention_output)

# Run the test
test_padding_mask_behavior()



