 ## 多查询注意力
import torch
from torch import nn
class MyMultiQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MyMultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        ## 初始化Q、K、V投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)

        """
        共享 k_i, v_i head (b, seq_len, d_k) torch.Size([2, 5, 8])
        -> 用于每次计算注意力分数
        """
        self.k_linear = nn.Linear(hidden_size, self.head_dim)
        self.v_linear = nn.Linear(hidden_size, self.head_dim) ###
        
        ## 输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)
        
        """
        对Q,K,V进行分割
            Q -> Q1...Qh 
            K -> K1 (核心)
        eg.
            Q (b, seq_len, h * d_k) -> Q (b, h, seq_len, d_k) 
            K (b, seq_len, 1 * d_k) -> K (b, 1, seq_len, d_k) (核心)   
        """ 
        query = self.split_head(query)
        key = self.split_head(key, 1)
        value = self.split_head(value, 1)
        
        """
        K (b, seq_len, 1 * d_k) -> K (b, h, seq_len, d_k) = [k1, k1, ..., k1] (核心)
        V (b, seq_len, 1 * d_k) -> V (b, h, seq_len, d_k) = [v1, v1, ..., v1] (核心)
        这里的 K 和 V 是共享的，只有 Q 是分开的
        eg. 
            torch.Size([2, 1, 5, 8]) -> torch.Size([2, 8, 5, 8])
        """
        key = key.expand(-1, self.num_heads, -1, -1)        
        value = value.expand(-1, self.num_heads, -1, -1) 
        
        ## 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9
        
        ## 对注意力分数进行归一化
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        output = torch.matmul(attention_probs, value)
        
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        
        output = self.o_linear(output)
        
        return output       
        
    def split_head(self, x, head_num=None):
        
        batch_size = x.size()[0]
        
        if head_num == None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        else:
            return x.view(batch_size, -1, head_num, self.head_dim).transpose(1,2)

if __name__ == "__main__":
    # Create random input
    batch_size = 2
    seq_len = 5
    hidden_size = 64
    num_heads = 8

    x = torch.randn(batch_size, seq_len, hidden_size)
    attention = MyMultiQueryAttention(hidden_size, num_heads)
    out = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)