import torch
from torch import nn
class MyMultiHeadAttention_(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MyMultiHeadAttention_, self).__init__()
        """
            hidden_size: d_model 模型维度
            nums_heads: h 多头注意力头数
            head_dim： d_k 每个头的维度
        """
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        """
        W_Q = [W_Q1 ... W_Qh]
        """
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        ## 输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, attention_mask=None):
        """
            hidden_state: X [batch_size, seq_len, d_model]
        """
        batch_size = hidden_state.size()[0]
        
        """
        拼接权重矩阵 W_Q = [W_Q1 ... W_Qh] -> 全连接层( 可学习参数 )
        
        eg.
            输入 X (batch_size, seq_len, d_model) 
            -> 
            X * W_Q 
            -> 
            输出 Q (batch_size,seq_len, h * d_k) [ h * d_k = d_model ]
        """
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        """
        对Q,K,V进行分割
            Q -> Q1...Qh 
            K -> K1...Kh
        eg.
            Q (b, seq_len, h * d_k) -> Q (b, h, seq_len, d_k) 参考笔记公式    
        """ 
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)
        
        """
        计算注意力分数
            Q * K^T / sqrt(d_k)
        eg.
            Q (b, h, seq_len, d_k)
            *
            K^T (b, h, d_k, seq_len)
        """
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        
        """
        掩码注意力
        -> 需要隐藏掉的元素会由 一个极小值 -1e9 替代
        """
        if attention_mask != None:
            attention_scores += attention_mask * -1e9
        
        ## 对注意力分数进行归一化
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        output = torch.matmul(attention_probs, value)

        """
        output 拼接
        (b, h, seq_len, d_k)  torch.Size([2, 8, 5, 8])
        (b, seq_len, h, d_k)  torch.Size([2, 5, 8, 8])
        (b, seq_len, h * d_k) torch.Size([2, 5, 64])
        """
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        
        output = self.o_linear(output)
        return output

    def split_head(self, x):
        """
        直接用 .view(batch_size, num_heads, -1, head_dim) 
        -> 会让 PyTorch 把内存按错误顺序切分，导致数据错乱。
        eg.
            PyTorch会尝试按照这个顺序切分内存：
                第一个维度是 batch_size
                第二个维度是 num_heads
                第三个维度是 -1（自动推断）
                第四个是 head_dim
        """
        batch_size = x.size()[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
    
if __name__ == "__main__":
    # Create random input
    batch_size = 2
    seq_len = 5
    hidden_size = 64
    num_heads = 8

    x = torch.randn(batch_size, seq_len, hidden_size)
    attention = MyMultiHeadAttention_(hidden_size, num_heads)
    out = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
