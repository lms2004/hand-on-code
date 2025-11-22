import torch
from torch import nn


class LN(nn.Module):
    # 初始化
    def __init__(self, normalized_shape,  
                 eps:float = 1e-5,  # 小常数，防止除零
                 elementwise_affine:bool = True):  
        super(LN, self).__init__()
        """
        初始化参数：
            1. eps：ε
            2. normalized_shape：需要对哪个维度的特征做LN
            3. elementwise_affine：是否需要可训练的缩放因子和偏置
        """
        # 需要对哪个维度的特征做LN, torch.size查看维度
        self.normalized_shape = normalized_shape  # [c,w*h]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # 构造可训练的缩放因子和偏置
        if self.elementwise_affine:  
            """
            gain: 
            -> 缩放参数 γ (与输入特征形状相同)
            bias:
            -> 偏移参数 β (与输入特征形状相同)
            """
            self.gain = nn.Parameter(torch.ones(normalized_shape))  # [c,w*h]
            self.bias = nn.Parameter(torch.zeros(normalized_shape))  # [c,w*h]
 
    # 前向传播
    def forward(self, x: torch.Tensor): # [b,c,w*h]
        """
        Layer Normalization (完整公式实现)
        计算公式：
            LayerNorm(x) = (x - μ) / sqrt(σ² + ε) * γ + β
                         = [(x - E[x]) / √(Var[x])] * γ + β
            其中：
                Var[x] = σ² + ε  
                σ² = E[X^2] - (E[X])^2 
        """
        # 需要做LN的维度和输入特征图对应维度的shape相同
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]  # [-2:]


        # 需要做LN的维度索引
        # -> [b,c,w*h] 维度上取 [-1,-2] 维度，即[c,w*h]
        dims = [-(i+1) for i in range(len(self.normalized_shape))]  
        
        # 计算特征图对应维度的均值和方差
        """"
        mean()
            dim: ( int or tuple of ints )
            Returns the mean value of each row of the input tensor in the given dimension dim
            eg.
                x.mean(dim=[-2,-1], keepdims=True)
                    == 
                (x.mean(dim=-2, keepdims=True)).mean(dim=-1, keepdims=True)
        """
        mean = x.mean(dim=dims, keepdims=True)      # μ = E[X] -> [b,1,1]
        mean_x2 = (x**2).mean(dim=dims, keepdims=True)  # E[X^2] -> [b,1,1]
        var = mean_x2 - mean**2                      # σ² = E[X^2] - (E[X])^2 

        # 计算 (x - μ) / sqrt(σ² + ε)
        x_norm = (x-mean) / torch.sqrt(var+self.eps)  # x : [b,c,w*h] -> x_norm : [b,c,w*h] 

        # 线性变换 x_norm * γ + β
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias  # [b,c,w*h]
        return x_norm
 
if __name__ == '__main__':
 
    x = torch.linspace(0, 23, 24, dtype=torch.float32)  # 构造输入层
    x = x.reshape([2,3,2*2])  # [b,c,w*h]
    """
    张量结构详解：
    - 输入x的原始形状[24]被重塑为[2,3,4]的三维结构
    - 维度的具体含义：
        dim=0 (size=2): batch维度，表示2个独立样本/图像
        dim=1 (size=3): channel维度，表示每个样本有3个特征通道
        dim=2 (size=4): 空间维度，将2×2的空间区域展平为4个元素

    示例数据分布：
        batch 0: 
            channel 0: [0,1,2,3]   channel 1: [4,5,6,7]   channel 2: [8,9,10,11]
        batch 1:
            channel 0: [12,13,14,15] channel 1: [16,17,18,19] channel 2: [20,21,22,23]
    """
    # 实例化
    ln = LN(x.shape[1:])
    # 前向传播
    x = ln(x)
    print(x.shape)