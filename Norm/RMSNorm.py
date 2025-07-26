import torch
from torch import nn

class MyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        """
        MyRMSNorm 初始化
        
        公式：
          γ: 可学习的缩放参数 (self.weight)
          ε: 数值稳定系数 (self.eps)
        
        参数：
          dim: 输入特征维度
          eps: 防止除零的小常数 (默认 1e-6)
        """
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        """
        核心归一化计算
        
        公式：
          RMS(x) = √(1/dim * Σx_i²)   [1,4](@ref)
          x̂ = x / √(RMS(x)² + ε)      [2,6](@ref)
        实现说明：
        1. mean(-1, keepdim=True)
            在归一化操作中，我们需要对每个样本（包括批次和序列位置）在特征维度（即hidden_dim）上计算均方根值
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


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
    ln = MyRMSNorm(x.shape[2])

    rms_norm = nn.RMSNorm(x.shape[2], eps=1e-6)  # 创建RMSNorm实例
    realx = rms_norm(x)
    # 前向传播
    x = ln(x)
    print(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print(f"batch {i}, channel {j}: {x[i,j,:]}")
            for k in range(x.shape[2]):
                if x[i,j,k] != realx[i,j,k]:
                    print(f"Mismatch at batch {i}, channel {j}, position {k}:")
                    print(f"    Expected: {realx[i,j,k]}")
                    print(f"    Got: {x[i,j,k]}")
                    exit(1)
    print("RMSNorm output matches expected output.")
    print("RMSNorm output:", x)
    print("RMSNorm output:", realx)
