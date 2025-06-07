import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    
    参数:
        scale: 缩放因子，通常为键向量维度的平方根
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        """
        前向传播
        
        参数:
            q: 查询向量 [batch_size, len_q, d_k]
            k: 键向量 [batch_size, len_k, d_k]
            v: 值向量 [batch_size, len_v, d_v]，其中 len_k == len_v
            mask: 掩码，用于屏蔽某些位置的注意力 [batch_size, len_q, len_k]
            
        返回:
            attn: 注意力权重 [batch_size, len_q, len_k]
            output: 注意力输出 [batch_size, len_q, d_v]
        """
        # 1. 计算注意力分数：Q和K的矩阵乘法
        # [batch_size, len_q, d_k] x [batch_size, d_k, len_k] -> [batch_size, len_q, len_k]
        attn_scores = torch.bmm(q, k.transpose(1, 2))
        
        # 2. 缩放注意力分数
        attn_scores = attn_scores / self.scale
        
        # 3. 应用掩码（如果提供）
        if mask is not None:
            # 将掩码位置的值设为负无穷，使 softmax 后的权重接近于 0
            attn_scores = attn_scores.masked_fill(mask, -np.inf)
        
        # 4. 应用 softmax 获取注意力权重
        attn = self.softmax(attn_scores)
        
        # 5. 计算输出：注意力权重与 V 的矩阵乘法
        # [batch_size, len_q, len_k] x [batch_size, len_v, d_v] -> [batch_size, len_q, d_v]
        output = torch.bmm(attn, v)
        
        return attn, output

# 示例参数
batch_size = 2
len_q = 3  # 查询序列长度
len_k = 4  # 键/值序列长度
d_k = 128  # 键向量维度
d_v = 64   # 值向量维度

# 创建随机输入
q = torch.randn(batch_size, len_q, d_k)
k = torch.randn(batch_size, len_k, d_k)
v = torch.randn(batch_size, len_k, d_v)

# 创建掩码（可选）
mask = torch.zeros(batch_size, len_q, len_k).bool()  # False 表示不掩码

# 初始化注意力机制
attention = ScaledDotProductAttention(scale=np.sqrt(d_k))

# 前向传播
attn, output = attention(q, k, v, mask)

print(f"注意力权重形状: {attn.shape}")  # [batch_size, len_q, len_k]
print(f"输出形状: {output.shape}")      # [batch_size, len_q, d_v]