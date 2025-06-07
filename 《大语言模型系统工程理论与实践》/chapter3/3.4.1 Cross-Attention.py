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
            v: 值向量 [batch_size, len_v, d_v]，其中len_k == len_v
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
            # 将掩码位置的值设为负无穷，使softmax后的权重接近于0
            attn_scores = attn_scores.masked_fill(mask, -np.inf)
        
        # 4. 应用softmax获取注意力权重
        attn = self.softmax(attn_scores)
        
        # 5. 计算输出：注意力权重与V的矩阵乘法
        # [batch_size, len_q, len_k] x [batch_size, len_v, d_v] -> [batch_size, len_q, d_v]
        output = torch.bmm(attn, v)
        
        return attn, output

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        d_k: 键向量维度（如果为None，则为d_model/n_heads）
        d_v: 值向量维度（如果为None，则为d_model/n_heads）
    """
    def __init__(self, d_model, n_heads, d_k=None, d_v=None):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        
        # 如果未指定d_k和d_v，则设为d_model/n_heads
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, self.d_k * n_heads, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k * n_heads, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v * n_heads, bias=False)
        self.W_o = nn.Linear(self.d_v * n_heads, d_model, bias=False)
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(scale=np.sqrt(self.d_k))
        
    def forward(self, q, k, v, mask=None):
        """
        前向传播
        
        参数:
            q: 查询向量 [batch_size, len_q, d_model]
            k: 键向量 [batch_size, len_k, d_model]
            v: 值向量 [batch_size, len_v, d_model]
            mask: 掩码 [batch_size, len_q, len_k]
            
        返回:
            output: 多头注意力输出 [batch_size, len_q, d_model]
            attn: 注意力权重 [batch_size, n_heads, len_q, len_k]
        """
        batch_size = q.size(0)
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)
        
        # 1. 线性投影并分割成多个头
        # [batch_size, len_q, d_model] -> [batch_size, len_q, n_heads * d_k] -> [batch_size, n_heads, len_q, d_k]
        q = self.W_q(q).view(batch_size, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, len_v, self.n_heads, self.d_v).transpose(1, 2)
        
        # 2. 调整掩码形状以适应多头
        if mask is not None:
            # [batch_size, len_q, len_k] -> [batch_size, 1, len_q, len_k] -> [batch_size, n_heads, len_q, len_k]
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # 3. 重塑张量以便于批量处理
        # [batch_size, n_heads, len_q/k/v, d_k/v] -> [batch_size * n_heads, len_q/k/v, d_k/v]
        q = q.reshape(-1, len_q, self.d_k)
        k = k.reshape(-1, len_k, self.d_k)
        v = v.reshape(-1, len_v, self.d_v)
        if mask is not None:
            mask = mask.reshape(-1, len_q, len_k)
        
        # 4. 应用缩放点积注意力
        attn, output = self.attention(q, k, v, mask)
        
        # 5. 重塑回多头形式
        # [batch_size * n_heads, len_q, d_v] -> [batch_size, n_heads, len_q, d_v]
        output = output.view(batch_size, self.n_heads, len_q, self.d_v)
        attn = attn.view(batch_size, self.n_heads, len_q, len_k)
        
        # 6. 连接多头输出并进行最终线性投影
        # [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads, d_v] -> [batch_size, len_q, n_heads * d_v]
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.W_o(output)
        
        return output, attn

class CrossAttention(nn.Module):
    """
    交叉注意力机制
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, n_heads)
        
    def forward(self, q, kv, mask=None):
        """
        前向传播
        
        参数:
            q: 查询序列 [batch_size, len_q, d_model]
            kv: 键值序列 [batch_size, len_kv, d_model]
            mask: 掩码 [batch_size, len_q, len_kv]
            
        返回:
            output: 交叉注意力输出 [batch_size, len_q, d_model]
            attn: 注意力权重 [batch_size, n_heads, len_q, len_kv]
        """
        # 在交叉注意力中，K和V来自同一个输入，Q来自另一个输入
        return self.multihead_attn(q, kv, kv, mask)

# 示例参数
batch_size = 2
len_q = 5  # 查询序列长度
len_kv = 10  # 键/值序列长度
d_model = 512  # 模型维度
n_heads = 8  # 注意力头数

# 创建随机输入
q = torch.randn(batch_size, len_q, d_model)  # 如解码器的输出
kv = torch.randn(batch_size, len_kv, d_model)  # 如编码器的输出

# 创建掩码（可选）
mask = torch.zeros(batch_size, len_q, len_kv).bool()  # False表示不掩码

# 初始化交叉注意力机制
cross_attn = CrossAttention(d_model=d_model, n_heads=n_heads)

# 前向传播
output, attn = cross_attn(q, kv, mask)

print(f"输出形状: {output.shape}")  # [batch_size, len_q, d_model]
print(f"注意力权重形状: {attn.shape}")  # [batch_size, n_heads, len_q, len_kv]
