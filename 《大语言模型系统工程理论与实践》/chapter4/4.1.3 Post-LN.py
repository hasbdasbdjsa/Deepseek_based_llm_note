import torch
import torch.nn as nn
import sys
import os

# 可以选择使用自定义的transformer模块
try:
    from transformer import MultiHeadAttention, LayerNorm, FeedForward as PositionwiseFeedForward
    print("成功导入自定义transformer模块")
except ImportError:
    print("使用本地定义的组件")

# 如果没有成功导入transformer模块，使用以下本地定义
if 'MultiHeadAttention' not in globals():
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads, dropout=0.1):
            super(MultiHeadAttention, self).__init__()
            assert d_model % num_heads == 0
            self.d_k = d_model // num_heads
            self.num_heads = num_heads
            self.linear_q = nn.Linear(d_model, d_model)
            self.linear_k = nn.Linear(d_model, d_model)
            self.linear_v = nn.Linear(d_model, d_model)
            self.linear_out = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)
            
            # 线性映射
            query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            
            # 注意力计算
            scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            
            # 应用注意力权重
            out = torch.matmul(attn, value)
            out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
            out = self.linear_out(out)
            
            return out, attn

if 'PositionwiseFeedForward' not in globals():
    class PositionwiseFeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super(PositionwiseFeedForward, self).__init__()
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            return self.w_2(self.dropout(torch.relu(self.w_1(x))))

if 'LayerNorm' not in globals():
    class LayerNorm(nn.Module):
        def __init__(self, features, eps=1e-6):
            super(LayerNorm, self).__init__()
            self.a_2 = nn.Parameter(torch.ones(features))
            self.b_2 = nn.Parameter(torch.zeros(features))
            self.eps = eps
            
        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# PyTorch中EncoderLayer的简化示例 (Post-LN，与原始Transformer一致)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 使用自定义的LayerNorm如果可用，否则使用PyTorch的LayerNorm
        if 'LayerNorm' in globals() and not isinstance(LayerNorm, type(nn.LayerNorm)):
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
        else:
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output)) # Post-LN
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output)) # Post-LN
        return x

# Pre-LN的实现方式通常如下：
class EncoderLayerPreLN(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayerPreLN, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 使用自定义的LayerNorm如果可用，否则使用PyTorch的LayerNorm
        if 'LayerNorm' in globals() and not isinstance(LayerNorm, type(nn.LayerNorm)):
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, mask):
        # 层归一化 + 多头自注意力 + 残差连接
        norm_x = self.norm1(x) # Pre-LN
        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)
        
        # 层归一化 + 前馈网络 + 残差连接
        norm_x = self.norm2(x) # Pre-LN
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        return x

# 示例使用
if __name__ == "__main__":
    # 创建一个小型的测试样例
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    
    # 测试Post-LN
    post_ln_layer = EncoderLayer(d_model, num_heads, d_ff)
    post_ln_output = post_ln_layer(x, mask)
    print("Post-LN output shape:", post_ln_output.shape)
    
    # 测试Pre-LN
    pre_ln_layer = EncoderLayerPreLN(d_model, num_heads, d_ff)
    pre_ln_output = pre_ln_layer(x, mask)
    print("Pre-LN output shape:", pre_ln_output.shape)
