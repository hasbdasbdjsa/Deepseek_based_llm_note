import torch
import torch.nn as nn
import math


def rotate_half(x):
    """将输入张量的后一半维度旋转"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 线性变换层
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        # 初始化RoPE参数
        theta = 10000.0 ** (-2 * torch.arange(0, self.head_dim // 2, dtype=torch.float32) / self.head_dim)
        self.register_buffer('theta', theta)

    def get_rotary_matrix(self, seq_len, device):
        """生成旋转矩阵的cos和sin分量"""
        m = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(m, self.theta.to(device))  # (seq_len, head_dim//2)
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rope(self, x, cos, sin):
        """应用旋转位置编码"""
        cos = cos.repeat_interleave(2, dim=-1)  # 扩展维度 (seq_len, head_dim)
        sin = sin.repeat_interleave(2, dim=-1)

        # 调整维度用于广播 (1, seq_len, 1, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return x * cos + rotate_half(x) * sin

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 线性投影
        q = self.wq(x)  # (B, L, d_model)
        k = self.wk(x)
        v = self.wv(x)

        # 拆分为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 生成RoPE矩阵
        cos, sin = self.get_rotary_matrix(seq_len, x.device)

        # 应用RoPE
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)

        # 应用注意力到value
        output = torch.matmul(attn, v)

        # 合并多头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)

        return self.wo(output)


# 测试示例
if __name__ == "__main__":
    # 超参数
    d_model = 64
    num_heads = 4
    seq_len = 10
    batch_size = 2

    # 初始化模型
    model = MultiHeadAttentionWithRoPE(d_model, num_heads)

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("\n前向传播测试通过!")

    # 梯度测试
    target = torch.randn_like(output)
    loss = torch.nn.MSELoss()(output, target)
    loss.backward()
    print("\n梯度反向传播测试通过!")