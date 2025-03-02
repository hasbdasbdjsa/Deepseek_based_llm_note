import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        self.attn_original = attn.clone()  # 保存原始注意力分数
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, self.attn_original, attn  # 返回原始分数和归一化后的权重


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn_original, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn_original, attn

def main():
    # 定义超参数
    d_model = 512
    n_head = 8
    d_k = d_v = d_model // n_head
    batch_size = 2
    len_q = len_k = len_v = 4  # 设置为 4，与 mask 示例一致

    # 创建 MultiHeadAttention 模型
    mha = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=0.1)

    # 手动构造输入数据
    q = torch.randn(batch_size, len_q, d_model)
    k = torch.randn(batch_size, len_k, d_model)
    v = torch.randn(batch_size, len_v, d_model)

    # 创建掩码（上三角矩阵）
    mask = torch.triu(torch.ones(len_q, len_k), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展为 [batch_size, len_q, len_k]

    # 前向传播
    output, attn_original, attn_weights = mha(q, k, v, mask=mask)

    # 打印输入和输出张量的形状
    print("Input Q shape:", q.shape)
    print("Input K shape:", k.shape)
    print("Input V shape:", v.shape)
    print("Output shape:", output.shape)

    # 打印掩码矩阵
    print("\nMask matrix:")
    print(mask[0].int())

    # 打印原始的注意力分数矩阵
    print("\nOriginal Attention Scores matrix:")
    print(attn_original[0].detach().numpy())

    # 打印应用 mask 后的注意力分数矩阵
    print("\nAttention Scores after applying mask:")
    # 模拟 mask 应用后的注意力分数
    masked_scores = attn_original.detach().clone()
    masked_scores[0].masked_fill_(mask[0].unsqueeze(0), float('-inf'))
    print(masked_scores.numpy())

    # 打印 softmax 归一化后的掩码注意力分数矩阵
    print("\nAttention weights after softmax:")
    print(attn_weights[0].detach().numpy())

    # 打印多头注意力输出示例
    print("\nMulti-Head Attention Output:")
    print(output.detach().numpy())

    # 可视化注意力权重矩阵
    plt.figure(figsize=(10, 6))
    plt.imshow(attn_weights[0, 0, :, :].detach().numpy(), cmap='viridis')
    plt.title("Attention Weights (Head 0, Batch 0)")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
