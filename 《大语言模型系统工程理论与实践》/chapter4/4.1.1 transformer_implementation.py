import torch
import torch.nn as nn
import math

# (此处省略MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, 
#  EncoderLayer, DecoderLayer, Encoder, Decoder的详细代码，
#  这些已在 /home/ubuntu/chapter4/code_examples/transformer_implementation.py 中提供)

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
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        
        # 应用注意力权重
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.linear_out(out)
        
        return out, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :].transpose(0, 1)
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 从PyTorch内置模块创建Transformer
        # 注意：PyTorch的Transformer模块期望输入形状为 (seq_len, batch_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=num_heads, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=d_ff, 
            dropout=dropout,
            batch_first=True # 设置为True，则输入形状为 (batch_size, seq_len, d_model)
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        # src: [batch_size, src_seq_len]
        # tgt: [batch_size, tgt_seq_len]
        
        src_padding_mask = (src == 0) # 假设0是padding token
        tgt_padding_mask = (tgt == 0)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        src_embedded = self.pos_encoding(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_embedded = self.pos_encoding(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        
        # PyTorch Transformer模块期望 (seq_len, batch_size, feature) 如果 batch_first=False (默认)
        # 或者 (batch_size, seq_len, feature) 如果 batch_first=True
        # 此处我们已在初始化时设置 batch_first=True
        output = self.transformer(src_embedded, tgt_embedded, 
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=src_padding_mask,
                                  tgt_mask=tgt_mask)
        
        return self.fc_out(output)

# 使用示例
# src_vocab = 1000
# tgt_vocab = 1000
# d_model_size = 512
# model = Transformer(src_vocab, tgt_vocab, d_model=d_model_size)
# src_data = torch.randint(1, src_vocab, (32, 10)) # (batch_size, seq_len)
# tgt_data = torch.randint(1, tgt_vocab, (32, 12))
# out = model(src_data, tgt_data)
# print(out.shape) # torch.Size([32, 12, 1000])
