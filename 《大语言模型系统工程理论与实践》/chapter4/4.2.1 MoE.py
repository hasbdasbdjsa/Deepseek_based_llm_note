import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    """单个专家网络，实现为简单的前馈网络"""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(ExpertLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MoELayer(nn.Module):
    """混合专家模型层"""
    def __init__(self, input_size, output_size, num_experts, hidden_size, k=1, capacity_factor=1.0, dropout=0.1):
        super(MoELayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            ExpertLayer(input_size, hidden_size, output_size, dropout)
            for _ in range(num_experts)
        ])
        
        # 路由器
        self.router = nn.Linear(input_size, num_experts, bias=False)
        
        # 初始化路由器权重
        nn.init.zeros_(self.router.weight)
    
    def forward(self, x, is_training=True):
        batch_size, seq_len, _ = x.shape
        
        # 计算路由概率
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # 在训练时添加噪声以提高稳定性
        if is_training:
            router_logits += torch.randn_like(router_logits) * 0.1
        
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # 选择Top-k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.k, dim=-1)
        
        # 归一化Top-k概率
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 计算每个专家的容量
        tokens_per_batch = batch_size * seq_len
        capacity = int(tokens_per_batch * self.capacity_factor * self.k / self.num_experts)
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 计算负载均衡损失
        # 计算每个专家的分配概率
        router_prob_per_expert = router_probs.sum(dim=[0, 1]) / (batch_size * seq_len)
        # 计算每个专家的实际使用率
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        
        # 对每个专家进行计算
        for expert_idx in range(self.num_experts):
            # 找到分配给当前专家的token
            indices = []
            for b in range(batch_size):
                for s in range(seq_len):
                    for k in range(self.k):
                        if top_k_indices[b, s, k].item() == expert_idx:
                            indices.append((b, s, k))
            
            # 更新专家使用率
            expert_usage[expert_idx] = len(indices) / (batch_size * seq_len * self.k)
            
            # 如果没有token分配给当前专家，跳过
            if not indices:
                continue
            
            # 如果分配的token数量超过容量，随机选择一部分
            if len(indices) > capacity:
                indices = [indices[i] for i in torch.randperm(len(indices))[:capacity]]
            
            # 提取分配给当前专家的输入
            expert_inputs = torch.stack([x[b, s] for b, s, _ in indices])
            
            # 计算专家输出
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # 将专家输出分配回原始位置
            for idx, (b, s, k) in enumerate(indices):
                output[b, s] += expert_output[idx] * top_k_probs[b, s, k]
        
        # 计算负载均衡损失
        load_balancing_loss = (router_prob_per_expert * expert_usage).sum() * self.num_experts
        
        return output, load_balancing_loss

# 使用示例
if __name__ == "__main__":
    # 创建一个小型的测试样例
    batch_size = 4
    seq_len = 16
    input_size = 512
    output_size = 512
    hidden_size = 1024
    num_experts = 8
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 创建MoE层
    moe_layer = MoELayer(input_size, output_size, num_experts, hidden_size, k=2)
    
    # 前向传播
    output, loss = moe_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"负载均衡损失: {loss.item():.4f}")
