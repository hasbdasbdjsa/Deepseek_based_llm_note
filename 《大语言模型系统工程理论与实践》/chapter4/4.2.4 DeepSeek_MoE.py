import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekMoERouter(nn.Module):
    """DeepSeek-MoE的分层路由器"""
    def __init__(self, input_size, num_experts, num_groups=4, capacity_factor=1.1):
        super(DeepSeekMoERouter, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_groups = num_groups
        self.experts_per_group = num_experts // num_groups
        self.capacity_factor = capacity_factor
        
        # 组路由器
        self.group_router = nn.Linear(input_size, num_groups, bias=False)
        
        # 专家路由器（每组一个）
        self.expert_routers = nn.ModuleList([
            nn.Linear(input_size, self.experts_per_group, bias=False)
            for _ in range(num_groups)
        ])
        
        # 复杂度估计器
        self.complexity_estimator = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 初始化路由器权重
        nn.init.zeros_(self.group_router.weight)
        for router in self.expert_routers:
            nn.init.zeros_(router.weight)
    
    def forward(self, x, is_training=True, min_experts=1, max_experts=2):
        batch_size, seq_len, _ = x.shape
        
        # 估计输入复杂度
        complexity_score = self.complexity_estimator(x).squeeze(-1)  # [batch_size, seq_len]
        num_active_experts = torch.clamp(
            (complexity_score * max_experts).int(),
            min=min_experts,
            max=max_experts
        )  # [batch_size, seq_len]
        
        # 第一阶段：选择组
        group_logits = self.group_router(x)  # [batch_size, seq_len, num_groups]
        group_probs = F.softmax(group_logits, dim=-1)  # [batch_size, seq_len, num_groups]
        
        # 创建分发张量和组合张量
        dispatch_tensor = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        combine_tensor = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        
        # 对每个token进行路由
        for i in range(batch_size):
            for j in range(seq_len):
                # 确定当前token需要激活的专家数量
                k = num_active_experts[i, j].item()
                
                # 选择Top-1组
                group_idx = torch.argmax(group_probs[i, j]).item()
                
                # 第二阶段：在选定的组内选择专家
                expert_logits = self.expert_routers[group_idx](x[i, j].unsqueeze(0))  # [1, experts_per_group]
                expert_probs = F.softmax(expert_logits, dim=-1)  # [1, experts_per_group]
                
                # 选择组内Top-k专家
                top_k_probs, top_k_indices = torch.topk(expert_probs, k=k, dim=-1)  # [1, k]
                
                # 归一化概率
                top_k_probs = top_k_probs / top_k_probs.sum()
                
                # 更新分发张量和组合张量
                for l in range(k):
                    expert_idx = group_idx * self.experts_per_group + top_k_indices[0, l].item()
                    dispatch_tensor[i, j, expert_idx] = 1.0
                    combine_tensor[i, j, expert_idx] = top_k_probs[0, l].item()
        
        # 计算负载均衡损失
        # 计算每个专家的分配概率（简化版）
        router_probs = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        for i in range(batch_size):
            for j in range(seq_len):
                group_idx = torch.argmax(group_probs[i, j]).item()
                expert_probs = F.softmax(self.expert_routers[group_idx](x[i, j].unsqueeze(0)), dim=-1)
                for l in range(self.experts_per_group):
                    expert_idx = group_idx * self.experts_per_group + l
                    router_probs[i, j, expert_idx] = group_probs[i, j, group_idx] * expert_probs[0, l]
        
        router_prob_per_expert = router_probs.sum(dim=[0, 1]) / (batch_size * seq_len)
        
        # 计算每个专家的实际使用率
        expert_usage = dispatch_tensor.sum(dim=[0, 1]) / (batch_size * seq_len)
        
        # 负载均衡损失
        aux_loss = (router_prob_per_expert * expert_usage).sum() * self.num_experts
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss

# 使用示例
if __name__ == "__main__":
    # 创建一个小型的测试样例
    batch_size = 4
    seq_len = 16
    input_size = 512
    num_experts = 8
    num_groups = 4
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 创建路由器
    router = DeepSeekMoERouter(input_size, num_experts, num_groups)
    
    # 前向传播
    dispatch, combine, probs, loss = router(x, min_experts=1, max_experts=2)
    
    print(f"输入形状: {x.shape}")
    print(f"分发张量形状: {dispatch.shape}")
    print(f"组合张量形状: {combine.shape}")
    print(f"路由概率形状: {probs.shape}")
    print(f"辅助损失: {loss.item():.4f}")
