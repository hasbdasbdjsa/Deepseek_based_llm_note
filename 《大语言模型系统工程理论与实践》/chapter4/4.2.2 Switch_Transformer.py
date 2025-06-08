import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchRouter(nn.Module):
    """Switch Transformer的Top-1路由器"""
    def __init__(self, input_size, num_experts, capacity_factor=1.1, z_loss_coef=1e-3):
        super(SwitchRouter, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.z_loss_coef = z_loss_coef
        
        # 路由器权重
        self.router = nn.Linear(input_size, num_experts, bias=False)
        
        # 初始化路由器权重
        nn.init.zeros_(self.router.weight)
    
    def forward(self, x, is_training=True):
        batch_size, seq_len, _ = x.shape
        
        # 计算路由logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # 计算Z-loss
        z_loss = self.z_loss_coef * torch.square(torch.logsumexp(router_logits, dim=-1)).mean()
        
        # 计算路由概率
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # 选择概率最高的专家（Top-1路由）
        expert_index = torch.argmax(router_probs, dim=-1)  # [batch_size, seq_len]
        
        # 计算每个专家的容量
        tokens_per_batch = batch_size * seq_len
        capacity = int(tokens_per_batch * self.capacity_factor / self.num_experts)
        
        # 创建分发张量和组合张量
        dispatch_tensor = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        combine_tensor = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        
        # 为每个专家分配token，考虑容量限制
        for expert_idx in range(self.num_experts):
            # 找到分配给当前专家的token
            expert_indices = (expert_index == expert_idx).nonzero(as_tuple=True)
            expert_count = len(expert_indices[0])
            
            # 如果分配的token数量超过容量，随机选择一部分
            if expert_count > capacity:
                if is_training:
                    # 训练时：随机选择capacity个token
                    perm = torch.randperm(expert_count)
                    selected_indices = perm[:capacity]
                    expert_indices = tuple(idx[selected_indices] for idx in expert_indices)
                else:
                    # 推理时：选择概率最高的capacity个token
                    token_probs = router_probs[expert_indices[0], expert_indices[1], expert_idx]
                    _, sorted_indices = torch.sort(token_probs, descending=True)
                    selected_indices = sorted_indices[:capacity]
                    expert_indices = tuple(idx[selected_indices] for idx in expert_indices)
            
            # 更新分发张量和组合张量
            if len(expert_indices[0]) > 0:
                dispatch_tensor[expert_indices[0], expert_indices[1], expert_idx] = 1.0
                combine_tensor[expert_indices[0], expert_indices[1], expert_idx] = 1.0
        
        # 计算负载均衡损失
        # 计算每个专家的分配概率
        router_prob_per_expert = router_probs.sum(dim=[0, 1]) / (batch_size * seq_len)
        # 计算每个专家的实际使用率
        expert_usage = dispatch_tensor.sum(dim=[0, 1]) / (batch_size * seq_len)
        # 负载均衡损失
        load_balancing_loss = (router_prob_per_expert * expert_usage).sum() * self.num_experts
        
        # 总辅助损失
        aux_loss = load_balancing_loss + z_loss
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss

# 使用示例
if __name__ == "__main__":
    # 创建一个小型的测试样例
    batch_size = 4
    seq_len = 16
    input_size = 512
    num_experts = 8
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 创建路由器
    router = SwitchRouter(input_size, num_experts)
    
    # 前向传播
    dispatch, combine, probs, loss = router(x)
    
    print(f"输入形状: {x.shape}")
    print(f"分发张量形状: {dispatch.shape}")
    print(f"组合张量形状: {combine.shape}")
    print(f"路由概率形状: {probs.shape}")
    print(f"辅助损失: {loss.item():.4f}")
