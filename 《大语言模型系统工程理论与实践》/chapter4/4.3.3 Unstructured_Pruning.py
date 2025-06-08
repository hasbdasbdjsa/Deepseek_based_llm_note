import torch
import torch.nn as nn
import sys

# 尝试导入自定义模型
try:
    from model import is_pytorch_version_compatible, SimpleModel
    use_transformers = False
    
    # 如果PyTorch版本满足要求，尝试导入transformers
    if is_pytorch_version_compatible():
        try:
            from transformers import GPT2LMHeadModel
            use_transformers = True
            print("使用transformers库的预训练模型")
        except:
            print("无法导入transformers库，将使用自定义模型")
    else:
        print("PyTorch版本不满足要求，将使用自定义模型")
except:
    # 如果无法导入自定义模型，直接尝试导入transformers
    try:
        from transformers import GPT2LMHeadModel
        use_transformers = True
        print("使用transformers库的预训练模型")
    except:
        print("无法导入transformers库或自定义模型，程序无法继续")
        sys.exit(1)

# 示例代码
if __name__ == "__main__":
    print("非结构化剪枝示例")
    
    # 加载预训练模型
    if use_transformers:
        try:
            # 使用transformers的预训练模型
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            print("成功加载GPT2预训练模型")
        except Exception as e:
            print(f"无法加载预训练模型: {e}")
            print("切换到自定义模型")
            use_transformers = False
            model = SimpleModel(vocab_size=50257, hidden_size=768, num_layers=12)
            print("创建了自定义模型")
    else:
        # 使用自定义模型
        model = SimpleModel(vocab_size=50257, hidden_size=768, num_layers=12)
        print("创建了自定义模型")

    # 设置目标稀疏度
    target_sparsity = 0.8  # 80%的参数将被剪枝

    # 收集所有权重参数
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:  # 只考虑权重矩阵，不包括偏置
            all_weights.append(param.data.abs().view(-1))

    # 将所有权重展平并排序
    all_weights = torch.cat(all_weights)
    sorted_weights, _ = torch.sort(all_weights)

    # 确定阈值
    threshold_idx = int(len(sorted_weights) * target_sparsity)
    threshold = sorted_weights[threshold_idx]

    # 应用剪枝
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            # 创建掩码：小于阈值的权重设为0
            mask = (param.data.abs() > threshold).float()
            param.data.mul_(mask)  # 应用掩码

    # 保存稀疏模型
    try:
        torch.save(model.state_dict(), "gpt2_sparse.pt")
        print("模型已保存为 gpt2_sparse.pt")
    except Exception as e:
        print(f"保存模型失败: {e}")

    # 计算实际稀疏度
    zero_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            zero_count += (param.data == 0).sum().item()
            total_count += param.numel()

    actual_sparsity = zero_count / total_count
    print(f"实际稀疏度: {actual_sparsity:.4f} (目标: {target_sparsity:.4f})")
    print(f"模型权重总数: {total_count:,}")
    print(f"被剪枝(置零)的权重数: {zero_count:,}")
    print(f"保留的权重数: {total_count - zero_count:,}")
