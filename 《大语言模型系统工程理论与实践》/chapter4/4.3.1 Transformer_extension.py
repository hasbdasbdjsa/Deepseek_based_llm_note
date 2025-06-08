import torch
import torch.nn as nn
import sys

# 尝试导入自定义模型
try:
    from model import is_pytorch_version_compatible, SimpleConfig, calculate_params
    use_transformers = False
    
    # 如果PyTorch版本满足要求，尝试导入transformers
    if is_pytorch_version_compatible():
        try:
            from transformers import GPT2Config, GPT2Model
            use_transformers = True
            print("使用transformers库的预训练模型")
        except:
            print("无法导入transformers库，将使用自定义模型")
    else:
        print("PyTorch版本不满足要求，将使用自定义模型")
except:
    # 如果无法导入自定义模型，直接尝试导入transformers
    try:
        from transformers import GPT2Config, GPT2Model
        use_transformers = True
        print("使用transformers库的预训练模型")
    except:
        print("无法导入transformers库或自定义模型，程序无法继续")
        sys.exit(1)

# 定义不同规模的模型配置
if use_transformers:
    configs = {
        "small": GPT2Config(
            n_layer=6,        # 深度：6层
            n_head=8,         # 8个注意力头
            n_embd=512,       # 隐藏层维度：512
            n_ctx=1024        # 上下文窗口：1024
        ),
        "medium": GPT2Config(
            n_layer=12,       # 深度：12层
            n_head=12,        # 12个注意力头
            n_embd=768,       # 隐藏层维度：768
            n_ctx=2048        # 上下文窗口：2048
        ),
        "large": GPT2Config(
            n_layer=24,       # 深度：24层
            n_head=16,        # 16个注意力头
            n_embd=1024,      # 隐藏层维度：1024
            n_ctx=4096        # 上下文窗口：4096
        ),
        "xl": GPT2Config(
            n_layer=36,       # 深度：36层
            n_head=20,        # 20个注意力头
            n_embd=1280,      # 隐藏层维度：1280
            n_ctx=8192        # 上下文窗口：8192
        )
    }
    
    # 创建不同规模的模型
    try:
        models = {size: GPT2Model(config) for size, config in configs.items()}
        print("成功创建所有模型")
    except Exception as e:
        print(f"无法创建模型: {e}")
        print("切换到自定义模型")
        use_transformers = False
        from model import SimpleConfig, calculate_params
else:
    # 使用自定义模型配置
    configs = {
        "small": SimpleConfig(
            n_layer=6,        # 深度：6层
            n_head=8,         # 8个注意力头
            n_embd=512,       # 隐藏层维度：512
            n_ctx=1024        # 上下文窗口：1024
        ),
        "medium": SimpleConfig(
            n_layer=12,       # 深度：12层
            n_head=12,        # 12个注意力头
            n_embd=768,       # 隐藏层维度：768
            n_ctx=2048        # 上下文窗口：2048
        ),
        "large": SimpleConfig(
            n_layer=24,       # 深度：24层
            n_head=16,        # 16个注意力头
            n_embd=1024,      # 隐藏层维度：1024
            n_ctx=4096        # 上下文窗口：4096
        ),
        "xl": SimpleConfig(
            n_layer=36,       # 深度：36层
            n_head=20,        # 20个注意力头
            n_embd=1280,      # 隐藏层维度：1280
            n_ctx=8192        # 上下文窗口：8192
        )
    }

# 计算各模型的参数量
if use_transformers:
    # 使用transformers模型的方法计算参数量
    for size, model in models.items():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{size.upper()} model parameters: {num_params:,}")
else:
    # 使用自定义方法计算参数量
    for size, config in configs.items():
        params = calculate_params(config)
        print(f"{size.upper()} model parameters: {params:,}")

# 分析参数量与各因素的关系
for size, config in configs.items():
    # 每层参数量（近似值）
    params_per_layer = 12 * config.n_embd**2  # 12是一个近似系数，实际更复杂
    
    # 总参数量
    total_params = params_per_layer * config.n_layer
    
    print(f"{size.upper()} analysis:")
    print(f"  - Depth contribution: {config.n_layer / configs['small'].n_layer:.1f}x")
    print(f"  - Width contribution: {(config.n_embd / configs['small'].n_embd)**2:.1f}x")
    print(f"  - Combined scaling: {total_params / (params_per_layer * configs['small'].n_layer):.1f}x")
    print()
