import torch
import torch.nn as nn
import sys

# 尝试导入自定义模型
try:
    from model import is_pytorch_version_compatible, SimpleAttentionModel
    use_transformers = False
    
    # 如果PyTorch版本满足要求，尝试导入transformers
    if is_pytorch_version_compatible():
        try:
            from transformers import BertModel, BertConfig
            use_transformers = True
            print("使用transformers库的预训练模型")
        except:
            print("无法导入transformers库，将使用自定义模型")
    else:
        print("PyTorch版本不满足要求，将使用自定义模型")
except:
    # 如果无法导入自定义模型，直接尝试导入transformers
    try:
        from transformers import BertModel, BertConfig
        use_transformers = True
        print("使用transformers库的预训练模型")
    except:
        print("无法导入transformers库或自定义模型，程序无法继续")
        sys.exit(1)

# 示例代码
if __name__ == "__main__":
    print("结构化剪枝示例")
    
    if use_transformers:
        try:
            # 加载预训练模型
            model = BertModel.from_pretrained("bert-base-uncased")
            print("成功加载BERT预训练模型")
        except Exception as e:
            print(f"无法加载预训练模型: {e}")
            print("切换到自定义模型")
            use_transformers = False
            model = SimpleAttentionModel(hidden_size=768, num_heads=12, num_layers=12)
            print("创建了自定义模型")
    else:
        # 使用自定义模型
        model = SimpleAttentionModel(hidden_size=768, num_heads=12, num_layers=12)
        print("创建了自定义模型")
    
    # 创建一个简单的示例输入
    batch_size = 2
    seq_len = 10
    hidden_dim = 768
    inputs = torch.randn(batch_size, seq_len, hidden_dim)

# 计算每个注意力头的重要性
importance_scores = []

# 为了示例，我们使用随机生成的重要性分数
for layer_idx in range(model.config.num_hidden_layers):
    # 为每个注意力头生成随机重要性分数
    head_importance = torch.rand(model.config.num_attention_heads)
    importance_scores.append(head_importance)

# 将所有层的重要性分数拼接起来
all_importance = torch.cat(importance_scores)

# 确定要剪枝的头（保留重要性最高的70%）
num_heads_to_keep = int(len(all_importance) * 0.7)
threshold = torch.sort(all_importance)[0][len(all_importance) - num_heads_to_keep]

# 创建掩码
head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads)
for layer_idx in range(model.config.num_hidden_layers):
    for head_idx in range(model.config.num_attention_heads):
        if importance_scores[layer_idx][head_idx] < threshold:
            head_mask[layer_idx][head_idx] = 0

# 使用掩码进行推理
def forward_with_mask(model, inputs, head_mask):
    if use_transformers:
        # 使用transformers模型的方式
        outputs = model(inputs, head_mask=head_mask)
    else:
        # 使用自定义模型的方式
        hidden_states = inputs
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        
        for layer_idx, layer in enumerate(model.encoder):
            # 自注意力
            query = layer['attention']['self']['query'](hidden_states)
            key = layer['attention']['self']['key'](hidden_states)
            value = layer['attention']['self']['value'](hidden_states)
            
            # 重塑为[batch_size, seq_len, num_heads, head_dim]形状，以便应用掩码
            head_dim = model.hidden_size // model.num_heads
            query = query.view(batch_size, seq_len, model.num_heads, head_dim)
            key = key.view(batch_size, seq_len, model.num_heads, head_dim)
            value = value.view(batch_size, seq_len, model.num_heads, head_dim)
            
            # 应用掩码
            for head_idx in range(model.num_heads):
                if head_mask[layer_idx, head_idx] == 0:
                    query[:, :, head_idx] = 0
                    key[:, :, head_idx] = 0
                    value[:, :, head_idx] = 0
            
            # 重塑回原始形状
            query = query.reshape(batch_size, seq_len, model.hidden_size)
            key = key.reshape(batch_size, seq_len, model.hidden_size)
            value = value.reshape(batch_size, seq_len, model.hidden_size)
            
            # 简化的注意力计算
            attn_output = layer['attention']['output'](value)
            hidden_states = hidden_states + attn_output
            
            # FFN
            intermediate = layer['intermediate'](hidden_states)
            output = layer['output'](intermediate)
            hidden_states = hidden_states + output
        
        outputs = type('SimpleOutput', (), {'last_hidden_state': hidden_states})
    
    return outputs

# 测试剪枝后的模型
with torch.no_grad():
    if use_transformers:
        try:
            # 使用transformers模型测试
            batch = {"input_ids": torch.randint(0, 30522, (batch_size, seq_len))}
            outputs = forward_with_mask(model, **batch, head_mask=head_mask)
        except Exception as e:
            print(f"使用预训练模型测试失败: {e}")
            print("改用自定义模型方式测试")
            outputs = forward_with_mask(model, inputs, head_mask)
    else:
        # 使用自定义模型测试
        outputs = forward_with_mask(model, inputs, head_mask)

print(f"原始模型注意力头数量: {model.config.num_hidden_layers * model.config.num_attention_heads}")
print(f"剪枝后保留的注意力头数量: {int(head_mask.sum().item())}")
print(f"剪枝率: {1 - head_mask.sum().item() / (model.config.num_hidden_layers * model.config.num_attention_heads):.2f}")
