import torch
import torch.nn as nn
import math

# 检查PyTorch版本是否满足transformers库要求
def is_pytorch_version_compatible():
    try:
        import torch
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        return major >= 2 and minor >= 1
    except:
        return False

# 通用配置类
class SimpleConfig:
    def __init__(self, n_layer, n_head, n_embd, n_ctx=1024, vocab_size=50257):
        self.n_layer = n_layer        # 层数
        self.n_head = n_head          # 注意力头数
        self.n_embd = n_embd          # 隐藏层维度
        self.n_ctx = n_ctx            # 上下文长度
        self.vocab_size = vocab_size  # 词表大小
        self.expansion_factor = 4     # FFN扩展因子
        self.num_hidden_layers = n_layer  # 兼容BERT配置
        self.num_attention_heads = n_head  # 兼容BERT配置

# GPT2风格的简单模型
class SimpleModel(nn.Module):
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12):
        super(SimpleModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # 创建多个线性层来模拟Transformer的结构
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': nn.Linear(hidden_size, hidden_size),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
            })
            self.layers.append(layer)
        
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids=None):
        if input_ids is None:
            input_ids = torch.randint(0, 50257, (4, 10))
            
        hidden_states = self.embeddings(input_ids)
        
        for layer in self.layers:
            attn_output = layer['attention'](hidden_states)
            hidden_states = hidden_states + attn_output
            
            mlp_output = layer['mlp'](hidden_states)
            hidden_states = hidden_states + mlp_output
        
        logits = self.lm_head(hidden_states)
        
        # 返回一个类似Hugging Face模型的输出对象
        return type('SimpleModelOutput', (), {'logits': logits, 'last_hidden_state': hidden_states})

# BERT风格的简单注意力模型
class SimpleAttentionModel(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, num_layers=12, vocab_size=30522):
        super(SimpleAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_attention_heads = num_heads  # 兼容示例代码
        self.head_dim = hidden_size // num_heads
        self.vocab_size = vocab_size
        
        # 嵌入层
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # 创建encoder层
        self.encoder = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder.append(self._create_layer())
        
        # 配置对象
        self.config = SimpleConfig(
            n_layer=num_layers,
            n_head=num_heads,
            n_embd=hidden_size,
            vocab_size=vocab_size
        )
    
    def _create_layer(self):
        layer = nn.ModuleDict({
            'attention': nn.ModuleDict({
                'self': nn.ModuleDict({
                    'query': nn.Linear(self.hidden_size, self.hidden_size),
                    'key': nn.Linear(self.hidden_size, self.hidden_size),
                    'value': nn.Linear(self.hidden_size, self.hidden_size)
                }),
                'output': nn.Linear(self.hidden_size, self.hidden_size)
            }),
            'intermediate': nn.Linear(self.hidden_size, self.hidden_size * 4),
            'output': nn.Linear(self.hidden_size * 4, self.hidden_size)
        })
        return layer
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, head_mask=None):
        if input_ids is None:
            input_ids = torch.randint(0, self.vocab_size, (2, 10))
            
        hidden_states = self.embeddings(input_ids)
        
        for i, layer in enumerate(self.encoder):
            # 应用head_mask（如果提供）
            layer_head_mask = None
            if head_mask is not None:
                layer_head_mask = head_mask[i]
            
            # 自注意力
            query = layer['attention']['self']['query'](hidden_states)
            key = layer['attention']['self']['key'](hidden_states)
            value = layer['attention']['self']['value'](hidden_states)
            
            # 添加输出属性，供结构化剪枝示例使用
            self.last_hidden_state = hidden_states
            
            # 计算注意力（简化版）
            attn_output = layer['attention']['output'](value)
            hidden_states = hidden_states + attn_output
            
            # FFN
            intermediate = layer['intermediate'](hidden_states)
            output = layer['output'](intermediate)
            hidden_states = hidden_states + output
        
        return type('SimpleOutput', (), {'last_hidden_state': hidden_states, 'pooler_output': hidden_states[:,0]})

# 计算模型参数量函数
def calculate_params(config):
    # 词嵌入参数
    vocab_size = config.vocab_size
    embedding_params = vocab_size * config.n_embd
    
    # 位置嵌入参数
    position_params = config.n_ctx * config.n_embd
    
    # 每个Transformer层的参数
    # 1. 多头注意力
    #    - Q, K, V 投影矩阵: 3 * (n_embd * n_embd)
    #    - 输出投影矩阵: n_embd * n_embd
    # 2. 前馈网络
    #    - 两个线性层: n_embd * (n_embd * 4) + (n_embd * 4) * n_embd
    # 3. 层归一化参数: 2 * 2 * n_embd (2个LayerNorm，每个有两组参数)
    params_per_layer = 4 * (config.n_embd ** 2) + 2 * config.n_embd * (4 * config.n_embd) + 4 * config.n_embd
    
    # 最后的层归一化和语言模型头
    final_layer_params = 2 * config.n_embd + config.n_embd * vocab_size
    
    # 总参数量
    total_params = embedding_params + position_params + (params_per_layer * config.n_layer) + final_layer_params
    
    return total_params
