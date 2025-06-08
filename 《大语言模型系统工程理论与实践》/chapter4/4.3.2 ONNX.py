import torch
import onnx
import tensorrt as trt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练模型
model = GPT2LMHeadModel.from_pretrained("gpt2-large")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

# 准备输入
input_text = "Hello, I am a language model"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 导出为ONNX格式
torch.onnx.export(
    model,
    input_ids,
    "gpt2_large.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=12
)

# 加载ONNX模型
onnx_model = onnx.load("gpt2_large.onnx")
onnx.checker.check_model(onnx_model)

# 示例使用说明
if __name__ == "__main__":
    print("ONNX和TensorRT示例代码")
    print("注意：TensorRT部分需要NVIDIA GPU和适当的驱动才能运行")
    
    print("\n以下代码展示了如何将模型转为TensorRT格式(需要适当的环境配置)")
    """
    # 创建TensorRT引擎
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    parser.parse(onnx_model.SerializeToString())

    # 配置TensorRT
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # 使用FP16精度

    # 构建引擎
    engine = builder.build_engine(network, config)

    # 保存引擎
    with open("gpt2_large.trt", "wb") as f:
        f.write(engine.serialize())
    """ 