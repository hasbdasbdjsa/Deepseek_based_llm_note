# 大模型部署第七章代码验证
本仓库包含了第七章的代码示例，包括单GPU部署、量化、蒸馏等技术。


## 实验验证环境

### 系统环境
- 操作系统: Windows 11
- CPU: AMD Ryzen 7 7435HS
- GPU: NVDIA GeForce RTX 4060 8G

### 软件环境
- Python: 3.8 
- PyTorch: 2.3.1 (CUDA 12.8)
- NumPy: 2.0.2
- Transformers: 4.52.4
### 验证结果
所有代码都已成功验证运行，但存在以下注意事项:
- 一定要安装Pytorch 2.3.1在后续的量化过程中需要指定这个版本，不然会报错
  
## 环境设置

```bash
# 安装依赖
pip install -r requirements.txt
```
## 代码文件说明

### 7.2.2 硬件架构特点对推理性能的影响
1.Instruction Set Optimization.py
- a. CPU上使用AVX-512指令集加速矩阵乘法示例
- b.依赖numpy和sklearn
- c.运行Instruction Set Optimization.py
2. GPU Shared Memory Optimization.py
- a. 在PyTorch中利用GPU共享内存优化注意力计算
- b.依赖torch
- c.运行GPU Shared Memory Optimization.py
3. Host and Device Memory Management.py
- a. 主机与设备内存管理示例
- b.依赖torch
- c.运行Host and Device Memory Management.py
### 7.3.1 参数规模、显存需求与算力预算
1. Graphics Memory Requirement Estimation.py
- a. 推理算力要求计算
- b.依赖python
- c.运行Graphics Memory Requirement Estimation.py
2. Reasoning about arithmetic requirements.py
- a. 显存预算估计示例
- b.依赖python
- c.运行Reasoning about arithmetic requirements.py
### 7.3.2 推理算力估算与延迟性能分析
1. Delayed estimation.py
- a. 推理延迟估算
- b.依赖python
- c.运行Delayed estimation.py
2. Throughput calculation.py
- a. 吞吐量估算函数
- b.依赖python和Delayed estimation.py
- c.运行Throughput calculation.py
### 7.3.3 大模型选择与硬件适配实践示例
1. Single GPU Deployment LLaMA-70B.py
- a. 单个GPU部署LLaMA-70B模型
- b.依赖torch、transformers和auto_gptq
- c.运行Single GPU Deployment LLaMA-70B.py
2. Multi-GPU Deployment LLaMA-13B.py
- a. 多设备部署的示例代码
- b.依赖torch、transformers和deepspeed
- c.运行Multi-GPU Deployment LLaMA-13B.py
3. Edge Deployment.py
- a. 边缘部署的示例代码
- b.依赖torch、transformers和onnxruntime
- c.运行Edge Deployment.py
### 7.4.1 量化（Quantization）原理与技术
1. Quantification of GPTQ.py
- a. GPTQ量化的示例代码
- b.依赖torch、transformers和auto_gptq
- c.运行Quantification of GPTQ.py
2. Quantification of AWQ.py
- a. AWQ量化的示例代码
- b.依赖torch、transformers和awq
- c.运行Quantification of AWQ.py
### 7.4.2 剪枝（Pruning）原理与技术
1. Decrease in range.py
- a. 基于幅度的简单剪枝示例代码
- b.依赖torch、transformers和numpy
- c.运行Decrease in range.py
### 7.4.3 蒸馏（Distillation）原理与技术
1. Knowledge Distillation.py
- a. 基本的知识蒸馏示例代码
- b.依赖torch、transformers和numpy
- c.运行Knowledge Distillation.py
### 7.4.4 量化、剪枝与蒸馏组合优化对比
1. Combining quantification and distillation.py
- a. 基本的知识蒸馏示例代码
- b.依赖torch、transformers、numpy、Knowledge Distillation.py和Decrease in range.py
- c.运行Combining quantification and distillation.py

## 模型自定义方案

由于环境兼容性和网络限制问题，添加了`download_model.py`文件提供以下功能：
- 提供自定义模型下载，只需要更改`download_model.py`下载模型的名称就可以实现下载，无需其他网络配置


## 注意事项

- 部分代码为教学示例，可能需要额外数据或模型才能完整运行
- 运行Multi-GPU Deployment LLaMA-13B.py相关代码需要安装deepspeed，同时建议在Linux环境下运行


























