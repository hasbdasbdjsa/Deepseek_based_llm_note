### 1.3RoPE旋转位置编码

#### 1.3.1位置编码的重要性

在自然语言处理（NLP）和代码建模任务中，Transformer 模型是当前主流的序列建模架构。然而，Transformer 本身并不具备序列位置信息，因为它依赖于自注意力机制（Self-Attention），而该机制在计算注意力分数时对输入 Token 进行无序匹配，无法直接感知 Token 的相对顺序。为了解决这一问题，我们通常需要位置编码（Positional Encoding），即给输入的 Token 赋予额外的位置信息，使得 Transformer 能够在注意力计算时区分不同位置的 Token。

#### 1.3.2RoPE概述

旋转位置编码（Rotary Position Embedding，RoPE）是论文[Roformer: Enhanced Transformer With Rotray Position Embedding](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2104.09864.pdf) 提出的一种能够将相对位置信息依赖集成到 self-attention 中并提升 transformer 架构性能的位置编码方式。

RoPE 主要依赖于旋转变换（Rotation Transformation）来编码Token之间的相对位置信息。其核心思想包括：将Token的嵌入向量映射到二维平面，并通过旋转操作对不同位置的 Token 进行编码；在计算自注意力（Self-Attention）时，利用旋转矩阵保持 Query-Key 之间的相对位置关系。

和相对位置编码相比，RoPE 具有更好的**外推性**，目前是大模型相对位置编码中应用最广的方式之一。外推性是指大模型在训练时和预测时的输入长度不一致，导致模型的泛化能力下降的问题。



三角函数、旋转矩阵、欧拉公式、复数等数学背景知识可以参考这篇[文章](https://github.com/harleyszhang/llm_note/blob/main/1-transformer_model/位置编码算法背景知识.md)学习。



#### 1.3.3相关torch函数

##### 1, torch.outer

外积（outer product）是指两个向量  a  和  b  通过外积操作生成的矩阵
$$
\text{result}[i, j] = a[i] \times b[j]
$$
其中 $a \otimes b$ 生成一个矩阵，行数等于向量 $a$ 的元素数，列数等于向量 $b$ 的元素数。

```bash
>>> a = torch.tensor([2,3,1,1,2], dtype=torch.int8)
>>> b = torch.tensor([4,2,3], dtype=torch.int8)
>>> c = torch.outer(a, b)
>>> c.shape
torch.Size([5, 3])
>>> c
tensor([[ 8,  4,  6],
        [12,  6,  9],
        [ 4,  2,  3],
        [ 4,  2,  3],
        [ 8,  4,  6]], dtype=torch.int8)
```
##### 2，`torch.matmul`

可以处理更高维的张量。当输入张量的维度大于 2 时，它将执行批量矩阵乘法。
```bash
>>> A = torch.randn(10, 3, 4)
>>> B = torch.randn(10, 4, 7)
>>> C = torch.matmul(A, B)
>>> D = torch.bmm(A, B)
>>> assert C.shape == D.shape # shape is torch.Size([10, 3, 7])
>>> True
```

##### 3，`torch.polar`

```python
# 第一个参数是绝对值（模），第二个参数是角度
torch.polar(abs, angle, *, out=None) → Tensor
```
构造一个复数张量，其元素是极坐标对应的笛卡尔坐标，绝对值为 abs，角度为 angle。
$$
\text{out=abs⋅cos(angle)+abs⋅sin(angle)⋅j}
$$


```python
# 假设 freqs = [x, y], 则 torch.polar(torch.ones_like(freqs), freqs) 
# = [cos(x) + sin(x)j, cos(y) + sin(y)j]
>>> angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
>>> z = torch.polar(torch.ones_like(angle), angle)
>>> z
tensor([ 6.1232e-17+1.0000j, -7.0711e-01-0.7071j], dtype=torch.complex128)
>>> a = torch.tensor([np.pi / 2], dtype=torch.float64) # 数据类型必须和前面一样
>>> torch.cos(a)
tensor([6.1232e-17], dtype=torch.float64)
```

##### 4，`torch.repeat_interleave`

```python
# 第一个参数是输入张量
# 第二个参数是重复次数
# dim: 沿着该维度重复元素。如果未指定维度，默认会将输入数组展平成一维，并返回一个平坦的输出数组。
torch.repeat_interleave(input, repeats, dim=None, *, output_size=None) → Tensor
```
返回一个具有与输入相同维度的重复张量

```bash
>>> keys = torch.randn([2, 12, 8, 512])
>>> keys2 = torch.repeat_interleave(keys, 8, dim = 2)
>>> keys2.shape
torch.Size([2, 12, 64, 512])
>>> x
tensor([[1, 2],
        [3, 4]])
>>> torch.repeat_interleave(x, 3, dim=1)
tensor([[1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4]])
>>> torch.repeat_interleave(x, 3)
tensor([1, 1, 1, 3, 3, 3, 4, 4, 4, 5, 5, 5])
```

**注意重复后元素的顺序**，以简单的一维为例 `x = [a,b,c,d]`，`torch.repeat_interleave(x, 3)` 后，结果是 `[a,a,a,b,b,b,c,c,c,d,d,d]`。



#### 1.3.4RoPE代码实现

