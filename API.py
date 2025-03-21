from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM 
'''Tokenizer分词器,将一段文本分割成很多单词或者子单词,对应模型的输入。'''
'''AutoModelForCausualLM是Hugging Face的transformers库中的一个类,主要用于因果语言建模(Causal Language Modeling).
这种模型的任务是给定之前的词或字符序列，预测文本序列中下一个词或字符，广泛应用于生成式任务，如对话系统、文本续写、摘要生成等‌.'''
import torch

# 请将这里替换为你实际存放大模型文件的路径
model_path = "D:\\pytorch_env"

# 加载本地模型
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    '''尝试从指定的模型路径加载预训练的分词器'''
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto",\
            torch_dtype=torch.float16)
    '''尝试从指定的模型路径加载预训练的基于因果语言模型的模型'''
    model = model.eval()
    '''将模型设置为评估模式，这会关闭一些在训练时使用但在评估时不需要的层'''
except Exception as e:      # 如果在上述加载和设置过程中出现任何异常，打印错误信息
    print(f"模型加载失败: {e}")
    raise

app = FastAPI()   # 创建一个 FastAPI 应用实例

class QueryRequest(BaseModel):
    text: str

# 定义一个 POST 请求的路由，路径为 /query
@app.post("/query")
async def query(request: QueryRequest):
    '''定义一个异步函数，用于处理 /query 路径的 POST 请求'''
    try:
        # 构建输入格式
        prompt = f"<|Human|>: {request.text}<|EndOfTurn|><|AI|>:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        # 调整生成参数
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=128, #用于控制新生成内容的长度
            temperature=0.9,
            top_k=50,
            top_p=0.95, #temperature, top_k, top_p 都用来控制生成文本的随机性和多样性（与长度无关）
            repetition_penalty=1.3,  # 加大重复惩罚，用于减少生成文本中的重复内容
            num_beams=1,  # 单束搜索，减少计算复杂度
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        '''将生成的回复封装在一个字典中返回'''
        return {"response": response}
    except Exception as e:
        return {"error": f"处理请求时出错: {e}"}

# 当脚本作为主程序运行时执行以下代码
if __name__ == "__main__":
    import uvicorn
    '''导入 uvicorn 库，用于运行 FastAPI 应用'''
    uvicorn.run(app, host="0.0.0.0", port=8000)
    