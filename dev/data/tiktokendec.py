"""
this is a test script to decode the tiktokendec .bin files back to text
using the same tokenizer that was used to encode them. This is important because different tokenizers
author: gjg @2024/10/9
may use different token ids for the same text. Therefore, the .bin files are not portable
between different tokenizers. You need to know which tokenizer was used to encode the .bin files
"""

import argparse
import struct
from transformers import AutoTokenizer
import tiktoken

def read_token_ids(bin_path, dtype="i"):
    """读取bin文件中的token ID（默认int32，对应格式字符'i'）"""
    with open(bin_path, "rb") as f:
        data = f.read()
    print(f"实际读取的字节数: {len(data)}")  # 新增：打印实际字节数
    token_size = struct.calcsize(dtype)
    print(f"单个token的字节数（{dtype}）: {token_size}")  # 新增：打印单个token字节数
    if len(data) % token_size != 0:
        raise ValueError(f"字节数不匹配：{len(data)} 不能被 {token_size} 整除")

    # 计算token数量：总字节数 / 每个token的字节数（int32为4字节）
    token_count = len(data) // struct.calcsize(dtype)
    # 解析为整数列表
    return struct.unpack(f"{token_count}{dtype}", data)

def decode_tokens(token_ids, model_desc):
    """用对应tokenizer解码为文本"""
    if model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        return enc.decode(token_ids)
    elif model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        return tokenizer.decode(token_ids)
    else:
        raise ValueError(f"不支持的模型：{model_desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="home town dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    args = parser.parse_args()
    token_ids = read_token_ids("hometown/t_train.bin")
    text = decode_tokens(token_ids, args.model_desc)
    print("解码后的文本：\n", text)

