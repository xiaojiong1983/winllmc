"""
tiktoken 分词可视化工具
=======================
该脚本使用 tiktoken 库对输入文本进行分词，并打印分词结果及相关信息。
用法示例
--------
```bash         
python tokenizertest.py -t "北方人吃饺子" -e gpt2
python tokenizertest.py --text "北方人吃饺子" --encoder cl100k_base
python tokenizertest.py --text "北方人吃饺子" --encoder p50k_base
```
参数说明
--------    
-t, --text : 待分词的原始文本 (默认: "北方人吃饺子")
-e, --encoder : 编码器名称 (默认: gpt2)，可选值包括 gpt2 / r50k_base / p50k_base / cl100k_base
依赖    
--------
- tiktoken 库 (安装: pip install tiktoken)

作者
------
gjg
日期    
------

"""

import argparse
import tiktoken


def tokenize(text: str, encoder_name: str = "gpt2"):
    enc = tiktoken.get_encoding(encoder_name)

    # display tiktoken details
    print("编码器名称:", enc.name)
    print("词表大小  :", enc.n_vocab)

    # encode & display tokenization details
    print("\n—— 分词结果 ——")
    token_ids = enc.encode(text)

    print(f"原始文本: {text}")
    print(f"文本长度: {len(text)} (中文需再乘以3)")
    print(f"Token 数: {len(token_ids)}")
    print("token 详情:")
    for tid in token_ids:
        token_bytes = enc.decode_single_token_bytes(tid)
        token_str = token_bytes.decode("utf-8", errors="replace")
        print(f"  ID {tid:>6} | 字节 {token_bytes!r:>15} | 显示 {token_str!r}")


def main():
    parser = argparse.ArgumentParser(description="tiktoken 分词可视化工具")
    parser.add_argument(
        "-t", "--text", 
        default="北京天安门", 
        help="待分词的原始文本")
    parser.add_argument(
        "-e", "--encoder",
        default="gpt2",
        choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base"],
        help="编码器名称 (默认: gpt2)",
    )
    args = parser.parse_args()

    tokenize(args.text, args.encoder)


if __name__ == "__main__":
    main()