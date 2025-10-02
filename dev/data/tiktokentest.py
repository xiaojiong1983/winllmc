"""
this is a test script to use tiktoken to tokenize chinese text
and save to a .bin file for training.
The .bin files can be directly memory-mapped by the training code,
which makes it very efficient to load data from disk to CPU over a network
author: gjg @2024/10/9
Note: the .bin files are not human-readable, you need to decode them back to text
using the same tokenizer that was used to encode them. This is important because different tokenizers
may use different token ids for the same text. Therefore, the .bin files are not portable
between different tokenizers. You need to know which tokenizer was used to encode the .bin files
in order to decode them back to text. This is usually done by specifying the model type
(e.g., gpt-2, llama-3) when encoding and decoding the .bin files.
The .bin files are designed to be used with the training code that can directly memory-map
"""

import argparse
import os

import tiktoken
from transformers import AutoTokenizer

from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hometown")

def tokenize(model_desc):
    if model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        eot = enc._special_tokens['<|endoftext|>'] # end of text token
    elif model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    else:
        raise ValueError(f"unknown model descriptor {model_desc}")
    data_filename = os.path.join(DATA_CACHE_DIR, "temp.txt")
    text = open(data_filename, 'r', encoding='utf-8').read()
    # let's treat every individual chunk of text as a separate "document"
    sections = text.split("\n\n")
    tokens = []
    for i, s in enumerate(sections):
        tokens.append(eot)
        # there was a mild bug where I originally intended to remove \n\n, but instead just added
        # the EOT right after each \n\n, so I'm keeping that behavior for backwards compatibility
        # therefore we have to here add an extra \n\n at the end of each section, except the last
        spad = s + "\n\n" if i != len(sections) - 1 else s
        tokens.extend(encode(spad))
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens = tokens[:4]
    train_tokens = tokens[4:]
    # save to file
    val_filename = os.path.join(DATA_CACHE_DIR, "t_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "t_train.bin")
    write_datafile(val_filename, val_tokens, model_desc)
    write_datafile(train_filename, train_tokens, model_desc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="home town dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    args = parser.parse_args()
    tokenize(args.model_desc)
