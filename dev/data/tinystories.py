"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is using either GPT-2 or LLaMA 3 tokenizer.

The output is written to a newly created tinystories/ folder.
The script prints:

For GPT-2:
Number of shards: 50
Tokenizing val split...
writing 19,043,638 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
writing 925,653,391 tokens to tinystories/TinyStories_train.bin

For LLaMA 3:
Number of shards: 50
Tokenizing val split...
writing 18,660,516 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
writing 907,021,844 tokens to tinystories/TinyStories_train.bin

And runs in few minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of uint16 (gpt-2) or uint32 (llama) numbers indicating the token ids.
"""

import argparse
import os
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tiktoken
from transformers import AutoTokenizer

# changed import to avoid circular import issues by gjg @2025/9/19
from data_common import download_file, append_tokens_to_file, write_datafile_header,HEADERS_INFO

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinystories")

def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

def process_shard(shard_index, shard_filename, model_desc):
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

    with open(shard_filename, "r") as f:
        data = json.load(f)
    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    for example in data:
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = encode(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    return all_tokens


def process_shard_and_save(shard_index, shard_filename, model_desc, temp_dir):
    # processes a single shard and saves the tokenized output to a temporary .bin file add by gjg @2025/9/19
    if model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        eot = enc._special_tokens['<|endoftext|>']
    elif model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
        eot = tokenizer.encode('')[0]
    else:
        raise ValueError(f"unknown model descriptor {model_desc}")

    with open(shard_filename, "r") as f:
        data = json.load(f)

    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)

    temp_filename = os.path.join(temp_dir, f"temp_{shard_index}.bin")
    with open(temp_filename, "wb") as f:
        for example in data:
            text = example["story"].strip()
            tokens = encode(text)
            eot_token = np.array([eot], dtype=HEADERS_INFO[model_desc]["token_dtype"])
            tokens_np = np.array(tokens, dtype=HEADERS_INFO[model_desc]["token_dtype"])
            f.write(eot_token.tobytes())
            f.write(tokens_np.tobytes())

    return temp_filename

def tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:

        print(f"Tokenizing {split_name} split...")

        split_filename = os.path.join(DATA_CACHE_DIR, f"TinyStories_{split_name}.bin")
        # remove existing .bin file if it exists add by gjg @2025/9/19
        if os.path.exists(split_filename):
            os.remove(split_filename)
            print(f"Removed existing file {split_filename}")

        temp_dir = os.path.join(DATA_CACHE_DIR, f"temp_{split_name}")
        os.makedirs(temp_dir, exist_ok=True)

        # parallel processing of shards and saving to temporary files add by gjg @2025/9/19
        # the other approach is to process each shard and return the tokens to the main process
        # but that can use a lot of memory if there are many tokens. this way we just write
        # each shard's tokens to a temporary file and then concatenate them later.
        temp_files = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_shard_and_save, shard_index, shard_filename, model_desc, temp_dir)
                       for shard_index, shard_filename in enumerate(split_shards)]
            for future in as_completed(futures):
                temp_files.append(future.result())
                print(f"Processed and saved shard to {future.result()}")
                print(f"Current number of temporary files: {len(temp_files)}")

        # first pass to count total tokens add by gjg @2025/9/19
        total_tokens = 0
        info = HEADERS_INFO[model_desc]
        dtype = info["token_dtype"]
        for temp_file in temp_files:
            file_size = os.path.getsize(temp_file)
            total_tokens += file_size // np.dtype(dtype).itemsize
        print(f"Total tokens counted: {total_tokens:,}")

        # second pass to actually write the tokens add by gjg @2025/9/19
        write_datafile_header(split_filename, total_tokens, model_desc)
        
        # last pass to actually write the tokens add by gjg @2025/9/19
        with open(split_filename, "ab") as out_f:
            for temp_file in temp_files:
                with open(temp_file, "rb") as in_f:
                    out_f.write(in_f.read())
                    print(f"Appended {temp_file} to {split_filename}")

        # cleanup temporary files add by gjg @2025/9/19
        for temp_file in temp_files:
            os.remove(temp_file)
        os.rmdir(temp_dir)  

        file_size = os.path.getsize(split_filename)
        print(f"Wrote {file_size} bytes to {split_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Stories dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    args = parser.parse_args()
    download()
    tokenize(args.model_desc)