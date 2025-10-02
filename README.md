# 说明
本项目是对 karpathy 的 [llm.c](https://github.com/karpathy/llm.c) 在 Windows 环境下的移植版本，用于学习和实战大型语言模型（LLM）。

# llm.c
这是一个用纯C编写的 LLM 项目，无需依赖 245MB 的 PyTorch 或 107MB 的 Python。当前重点是预训练，尤其是复现 GPT-2 和 GPT-3 的迷你版本，同时提供一个并行的 PyTorch 参考实现 train_gpt2.py。目前，llm.c 的速度比 PyTorch Nightly 快约 7%。这里提供了一个简洁的 CPU fp32(32位单精度浮点数) 参考实现 train_gpt2.c，仅约 1000 行代码。

## 快速开始
目前最好的入门方式是复现 GPT-2（124M）模型，详细步骤见 [Discussion #481](https://github.com/karpathy/llm.c/discussions/481) 
我们也可以使用 llm.c 或 PyTorch 复现 GPT-2 和 GPT-3 系列的其他模型，详见 scripts README。

我们在CPU进行体验，体验 llm.c 的训练过程，只是速度不会太快。例如，你可以微调 GPT-2 small（124M）来生成莎士比亚风格的文本：

```bash
build_tran_gpt2_cpu
OMP_NUM_THREADS=8 
./train_gpt2
```

可以通过下述脚本来获得bin文件

```bash
python dev/data/tinyshakespeare.py
python train_gpt2.py
```

上述命令会：
下载已分词的 tinyshakespeare 数据集和 GPT-2 124M 权重
在 C 中加载权重，用 AdamW 优化器训练 40 步（batch size=4，上下文长度=64）
评估验证集损失并生成一些文本。


他在上面的第（1）行下载了一个已经标记的[tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)数据集并下载GPT-2（124M）权重，（3）用C从中初始化，用AdamW在tineshakespeare上训练40步（使用批量4，上下文长度仅为64），评估验证损失，并对一些文本进行采样。老实说，除非你有一个强大的CPU（并且可以在启动命令中增加OMP线程的数量），否则你在CPU训练LLM方面不会走那么远，但这可能是一个很好的演示/参考。在我的MacBook Pro（苹果Silicon M3 Max）上，输出如下：


The above lines (1) download an already tokenized [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset and download the GPT-2 (124M) weights, (3) init from them in C and train for 40 steps on tineshakespeare with AdamW (using batch size 4, context length only 64), evaluate validation loss, and sample some text. Honestly, unless you have a beefy CPU (and can crank up the number of OMP threads in the launch command), you're not going to get that far on CPU training LLMs, but it might be a good demo/reference. The output looks like this on my MacBook Pro (Apple Silicon M3 Max):

```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124439808
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 73323776
val loss 5.252026
step 0: train loss 5.356189 (took 1452.121000 ms)
step 1: train loss 4.301069 (took 1288.673000 ms)
step 2: train loss 4.623322 (took 1369.394000 ms)
step 3: train loss 4.600470 (took 1290.761000 ms)
... (trunctated) ...
step 39: train loss 3.970751 (took 1323.779000 ms)
val loss 4.107781
generating:
---
Come Running Away,
Greater conquer
With the Imperial blood
the heaviest host of the gods
into this wondrous world beyond.
I will not back thee, for how sweet after birth
Netflix against repounder,
will not
flourish against the earlocks of
Allay
---
```

## 数据集

/dev/data/(dataset).py 文件负责下载、分词并将 token 保存为 .bin 文件，方便 C 代码读取。
例如：

The data files inside `/dev/data/(dataset).py` are responsible for downloading, tokenizing and saving the tokens to .bin files, readable easily from C. So for example when you run:

```bash
python dev/data/tinyshakespeare.py
```

我们下载并标记[tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)数据集。其输出如下：


We download and tokenize the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. The output of this looks like this:

```
writing 32,768 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_val.bin
writing 305,260 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_train.bin
```
.bin 文件结构：前 1024 字节是头部，后面是 uint16 类型的 token 流，使用 GPT-2 的分词器。
更多数据集见 /dev/data 目录。

The .bin files contain a short header (1024 bytes) and then a stream of tokens in uint16, indicating the token ids with the GPT-2 tokenizer. More datasets are available in `/dev/data`.

## test

我附带了一个简单的单元测试，确保 C 代码与 PyTorch 代码结果一致。


I am also attaching a simple unit test for making sure our C code agrees with the PyTorch code. On the CPU as an example, compile and run with:

```bash
make test_gpt2
./test_gpt2
```

现在，它加载由train_gpt2.py编写的`gpt2_124M_debug_state.bin`文件，运行前向传递，将logits和loss与PyTorch参考实现进行比较，然后与Adam进行10次迭代训练，并确保loss与PyTorch匹配。为了测试GPU版本，我们运行：

This now loads the `gpt2_124M_debug_state.bin` file that gets written by train_gpt2.py, runs a forward pass, compares the logits and loss with the PyTorch reference implementation, then it does 10 iterations of training with Adam and makes sure the losses match PyTorch. To test the GPU version we run:

```bash
# fp32 test (cudnn not supported)
make test_gpt2cu PRECISION=FP32 && ./test_gpt2cu
# mixed precision cudnn test
make test_gpt2cu USE_CUDNN=1 && ./test_gpt2cu
```

这测试了fp32路径和混合精度路径。测试应该通过并打印“总体正常：1”。

This tests both the fp32 path and the mixed precision path. The test should pass and print `overall okay: 1`.

## 教程

参见 [doc/layernorm/layernorm.md](doc/layernorm/layernorm.md). 逐步讲解如何实现 GPT-2 的一个层（LayerNorm），适合入门理解 C 中的层实现。C.

**flash attention**. As of May 1, 2024 we use the Flash Attention from cuDNN. Because cuDNN bloats the compile time from a few seconds to ~minute and this code path is right now very new, this is disabled by default. You can enable it by compiling like this:

```bash
make train_gpt2cu USE_CUDNN=1
```

This will try to compile with cudnn and run it. You have to have cuDNN installed on your system. The [cuDNN installation instructions](https://developer.nvidia.com/cudnn) with apt-get will grab the default set of cuDNN packages. For a minimal setup, the cuDNN dev package is sufficient, e.g. on Ubuntu 22.04 for CUDA 12.x:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12
```

On top of this you need the [cuDNN frontend](https://github.com/NVIDIA/cudnn-frontend/tree/main), but this is just header files. Simply clone the repo to your disk. The Makefile currently looks for it in either your home directory or the current directory. If you have put it elsewhere, add `CUDNN_FRONTEND_PATH=/path/to/your/cudnn-frontend/include` to the `make` command-line.

## multi-GPU training

Make sure you install MPI and NCCL, e.g. on Linux:

```bash
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
```

For NCCL follow the instructions from the [official website](https://developer.nvidia.com/nccl/nccl-download) (e.g. network installer)

and then:

```bash
make train_gpt2cu
mpirun -np <number of GPUs> ./train_gpt2cu
```

or simply run one of our scripts under `./scripts/`.

## multi-node training

Make sure you've installed `NCCL` following instructions from [multi-GPU](#multi-gpu-training) section.

There are 3 ways we currently support that allow you to run multi-node training:
1) Use OpenMPI to exchange nccl id and initialize NCCL. See e.g. `./scripts/multi_node/run_gpt2_124M_mpi.sh` script for details.
2) Use shared file system to init NCCL. See `./scripts/multi_node/run_gpt2_124M_fs.sbatch` script for details.
3) Use TCP sockets to init NCCL. See `./scripts/multi_node/run_gpt2_124M_tcp.sbatch` script for details.

Note:
* If you're running in a slurm environment and your slurm doesn't support PMIx (which we assume will be a common situation given that `slurm-wlm` dropped PMIx support) you will have to use FS (2) or TCP (3) approach. To test whether your slurm supports PMIx run: `srun --mpi=list` and see whether you get `pmix` in the output.
* If you don't have slurm set up, you can kick off a multi-node run using `mpirun` - MPI (1).

None of these 3 methods is superior, we just offer you options so that you can run in your specific environment.

## experiments / sweeps

Just as an example process to sweep learning rates on a machine with 4 GPUs on TinyStories. Run a shell script `sweep.sh` (after you of course `chmod u+x sweep.sh`):

```bash
#!/bin/bash

learning_rates=(3e-5 1e-4 3e-4 1e-3)

for i in {0..3}; do
    export CUDA_VISIBLE_DEVICES=$i
    screen -dmS "tr$i" bash -c "./train_gpt2cu -i data/TinyStories -v 250 -s 250 -g 144 -l ${learning_rates[$i]} -o stories$i.log"
done

# you can bring these down with
# screen -ls | grep -E "tr[0-3]" | cut -d. -f1 | xargs -I {} screen -X -S {} quit
```

This example opens up 4 screen sessions and runs the four commands with different LRs. This writes the log files `stories$i.log` with all the losses, which you can plot as you wish in Python. A quick example of how to parse and plot these logfiles is in [dev/vislog.ipynb](dev/vislog.ipynb).


## license

MIT
