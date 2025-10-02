@echo off
setlocal enabledelayedexpansion

python train_gpt2.py ^
--input_bin=dev/data/tinyshakespeare/tiny_shakespeare_train.bin ^
--input_val_bin=dev/data/tinyshakespeare/tiny_shakespeare_val.bin ^
--output_dir=model_out ^
--model=gpt2 ^
--sequence_length=64 ^
--batch_size=4 ^
--total_batch_size=256 ^
--num_iterations=6 ^
--learning_rate=1e-4 ^
--warmup_iters=0