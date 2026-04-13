[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/allbilly/qwen3)

# Qwen3: from zero to megakernel 

This is a blog series to implement Qwen3 0.6B inference engine from zero to [Qwen3 megakernel](https://github.com/Infatoshi/grokking-megakernels) [Qwen3.5 megakernel](https://github.com/Luce-Org/luce-megakernel) and to document my learning progress. 

We will be reimplementing
- https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3
- https://github.com/adriancable/qwen3.c
- https://github.com/andrewkchan/yalm (for CUDA)
- https://github.com/zeux/calm (for Metal)
- https://github.com/Infatoshi/grokking-megakernels
- https://github.com/xiaguan/pegainfer (RUST maybe)

I also like to use mega kerenl to accerlate inference rollout for 
- https://github.com/YuvrajSingh-mist/smolcluster

# Table of Index
- [PART1 qwen3_torch.py](./1_qwen3_torch.md)
- [PART2 qwen3.c](./2_qwen3_c.md)

# How to compile
refernce code from https://github.com/adriancable/qwen3.c
- gcc -o qwen3_ref qwen3_ref.c -lm -O2 && ./qwen3_ref Qwen3-0.6B.bin -i "Hello world"

Our qwen3.c
- gcc qwen3.c -o qwen3 && ./qwen3
