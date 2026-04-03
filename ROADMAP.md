# Quickstart
```
hf download Qwen/Qwen3-0.6B
uv venv
source .venv/bin/activate
uv pip install torch safetensor packaging tokenizers numpy
python qwen3_torch.py
```

# roadmap1
- list kernels nedeed 
- write naive metal
- profile and optimize like yalm
- add deepseekv2 to calm infer.c
- add to infer.m
- profile and optimize

# roadmap2
- custom tokenizer in python
- qwen3_pure.py
- qwen3.c
- qwen3.m
- pytorch offload matmul to metal kernel
- deepseekv2_torch.py
- deepseekv2_pure.py
- deepseekv2.c
- deepseekv2.m