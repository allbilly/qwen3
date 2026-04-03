# Qwen3: from zero to megakernel 

This is a blog seris to implement Qwen3 0.6B inference engine from zero to megakernel and to document my learning progress. 

So how does qwen3 architecture look like.

![Qwen3 architecture](./asset/qwen3.png)

Let's create a python file qwen3_torch.py

```
import torch
prompt = "What is the Ultimate Answer to Life, the Universe, and Everything?"
```

LLM cant read the ASCII character. It read embeddings. 
Embedding is the process to convert input into vector size of hidden diemension (1,1024)
But how should we embed our prompt? 

When human reads a sentence, we read word by word "What-is-the..." but not by character "W-h-a-t...".
So the plan is to first split the prompt by space.
and we get

```
words = prompt.split(" ")
['What', 'is', 'the', 'Ultimate', 'Answer', 'to', 'Life,', 'the', 'Universe,', 'and', 'Everything?']
```
