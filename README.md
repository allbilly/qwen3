# Qwen3: from zero to megakernel 

This is a blog series to implement Qwen3 0.6B inference engine from zero to megakernel and to document my learning progress. 

So how does qwen3 architecture look like. 
Here is the  Qwen3 architecture diagram by Sebastian Raschka.

![Qwen3 architecture](./asset/qwen3.png)

Let's create a python file qwen3_torch.py

```python
import torch
prompt = "What is the Ultimate Answer to Life, the Universe, and Everything?"
```

But LLM cant read ASCII characters. It read vector. 
We can use Embedding to convert input into vector size of (1,1024).

When human reads a sentence, we read word by word "What-is-the..." but not by character "W-h-a-t...".
So the plan is to first split the prompt by space. 
and we get

```python
words = prompt.split(" ")
['What', 'is', 'the', 'Ultimate', 'Answer', 'to', 'Life,', 'the', 'Universe,', 'and', 'Everything?']
```

and we can embed each word and LLM will understand each word like a human does.
To save compute, we can also save the emdedding result vector to a lookup table.

Wait, google told me there are a million english vocab out there 
and LLM can understand multiple languages
and there will be new slogan every year
and if you looks closely to the architecture diagram

Qwen3 only have a vocab size of 151k and can chat with you can me.
The secret is the step between Sample input text and Token embedding layer, the tokenizer.

Tokenizer splits sentence into tokens. 
Tokens are like words but saves computation and handles edge cases above.

For example, if "Answer to" is a common phase that used together a lot, they can be merged into same token.
Such that when LLM process our prompt, it saves computation by one token.

The best way to understand tokenizer is to implement one.
- Character-Level Tokenizer (stoi/itos/BOS) https://www.deep-ml.com/problems/374
- Byte Pair Encoding (BPE) Tokenizer https://www.deep-ml.com/problems/380

Now, lets follow the architecture diagram to tokenizer our prompt
Instead of split by space, we want something like

```python
input_token_ids = tokenizer.encode(prompt)
```

We need to implement the class for tokenizer and the method encode.
```python
class Qwen3Tokenizer:
    def __init__(self):
        pass
    def encode(self, prompt):
        pass

tokenizer = Qwen3Tokenizer()
```

We need tokenizer.json that contains all of the 151k vocab
https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer.json

```python
import json
def __init__(self):
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    self.vocab = data["model"]["vocab"]
    print(list(self.vocab)[:100])
    print(len(self.vocab))
tokenizer = Qwen3Tokenizer("./tokenizer.json")
```

And we can see
```python
['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '¡', '¢', '£', '¤', '¥', '¦']
151643
```

to implement encode function, we first split into characters and then do BPE merging 
before continue, make sure to finish Byte Pair Encoding (BPE) Tokenizer https://www.deep-ml.com/problems/380

My solution is
```python
    res = []
    while (num_merges>0):
        count = {}
        for text, freq in corpus.items():
            text = text.split(" ")
            for i in range(len(text)-1):
                pair = text[i] + " " + text[i+1]
                if pair in count:
                    count[pair] += freq
                else:
                    count[pair] = freq
        most_freq = sorted(count.items(), key=lambda x:x[1])[-1:]
        res.append((most_freq[0][0].split(" ")[0], most_freq[0][0].split(" ")[1]))

        new_corpus = {}
        for text, freq in corpus.items():
            if most_freq[0][0] in text:
                text = text.replace(most_freq[0][0], "".join(most_freq[0][0].split(" ")))
            new_corpus[text] = freq
        corpus = new_corpus
        num_merges -= 1

    return res
```

But the Byte Pair Encoding (BPE) Tokenizer exercise is for training, for inference we use a precomputed merge_rules from the json

```python
def __init__(self):
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    self.vocab = data["model"]["vocab"]

    self.merge_rules = {}
    for i, merge in enumerate(data["model"]["merges"]):
        token_a, token_b = merge[0], merge[1]
        self.merge_rules[(token_a, token_b)] = i

def encode(self, prompt):
    # Step 1: Split into characters (initial tokens)
    tokens = list(prompt)
    
    # Step 2: BPE merging loop
    while True:
        best_merge_idx = float("inf")
        best_pos = -1
        best_pair = None
        
        # Find the best consecutive pair to merge
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merge_rules:
                merge_idx = self.merge_rules[pair]
                if merge_idx < best_merge_idx:
                    best_merge_idx = merge_idx
                    best_pos = i
                    best_pair = pair
        
        # No more merges possible
        if best_pos == -1:
            break
        
        # Apply the merge: combine tokens at best_pos
        merged_token = best_pair[0] + best_pair[1]
        tokens[best_pos] = merged_token
        del tokens[best_pos + 1]
    
    # Step 3: Convert tokens to IDs
    token_ids = []
    for token in tokens:
        if token in self.vocab:
            token_ids.append(self.vocab[token])
        else:
            # Handle unknown tokens (shouldn't happen with BPE)
            for char in token:
                token_ids.append(self.vocab.get(char, 0))
    
    return token_ids
```
and we get
```
[3838, 0, 285, 0, 1782, 0, 43484, 3426, 0, 16141, 0, 983, 0, 25749, 11, 0, 1782, 0, 1806, 8034, 11, 0, 437, 0, 34964, 30]
```

but the token output list is not the same as official huggingface ouput
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
prompt = 'What is the Ultimate Answer to Life, the Universe, and Everything?'
official_tokens = tokenizer.encode(prompt)
print('Official tokens:', official_tokens)
[3838, 374, 279, 28850, 21806, 311, 9414, 11, 279, 28749, 11, 323, 20094, 30]
``` 

Problem: Character-Level Split Doesn't Match Vocab
tokens = list(prompt)  # ["W", "h", "a", "t", " ", "i", ...]
But HuggingFace BPE uses byte-level encoding, where:
- Space " " (byte 32) → "Ġ" (U+0120) in vocab
- Newline "\n" (byte 10) → "Ċ" (U+010A) in vocab
- etc.
Your split produces " " but vocab has "Ġ", so lookup fails → returns 0.

fix is to add
```python
def bytes_to_unicode():
    """Map each byte to its unicode representation in BPE vocab"""
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
BYTE_ENCODER = bytes_to_unicode()
```

