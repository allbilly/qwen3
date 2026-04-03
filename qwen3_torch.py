import torch
import json

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

class Qwen3Tokenizer:
    def __init__(self, tokenizer_json_path):
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
        tokens = [BYTE_ENCODER[b] for b in prompt.encode("utf-8")]
        
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

tokenizer = Qwen3Tokenizer("./tokenizer.json")

prompt = "What is the Ultimate Answer to Life, the Universe, and Everything?"
input_token_ids = tokenizer.encode(prompt)
print(input_token_ids)