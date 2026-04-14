We will reference implementation at https://github.com/adriancable/qwen3.c/blob/main/runq.c

Let's start with a minimal C file

```C
const char* prompt = "<|im_start|>user\n" "What is the Ultimate Answer to Life, the Universe, and Everything?" "<|im_end|>\n" "<|im_start|>assistant" "\n";

int main() {
    return 0;
}
```

and print out prompt
```C
int main() {
    printf("Prompt: %s\n", prompt);
    return 0;
}
```

```
   8 |     printf("Prompt: %s\n", prompt);
      |     ^
qwen3.c:8:5: note: include the header <stdio.h> or explicitly provide a declaration for 'printf'
```

as printf requires stdio

```C
#include <stdio.h>
```

and we have our prompt printed correctly
```
Prompt: <|im_start|>user
What is the Ultimate Answer to Life, the Universe, and Everything?<|im_end|>
<|im_start|>assistant
```

next we encode our prompt, because C does not have OOP we dont need a class
```C
void encode(){
}
```

we need to
1. split out the special token
2. do bpe merge

in python we can do 
parts = self._SPLIT_RE.split(prompt)

but in C people dont tend to import regex libary but do manual special token scanning
with checking char "<" and ">" in a for loop
```C
void encode(){
    if (*c == '<') {
        int end_of_token_pos = -1;
        found_special_token = 0;
        for (int k = 0; *c != 0 && k < 64; k++) {
            if (c[k] == '>') {
                end_of_token_pos = k;
                break;
            }
        }
    }
}
```


and when matched, we want to convert from token to id
1. for each char in text
2. find '<' and '>' pair for special tokens
3. token to id


```C
void encode(Tokenizer *t, char *text){
    // 1. for each char in text
    char special_token[64+1];
    int end_token_pos = -1; 
    int id, found_special_token = 0 ;
    for (char *c = text; *c!=0 ; c++){
        if (*c == '<'){
            // 2. find '<' and '>' pair for special tokens
            for (int k=0; k<64; k++){
                if (c[k] == '>'){
                    end_token_pos = k ;
                    break;
                }
            }
            
            if (end_token_pos != -1){
                strncpy(special_token, c, end_token_pos + 1);
                special_token[end_token_pos+1] = 0;
                printf("special: %s", special_token);

                // 3. token to id
                // we need tokenizer to provide us t->vocab and t->vocab_size
                id = str_lookup(special_token, t->vocab, t->vocab_size);
                printf("%d\n", id);

                c += end_token_pos;
            }
        }
    }
}
```

str_lookup is just a for loop to check if string matched
```C
int str_lookup(char *str, char **vocab, int vocab_size) {
    for (int i = 0; i < vocab_size; i++)
        if (!strcmp(str, vocab[i]))
            return i;
    return -1;
}
```

we need tokenizer to provide us t->vocab and t->vocab_size
lest create one
```C
typedef struct {
    char **vocab;  
    int vocab_size;  
} Tokenizer;
```

but where is vocab_size, its in the header of our .bin file
so lets create a Config struct to read all config first
```C
typedef struct {
    int magic_number; // checkpoint magic number
    int version; // file format version
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    int head_dim; // head dimension
    int shared_classifier; // 1 if wcls == p_tokens
    int group_size; // quantization group size (export.py uses 64)
} Config;
```

now we can initilizae config and build it from checkpoint_path
```C
char *checkpoint_path = "Qwen3-0.6B.bin" ;

Config config;
build_config(&config, checkpoint_path);
```

```C
void build_config(Config *config, char *checkpoint) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open checkpoint %s\n", checkpoint); exit(EXIT_FAILURE); }

    fseek(file, 0, SEEK_END); // move file pointer to end of file
    ssize_t file_size = ftell(file); // get the file size, in bytes

    float *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    fclose(file);

    // checkpoint format is 256-byte header, and then the model weights
    // but we are just reading sizeof(Config)=48bytes other are reserved
    memcpy(config, data, sizeof(Config));
    if (config->magic_number != 0x616a6331) { fprintf(stderr, "File %s is not a qwen3.c checkpoint\n", checkpoint); exit(EXIT_FAILURE); }
    if (config->version != 1) { fprintf(stderr, "Checkpoint %s is version %d, need version 1\n", checkpoint, config->version); exit(EXIT_FAILURE); }

    GS = config->group_size; // set as global, as it will be used in many places
}
```

As said by big pickle, Safetensors is basically PyTorch tensor serialization — extremely complex to parse in C. You'd need to:
- Parse safetensors header (JSON metadata)
- Understand PyTorch tensor format
- Handle bf16/fp16/bfp16 quantization
- Implement the full serialization spec
This defeats the entire purpose of using C (simplicity).

therefore, we would need a export script to convert .safetensor model weight to .bin
our export.py is from https://github.com/adriancable/qwen3.c/blob/main/export.py


