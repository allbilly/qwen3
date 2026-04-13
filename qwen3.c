#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

int GS = 0;

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

typedef struct {
    char **vocab;  
    float *merge_scores;
    int vocab_size;  
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
} Tokenizer;

int str_lookup(char *str, char **vocab, int vocab_size) {
    // find a match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < vocab_size; i++)
        if (!strcmp(str, vocab[i]))
            return i;
    return -1;
}

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
                // we need tokenizer struct to provide us t->vocab and t->vocab_size
                id = str_lookup(special_token, t->vocab, t->vocab_size);
                printf("%d\n", id);

                c += end_token_pos;
            }
        }
    }
}

void read_checkpoint(char *checkpoint, Config *config) {
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

void build_config(Config *c, char *checkpoint_path) {
    read_checkpoint(checkpoint_path, c);
}

void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size) {
    char tokenizer_path[1024];

    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->merge_scores = (float *)malloc(vocab_size * sizeof(float));

    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    int len;

    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->merge_scores + i, sizeof(float), 1, file) != 1) {
            t->vocab[i] = (char *)malloc(1);
            t->vocab[i][0] = 0; // add the string terminating token
        } else {
            fread(&len, sizeof(int), 1, file);
            t->vocab[i] = (char *)malloc(len + 1);
            fread(t->vocab[i], 1, len, file);
            t->vocab[i][len] = 0; // add the string terminating token
        }
    }
    fclose(file);
}

const char* prompt = "<|im_start|>user\n" "What is the Ultimate Answer to Life, the Universe, and Everything?" "<|im_end|>\n" "<|im_start|>assistant" "\n";

int main(){
    printf("Prompt:  %s\n", prompt);
    char *checkpoint_path = "Qwen3-0.6B.bin" ;

    Config config;
    build_config(&config, checkpoint_path);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, checkpoint_path, config.vocab_size);

    encode(&tokenizer, prompt);
    return 0;
}