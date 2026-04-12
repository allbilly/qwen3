#include <stdio.h>
#include <string.h>

const char* prompt = "<|im_start|>user\n" "What is the Ultimate Answer to Life, the Universe, and Everything?" "<|im_end|>\n" "<|im_start|>assistant" "\n";

// Tokenizer

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
                printf("special: %s\n", special_token);

                // 3. token to id
                printf("Matching its id");
                // we need tokenizer struct to provide us t->vocab and t->vocab_size
                id = str_lookup(special_token, t->vocab, t->vocab_size);

                c += end_token_pos;
            }
        }


    }
}

int main(){
    printf("Prompt:  %s\n", prompt);
    encode(prompt);
    return 0;
}