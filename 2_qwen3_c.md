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
```