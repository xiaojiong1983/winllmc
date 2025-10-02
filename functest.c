// A simple functional test to make sure the code compiles and links correctly
// and that a model can be built from a checkpoint file.
// This does not do any actual training or inference.
// To compile and run this test, use:
// gcc -o functest functest.c -lm
// ./functest   
// autor: gjg 20250927
#ifndef TESTING
#define TESTING
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "train_gpt2.c"

int main(int argc, char *argv[]) {
    // build the GPT-2 model from a checkpoint
    GPT2 model;
    const char* load_filename = "gpt2:d12"; 

    memset(&model, 0, sizeof(GPT2));

    gpt2_init_common(&model);

    gpt_build_from_descriptor(&model, load_filename);

    printf("Model built successfully with %Iu parameters.\n", model.num_parameters);

    // gpt2_build_from_checkpoint(&model, "gpt2_d12.bin");
    return 0;
}
