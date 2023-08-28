#pragma once
#include<stdio.h>

typedef struct Parameters {
	int batch_size;
	int input_channels;
	int input_height;
	int input_width;
	int output_size;
	int score_size;
};