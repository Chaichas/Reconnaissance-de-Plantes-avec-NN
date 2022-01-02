#include "Pooling_layer.h"
#define Pooling_size 2

Pooling_layer::Pooling_layer(){} //constructor


void Pooling_layer::Pooling_parameters(const std::vector<double>& vec_convolution, int input_height, int input_width){

	int Pooling_height = input_height/Pooling_size; //Height of the output pooling matrix
	int Pooling_width = input_width/Pooling_size; //Width of the output pooling matrix
	
}



Pooling_layer::~Pooling_layer(){} //Destructor
