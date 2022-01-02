#include "Pooling_layer.h"
#include <algorithm>
#include <cmath>

//Parameters of Pooling
#define Pooling_size 2
#define Filter_height 2
#define Filter_width 2

Pooling_layer::Pooling_layer(){} //constructor


/*Backpropagation from the softmax activation function to the pooling layer

cache_BK for implementing the backward phase, crucial for caching:
(1) the input from convolution layer "before flattening" it, (2) the input "after flattening" and (3) the values input of the "softmax activation function" */

void Pooling_layer::cache_BK(const std::vector<double>& vect){ //caching
  
	CacheMat_input.clear(); //Clear the old CacheMat_input
	CacheMat_input.resize(vect.size()); //Resize CacheMat_input

	//Copy: output.assign(input.begin(), input.end());
	CacheMat_input.assign(vect.begin(), vect.end()); //output = CacheMat_input
	
	int CacheMat_height = (Pooling_height * Pooling_size); //High of the cache matrix
	int CacheMat_width = (Pooling_width * Pooling_size); //Width of the cache matrix
}



void Pooling_layer::Pooling_parameters(const std::vector<double>& vec_convolution, int input_height, int input_width){

	int Pooling_height = input_height/Pooling_size; //Height of the output pooling matrix
	int Pooling_width = input_width/Pooling_size; //Width of the output pooling matrix
	
	Pooling_Matrix.clear(); //clear the last matrix
	
	//Pooling procedure for the 8 filters used
	Pooling_process(vec_convolution, 0); //1st convolution matrix
	Pooling_process(vec_convolution, 1); //2sd convolution matrix
	Pooling_process(vec_convolution, 2); //3rd convolution matrix
	Pooling_process(vec_convolution, 3); //4th convolution matrix
	Pooling_process(vec_convolution, 4); //5th convolution matrix
	Pooling_process(vec_convolution, 5); //6th convolution matrix
	Pooling_process(vec_convolution, 6); //7th convolution matrix
	Pooling_process(vec_convolution, 7); //8th convolution matrix
	
	
	cache_BK(vec_convolution); //Cache of vec_convolution input
}


//Pooling Process
void Pooling_layer::Pooling_process(const std::vector<double>& pixel, int idx){
	std::vector<double> vec;
	
	for (int ii = 0; ii < (Pooling_size * input_height); ii+= Pooling_size){ 
		for (int jj = 0; jj < (Pooling_size * input_width); jj+= Pooling_size){ 
		
			std::vector<double> v; //initialize a vector v
			for (int kk = 0; kk < Filter_height; kk++){
				for (int hh = 0; hh < Filter_width; hh++){
			
					v.push_back(pixel[idx][(ii + kk) * (Pooling_height * Pooling_size) + (jj + hh)]); //store the elements of input matrix to the pooling layer
				}
			}
		
			//To search the max element, we use the function: max_element(v.begin(), v.end());
			double weightMax = *max_element(v.begin(), v.end()); //search the max of vector v elements
			vec.push_back(weightMax); //store the max weight for each matrix bloc in vec
		}
	}
	
	Pooling_Matrix.push_back(vec); //The output pooling matrix
}
		
		
		

Pooling_layer::~Pooling_layer(){} //Destructor
