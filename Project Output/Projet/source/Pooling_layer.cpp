#include "Pooling_layer.h"
#include <algorithm>
#include <cmath>

//Parameters of Pooling
#define Pooling_size 2
#define Filter_height 2
#define Filter_width 2

Pooling_layer::Pooling_layer(){} //constructor


void Pooling_layer::Pooling_parameters(const std::vector<double>& vec_convolution, int input_height, int input_width){

	int Pooling_height = input_height/Pooling_size; //Height of the output pooling matrix
	int Pooling_width = input_width/Pooling_size; //Width of the output pooling matrix
	
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
