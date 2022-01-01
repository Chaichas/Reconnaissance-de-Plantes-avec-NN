#include "Convolution_layer.h"
#include "Random_weights" //Generation of random numbers

//Input Parameters for the Convolution_layer class
#define filter_number 8  //number of filters used
#define filter_height 3 //hight of the filter
#define filter_width 3 //width of the filter
#define padding 0 //padding (explained in the descriptive)
#define stride 1 //offset


Convolution_layer::Convolution_layer(){} //constructor



void Convolution_layer::convolution_parameters(const std::vector<double>& kernel, int inputImage_height, int inputImage_width){

	bool initialization = true; //initializing filters with random values
	
	ConvMat_height = ((inputImage_height - filter_height +2 *padding)/stride )+1; //output convolution matrix height
	ConvMat_width = ((inputImage_width - filter_width +2 *padding)/stride )+1; //output convolution matrix width
	
	//initialization of the filter weights by random values (class Random_weights)
	if (initialization){
		Random_weights(nb_filters, filter_height*filter_width, filter_matrix); //initialization of weights with random values
		
		//normalizing wight values, Ref3
		for (size_t ii=0; ii < filter_number; ii++){ //loop on number of filters
			for (size_t jj=0; jj < (filter_height*filter_width); jj++){ //loop on total number of weights within each filter
				filter_matrix[ii][jj] = (double) filter_matrix[ii][jj]/(double) (filter_height*filter_width); //normalized filter
			}
		}
		initialization = false;
	}
	
}
	



/* References:

Ref3: https://en.wikipedia.org/wiki/Kernel_(image_processing)

*/
