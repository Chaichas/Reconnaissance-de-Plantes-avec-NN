#include "Convolution_layer.h"
#include "Random_weights" //Generation of random numbers

//Input Parameters for the Convolution_layer class
#define filter_number 8  //number of filters used
#define filter_height 3 //hight of the filter
#define filter_width 3 //width of the filter
#define padding 0 //padding (explained in the descriptive)
#define stride 1 //offset


Convolution_layer::Convolution_layer(){} //constructor


void Convolution_layer::convolution_parameters(const std::vector<double>& pixel_vec, int inputImage_height, int inputImage_width){

	bool initialization = true; //initializing filters with random values
	
	int ConvMat_height = ((inputImage_height - filter_height +2 *padding)/stride )+1; //output convolution matrix height
	int ConvMat_width = ((inputImage_width - filter_width +2 *padding)/stride )+1; //output convolution matrix width
	
	//initialization of the filter weights by random values (class Random_weights)
	if (initialization){
		Random_weights(nb_filters, filter_height*filter_width, filter_matrix); //initialization of weights with random values
		
		//normalizing wight values, Ref3
		for (size_t ii = 0; ii < filter_number; ii++){ //loop on number of filters
			for (size_t jj = 0; jj < (filter_height*filter_width); jj++){ //loop on total number of weights within each filter
				filter_matrix[ii][jj] = (double) filter_matrix[ii][jj]/(double) (filter_height*filter_width); //normalized filter
			}
		}
		initialization = false;
	}
	
	ConvMat.clear();
	
	//Convolution procedure for filter_number
	convolution_process(pixel_vec, 0); //1st filter
	convolution_process(pixel_vec, 1); //2sd filter
	convolution_process(pixel_vec, 2); //3rd filter
	convolution_process(pixel_vec, 3); //4th filter
	convolution_process(pixel_vec, 4); //5th filter
	convolution_process(pixel_vec, 5); //6th filter
	convolution_process(pixel_vec, 6); //7th filter
	convolution_process(pixel_vec, 7); //8th filter
}

//Convolution Process
void Convolution_layer::convolution_process(const std::vector<double>& pixel, int idx){
	std::vector<double> vec;
	
	for (int ii = 0; ii < ConvMat_height; ii++){ //loop on the height of the convolution matrix
		for (int jj = 0; jj < ConvMat_width; jj++){ //loop on the width of the convolution matrix
			
			double sum = 0; //initialization of the summation
			
				for (int kk=0; kk < filter_height; kk++){ //loop on the height of the filter
					for (int hh=0; hh < filter_width; hh++){ //loop on the width of the filter
					
						double Temp = (pixel[((ii + kk) * ((ConvMat_width - 1) * stride + filter_width - 2*padding) + (jj + hh))]); //pixel value of the image stored in Temp
						
						sum = sum + Temp * filter_matrix[idx][kk * filter_width + hh]; //sum of the pixel value * filter value
					
					}				
				}
				vec.push_back(sum); //adding sum value to vec		
		}		
	}
	ConvMat.push_back(vec); //adding vec value to ConvMat
}


//Here, we will update the weights using Backpropagation

void Convolution_layer::cache_BK(const std::vector<double>& vect){
  
	CacheMat.clear(); //Clear the old CacheMat
	CacheMat.resize(input.size()); //Resize CacheMat

	//Copy: output.assign(input.begin(), input.end());
	CacheMat.assign(input.begin(), input.end()); //output = CacheMat
}








Convolution_layer::~Convolution_layer(){} //Destructor



/* References:

Ref3: https://en.wikipedia.org/wiki/Kernel_(image_processing)

*/

