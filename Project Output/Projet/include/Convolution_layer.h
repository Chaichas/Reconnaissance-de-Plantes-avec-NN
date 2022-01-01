#ifndef __CONVOLUTION_LAYER_H__
#define __CONVOLUTION_LAYER_H__

#include <iostream>


class Convolution_layer{

public:
	Convolution_layer(); //Constructor
	~Convolution_layer(); //Destructor

	void convolution_parameters(const std::vector<double>& kernel, int inputImage_height, int inputImage_width);

	 //Get Height and Width of the convolutional output matrix
	int getMatHeight() {return ConvMat_height;} //Convolutional output matrix height
	int getMatWidth() {return ConvMat_width;} //Convolutional output matrix width

}

#endif
