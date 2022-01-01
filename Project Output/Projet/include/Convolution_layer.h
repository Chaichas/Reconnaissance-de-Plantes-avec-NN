#ifndef __CONVOLUTION_LAYER_H__
#define __CONVOLUTION_LAYER_H__

#include <iostream>


class Convolution_layer{

public:
	Convolution_layer(); //Constructor
	~Convolution_layer(); //Destructor

	void convolution_parameters(const std::vector<double>& pixel_vec, int inputImage_height, int inputImage_width);

	 //Get Height and Width of the convolutional output matrix
	int getMatHeight() {return ConvMat_height;} //Convolutional output matrix height
	int getMatWidth() {return ConvMat_width;} //Convolutional output matrix width
	
	//Get Convolution output Matrix
	const std::vector<std::vector<double>>& getConvMat() const {return ConvMat;} 
	
private

	std::vector<std::vector<double>> ConvMat; //convolution matrix filled with values
	
	std::vector<std::vector<double>> filter_matrix; //Filter matrix filled with values
	
	void convolution_process(const std::vector<double>& pixel, int idx); //multiplication and summation pixel value * filter value
	
	//Backpropagation
	std::vector<double> CacheMat; //Cache of inputs
	void cache_BK(const std::vector<double>& vect);

}

#endif
