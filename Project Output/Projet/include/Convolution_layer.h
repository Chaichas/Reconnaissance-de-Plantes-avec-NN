#ifndef __CONVOLUTION_LAYER_H__
#define __CONVOLUTION_LAYER_H__

#include <iostream>
#include "Data.h"

//Input Parameters for the Convolution_layer class
#define filter_number 8  //number of filters used
#define filter_height 3 //hight of the filter
#define filter_width 3 //width of the filter
#define padding 0 //padding (explained in the descriptive)
#define stride 1 //offset
#define pooling_size 2 //pooling size

//Note: This is the case of a valid convolution (i.e., the padding is null)

class Convolution_layer{

public:
	Convolution_layer(); //Constructor
	~Convolution_layer(); //Destructor

	void convolution_parameters(const std::vector<double>& vec_pixel, int inputImage_height, int inputImage_width);

	 //Get Height and Width of the convolutional output matrix
	int getMatHeight() {return ConvMat_height;} //Convolution output matrix height
	int getMatWidth() {return ConvMat_width;} //Convolution output matrix width
	
	//Get Convolution output Matrix
	const std::vector<std::vector<double>>& getConvMat() const {return ConvMat;} 
	
private:

	std::vector<std::vector<double>> ConvMat; //convolution matrix filled with values
	
	std::vector<std::vector<double>> filter_matrix; //Filter matrix filled with values
	
	void convolution_process(const std::vector<double>& pixel, int idx); //multiplication and summation pixel value * filter value
	
	//Backpropagation: Hiding inputs
	std::vector<double> HiddenMat; //Hidding of inputs
	void Hidden(const std::vector<double>& vect);

public:

	//Case of Concolution layer propagation
	void BackPropagation(std::vector<std::vector<double>> dloss_dlayer_output, double LearningRate);

};

#endif
