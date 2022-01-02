#ifndef __POOLING_LAYER_H__
#define __POOLING_LAYER_H__

#include <iostream>
#include <vector> //vector


class Pooling_layer{

public:

	Pooling_layer(); //Constructor
	~Pooling_layer(); //Destructor
	
	void Pooling_parameters(const std::vector<double>& vec_convolution, int input_height, int input_width);
	
};

#endif
