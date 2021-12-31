#include "Random_weights.h"

void Random_weights(double nb_filters, double nb_weights, std::vector<std::vector<double>>& nb_tot){
	
	//construct a random generator engine from a time-based seed, Ref1
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(); //time; system real time
	
	std::default_random_engine generator(seed); //random generator engine
	std::normal_distribution<double> distribution(0.0,1.0); //( result_type mean = 0.0, result_type stddev = 1.0 )
	
	
	for(int ii = 0; ii < nb_filters; ii++){ //loop on the total number of filters
	
	std::vector<double> one_filter; //array initialisation with no defined size

	for (int jj = 0; jj < nb_weights; jj++) { //nb_weights = filter height * filter width
    
      double number = (distribution(generator)); //random number from a random generator engine, Ref2
      
      one_filter.push_back(number); // Filling of 1 filter
     
    }
    
    nb_tot.push_back(temp); //Filling of all filters
  }
  
}


/*	References:

	Ref1: https://www.cplusplus.com/reference/random/normal_distribution/normal_distirbution/
	
	Ref2: https://www.cplusplus.com/reference/random/normal_distribution/
	
 */
