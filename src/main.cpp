#include <string>
#include <omp.h>
#include "../include/Output.h"

int main()
{
        double start,end;
	std::string trainingPath = "trainset_3665";
	std::string testingPath = "testset_700";
	output* cnn = new output(trainingPath, testingPath);
	// Pour une grande precision, on doit definir plus d'epoch
	// Le deuxieme parametre est le taux d'apprentissage
	start = omp_get_wtime();
	cnn->Training_data(1, 0.003);
	cnn->Testing_data();
	end =omp_get_wtime();
	printf("work took %f seconds\n", end - start);
	delete cnn;
	return 0;
}
