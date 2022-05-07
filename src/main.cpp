#include <string>
#include "../include/Output.h"
#include <mpi.h>

int main(int argc, char **argv)
{
	//AM: Initializing the MPI
	MPI_Init(&argc, &argv);

	std::string trainingPath = "trainset_3665";
	std::string testingPath = "testset_700";
	output* cnn = new output(trainingPath, testingPath);

	cnn->Training_data(3, 0.003);
	cnn->Testing_data();
	delete cnn;

	//AM: Finalizing the MPI
	MPI_Finalize();
	return 0;
}
