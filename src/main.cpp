#include <string>
#include "../include/Output.h"
#include <mpi.h> //AM: MPI addition

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv); //AM: initializing MPI

	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	std::string trainingPath = "trainset_3665";
	std::string testingPath = "testset_700";
	output* cnn = new output(trainingPath, testingPath);
	
	cnn->Training_data(3, 0.003);
	cnn->Testing_data();
	delete cnn;

	MPI_Finalize(); //AM: Ending MPI
	return 0;
}
