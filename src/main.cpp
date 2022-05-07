#include <string>
#include "../include/Output.h"
#include <mpi.h> //AM: Adding MPI

int main(int argc, char **argv)
{
	//AM: Initializing MPI
	MPI_Init(&argc, &argv);

    //AM: Initializing and getting MPI rank 
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	std::string trainingPath = "trainset_3665";
	std::string testingPath = "testset_700";

	output* cnn = new output(trainingPath, testingPath);

	cnn->Training_data(3, 0.003); //Training_data(epochs, Learning_rate)

	//AM: Testing data is done only on 1 proc
	if (rank == 0) {
		cnn->Testing_data();
	}

	delete cnn;

	//AM: Finalizing MPI
	MPI_Finalize();
	return 0;
}
