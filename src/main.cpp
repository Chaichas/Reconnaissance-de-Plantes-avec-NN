#include <string>
#include "../include/Output.h"
#include <mpi.h>

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	std::string trainingPath = "trainset_3665";
	std::string testingPath = "testset_700";
	output* cnn = new output(trainingPath, testingPath);
	// Pour une grande precision, on doit definir plus d'epoch
	// Le deuxieme parametre est le taux d'apprentissage
	cnn->Training_data(3, 0.003);
	cnn->Testing_data();
	delete cnn;
	MPI_Finalize();
	return 0;
}
