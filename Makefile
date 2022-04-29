#Makefile 

# project name
TARGET = CNN

CC = g++ 

# Compiling flags
CFLAGS = -Ofast -march=native -funroll-loops -finline-functions -Wall

# linking flags & libraries
LFLAGS = -I/usr/include/opencv4/ -fopenmp -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lgomp 

# Proper directories 
SRCDIR = src
INCDIR = include


all: program

program : main convolution_layer Data Pooling_layer output softmax_layer
	$(CC)  $(CFLAGS) -o $@ $^  $(LFLAGS)

Run_program: program
	./program

convolution_layer: ${SRCDIR}/Convolution_layer.cpp ${INCDIR}/Convolution_layer.h
	$(CC)  $(CFLAGS) -c $< -o $@  $(LFLAGS)
	
output : ${SRCDIR}/output.cpp ${INCDIR}/Output.h 
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)
	
Data : ${SRCDIR}/Data.cpp ${INCDIR}/Data.h
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)
	
Pooling_layer : ${SRCDIR}/Pooling_layer.cpp ${INCDIR}/Pooling_layer.h
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)
	
main : ${SRCDIR}/main.cpp ${INCDIR}/Convolution_layer.h ${INCDIR}/Data.h ${INCDIR}/Pooling_layer.h ${INCDIR}/softmax_layer.h
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)
	
softmax_layer : ${SRCDIR}/softmax_layer.cpp ${INCDIR}/softmax_layer.h ${INCDIR}/Convolution_layer.h
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)


clean:
	rm -Rf *~ program main convolution_layer Data Pooling_layer output softmax_layer *.optrpt

