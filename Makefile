#Makefile 

# project name
TARGET = CNN

CC=mpiCC

# Compiling flags
#CFLAGS = -O3 -march=native -Wall -mavx -g
CFLAGS = -O3

# linking flags & libraries
LFLAGS = -I/usr/include/opencv4/ -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
#LFLAGS = -I/usr/include/opencv4/ -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lgomp  

# Proper directories 
SRCDIR = src
INCDIR = include


all: program

program : main convolution_layer Data Pooling_layer output softmax_layer
	$(CC)  $(CFLAGS) -o $@ $^  $(LFLAGS)

#Run_program: program
#	./program

main : ${SRCDIR}/main.cpp convolution_layer output Data Pooling_layer softmax_layer
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)

convolution_layer: ${SRCDIR}/Convolution_layer.cpp ${INCDIR}/Convolution_layer.h
	$(CC)  $(CFLAGS) -c $< -o $@  $(LFLAGS)
	
output : ${SRCDIR}/output.cpp ${INCDIR}/Output.h 
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)
	
Data : ${SRCDIR}/Data.cpp ${INCDIR}/Data.h
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)
	
Pooling_layer : ${SRCDIR}/Pooling_layer.cpp ${INCDIR}/Pooling_layer.h
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)
	
softmax_layer : ${SRCDIR}/softmax_layer.cpp ${INCDIR}/softmax_layer.h ${INCDIR}/Convolution_layer.h
	$(CC)  $(CFLAGS) -c $< -o $@ $(LFLAGS)


clean:
	rm -Rf *~ program main convolution_layer Data Pooling_layer output softmax_layer *.optrpt

