#Makefile 

# project name
TARGET = CNN

LINKER = g++ 


LFLAGS = -I/usr/include/opencv4/ -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lgomp 

CFLAGS = 

# Proper directories 
SRCDIR = include


all: program main convolution_layer Data Pooling_layer program output softmax_layer 

program : main convolution_layer Data Pooling_layer program output softmax_layer
	$(LINKER)  $(OPTFLAGS) -o $@ $^  $(LFLAGS)

Run_program: program
	./program

convolution_layer: Convolution_layer.cpp ${SRCDIR}/Convolution_layer.h
	$(LINKER)  $(OPTFLAGS) -c $< -o $@  $(LFLAGS)
	
output : output.cpp ${SRCDIR}/Output.h 
	$(LINKER)  $(OPTFLAGS) -c $< -o $@ $(LFLAGS)
	
Data : Data.cpp ${SRCDIR}/Data.h
	$(LINKER)  $(OPTFLAGS) -c $< -o $@ $(LFLAGS)
	
Pooling_layer : Pooling_layer.cpp ${SRCDIR}/Pooling_layer.h
	$(LINKER)  $(OPTFLAGS) -c $< -o $@ $(LFLAGS)
	
main : main.cpp ${SRCDIR}/Convolution_layer.h ${SRCDIR}/Data.h ${SRCDIR}/Pooling_layer.h ${SRCDIR}/softmax_layer.h
	$(LINKER)  $(OPTFLAGS) -c $< -o $@ $(LFLAGS)
	
softmax_layer : softmax_layer.cpp ${SRCDIR}/softmax_layer.h ${SRCDIR}/Convolution_layer.h
	$(LINKER)  $(OPTFLAGS) -c $< -o $@ $(LFLAGS)


clean:
	rm -Rf *~ Run_program main convolution_layer Data Pooling_layer program output softmax_layer *.optrpt

