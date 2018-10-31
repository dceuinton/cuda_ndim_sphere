CC = nvcc
BIN = a5

all: ${BIN}

clean:
	rm main
	rm *.o

a5: 
	${CC} -o main a5.cu