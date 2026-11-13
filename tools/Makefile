CXX ?= g++
CXXFLAGS = -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp -Wall -Wextra -Werror

TARGET = ../bin/vultorch-codec

all: $(TARGET)

$(TARGET): vultorch_codec.cpp
	mkdir -p ../bin
	$(CXX) $(CXXFLAGS) vultorch_codec.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)
