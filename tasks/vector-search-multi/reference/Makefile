CXX := /usr/bin/clang++
CXXFLAGS := -O3 -mcpu=native -DNDEBUG -std=c++20
.PHONY: all
all:
	$(CXX) $(CXXFLAGS) -I. -Ivendor/include -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp driver.cpp vendor/libfaiss.a -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate -o engine
