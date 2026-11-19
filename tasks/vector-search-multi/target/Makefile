CXX := /usr/bin/clang++
CXXFLAGS := -O3 -mcpu=native -DNDEBUG -std=c++20

.PHONY: all clean
all: engine
engine: driver.cpp engine.h $(wildcard src/*.h)
	$(CXX) $(CXXFLAGS) -I. driver.cpp -framework Accelerate -o engine
clean:
	rm -f engine
