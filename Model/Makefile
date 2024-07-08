PYTHON=python3
PYTHONCONFIG=python3-config
CXX=clang++

all: cppspeedup

cppspeedup:
	$(CXX) -O2 --shared --std=c++11 -fPIC `$(PYTHON) -m pybind11 --includes` cppspeedup.cpp -o cppspeedup`$(PYTHONCONFIG) --extension-suffix`
