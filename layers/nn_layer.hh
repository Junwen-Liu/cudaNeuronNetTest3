#pragma once

#include <iostream>

#include "../nn_utils/matrix.hh"

class NNLayer {
protected:
	std::string name;

public:
	//virtual means function declared in base class and is re-defined(overriden) by a derived class
	//~function is a deconstructor function
	virtual ~NNLayer() = 0;

	virtual Matrix& forward(Matrix& A) = 0;
	virtual Matrix& backprop(Matrix& dZ, float learning_rate) = 0;

	std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}
