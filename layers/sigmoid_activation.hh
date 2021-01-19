#pragma once

#include "nn_layer.hh"

class SigmoidActivation : public NNLayer {
private:
	//layer's output
	Matrix A;

	//layer's input
	Matrix Z;
	Matrix dZ;

public:
	SigmoidActivation(std::string name);
	~SigmoidActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
