#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sigmoid_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>

__device__ float sigmoid(float x) {
	return 1.0f / (1 + exp(-x));
}

//Z is layer's input, A is layer's output
__global__ void sigmoidActivationForward(float* Z, float* A,
										 int Z_x_dim, int Z_y_dim) {
	//calculate index for current thread
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = sigmoid(Z[index]);
	}
}

//dA is the error introduced by this layer
__global__ void sigmoidActivationBackprop(float* Z, float* dA, float* dZ,
										  int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
	}
}

SigmoidActivation::SigmoidActivation(std::string name) {
	//to access members of a structure through pointer, use arrow operator, shorthand for (*this).name
	this->name = name;
}

SigmoidActivation::~SigmoidActivation()
{ }

Matrix& SigmoidActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	sigmoidActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														   	Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid forward propagation.");

	return A;
}

Matrix& SigmoidActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	sigmoidActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
															 dZ.data_device.get(),
															 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid back propagation");

	return dZ;
}
