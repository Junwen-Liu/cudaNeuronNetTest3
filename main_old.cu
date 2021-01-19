//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <iostream>
//#include <time.h>
//
//#include "neural_network.hh"
//#include "layers/linear_layer.hh"
//#include "layers/relu_activation.hh"
//#include "layers/sigmoid_activation.hh"
//#include "nn_utils/nn_exception.hh"
//#include "nn_utils/bce_cost.hh"
//
//#include "coordinates_dataset.hh"
//
//float computeAccuracy(const Matrix& predictions, const Matrix& targets);
//
//int main() {
//
//	srand( time(NULL) );
//
//	//batch_size=100, number of batches=21, use 20 batches for training and 1 batch for testing(get accuracy score)
//	CoordinatesDataset dataset(100, 21);
//	BCECost bce_cost;
//
//	NeuralNetwork nn;
//	//linear layer with 2 input neuron and 30 output/hidden neurons 
//	nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
//	nn.addLayer(new ReLUActivation("relu_1"));
//	//linear layer with 30 input neurons and 1 output neuron
//	nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
//	nn.addLayer(new SigmoidActivation("sigmoid_output"));
//
//	// network training
//	Matrix Y;
//	for (int epoch = 0; epoch < 1001; epoch++) {
//		float cost = 0.0;
//
//		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
//			//get training batches as input for forward
//			Y = nn.forward(dataset.getBatches().at(batch));
//			//get label/target batches as input for backprop
//			nn.backprop(Y, dataset.getTargets().at(batch));
//			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
//		}
//
//		if (epoch % 100 == 0) {
//			std::cout 	<< "Epoch: " << epoch
//						<< ", Cost: " << cost / dataset.getNumOfBatches()
//						<< std::endl;
//		}
//	}
//
//	// compute accuracy
//	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
//	Y.copyDeviceToHost();
//
//	float accuracy = computeAccuracy(
//			Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
//	std::cout 	<< "Accuracy: " << accuracy << std::endl;
//
//	return 0;
//}
//
////count number of correctly predicted values and divide it by the size of output vector
//float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
//	int m = predictions.shape.x;
//	int correct_predictions = 0;
//
//	for (int i = 0; i < m; i++) {
//		float prediction = predictions[i] > 0.5 ? 1 : 0;
//		if (prediction == targets[i]) {
//			correct_predictions++;
//		}
//	}
//
//	return static_cast<float>(correct_predictions) / m;
//}
