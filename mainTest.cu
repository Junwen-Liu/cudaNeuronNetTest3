#include <map>
#include <sstream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"

#include "coordinates_dataset.hh"
#include <iostream>

#include "cuda_dll.h"
#include "mainTest3.h"

using namespace std;
using std::map;
using std::pair;
using std::vector;
using std::stringstream;

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

void JavaHashMapToJNIHashMap(JNIEnv *env, jobject hashMap, std::map<std::string, std::string>& mapOut) {
	// Get the Map's entry Set.
	jclass mapClass = env->FindClass("java/util/Map");
	if (mapClass == NULL) {
	  return;
	}
	jmethodID entrySet =
	  env->GetMethodID(mapClass, "entrySet", "()Ljava/util/Set;");
	if (entrySet == NULL) {
	  return;
	}
	jobject set = env->CallObjectMethod(hashMap, entrySet);
	if (set == NULL) {
	  return;
	}
	// Obtain an iterator over the Set
	jclass setClass = env->FindClass("java/util/Set");
	if (setClass == NULL) {
	  return;
	}
	jmethodID iterator =
	  env->GetMethodID(setClass, "iterator", "()Ljava/util/Iterator;");
	if (iterator == NULL) {
	  return;
	}
	jobject iter = env->CallObjectMethod(set, iterator);
	if (iter == NULL) {
	  return;
	}
	// Get the Iterator method IDs
	jclass iteratorClass = env->FindClass("java/util/Iterator");
	if (iteratorClass == NULL) {
	  return;
	}
	jmethodID hasNext = env->GetMethodID(iteratorClass, "hasNext", "()Z");
	if (hasNext == NULL) {
	  return;
	}
	jmethodID next =
	  env->GetMethodID(iteratorClass, "next", "()Ljava/lang/Object;");
	if (next == NULL) {
	  return;
	}
	// Get the Entry class method IDs
	jclass entryClass = env->FindClass("java/util/Map$Entry");
	if (entryClass == NULL) {
	  return;
	}
	jmethodID getKey =
	  env->GetMethodID(entryClass, "getKey", "()Ljava/lang/Object;");
	if (getKey == NULL) {
	  return;
	}
	jmethodID getValue =
	  env->GetMethodID(entryClass, "getValue", "()Ljava/lang/Object;");
	if (getValue == NULL) {
	  return;
	}
	// Iterate over the entry Set
	while (env->CallBooleanMethod(iter, hasNext)) {
	  jobject entry = env->CallObjectMethod(iter, next);
	  jstring key = (jstring) env->CallObjectMethod(entry, getKey);
	  jstring value = (jstring) env->CallObjectMethod(entry, getValue);
	  const char* keyStr = env->GetStringUTFChars(key, NULL);
	  if (!keyStr) {  // Out of memory
		return;
	  }
	  const char* valueStr = env->GetStringUTFChars(value, NULL);
	  if (!valueStr) {  // Out of memory
		env->ReleaseStringUTFChars(key, keyStr);
		return;
	  }
  
	  mapOut.insert(std::make_pair(string(keyStr), string(valueStr)));
  
	  env->DeleteLocalRef(entry);
	  env->ReleaseStringUTFChars(key, keyStr);
	  env->DeleteLocalRef(key);
	  env->ReleaseStringUTFChars(value, valueStr);
	  env->DeleteLocalRef(value);
	}

}


JNIEXPORT jboolean JNICALL Java_mainTest3_simpleNN(JNIEnv *env, jobject, jobject hashMap){
// JNIEXPORT jboolean JNICALL Java_mainTest_simpleNN(JNIEnv *, jobject, jint numOfLayers, jobjectArray layerSpecs);
//int mainTest(map<int, int*> inputMap) {

	//Todo: parse jobject to hashMap 
	map<string, string>stringMap;
	map<int, vector<string>>inputMap;
	JavaHashMapToJNIHashMap(env, hashMap, stringMap);

	//parse string hashMap to int Hashmap
	map<string,string>::iterator it;
	for(it = stringMap.begin(); it != stringMap.end(); it++)
	{
		//parse second string to vector
		vector<string> vect;
		std::stringstream ss(it->second);

		while(ss.good())
		{
			string substr;
			getline(ss, substr, ',');
			vect.push_back( substr);
		}
		inputMap.insert(std::pair<int, vector<string>>(stoi(it->first), vect));
	}

	srand( time(NULL) );

	//batch_size=100, number of batches=21, use 20 batches for training and 1 batch for testing(get accuracy score)
	CoordinatesDataset dataset(100, 21);
	BCECost bce_cost;

	NeuralNetwork nn;

	//map<int, vector<string>>::iterator it;

	for (int i = 0; i < inputMap.size(); i++)
	{
		// it = inputMap.find(i);
		// if(it!=inputMap.end())
		vector<string> val = inputMap[i];

		switch(stoi(val[0])){
			case 0: 
				nn.addLayer(new LinearLayer("linear_"+i, Shape(stoi(val[1]), stoi(val[2]))));
				cout << "A linearLayer_" << i << " layer has been added in Nueron Net\n";
				break;
			case 1: 
				nn.addLayer(new ReLUActivation("relu_"+i));
				cout << "A relu_" << i << " layer has been added in Nueron Net\n";
				break;
			case 2:
				nn.addLayer(new SigmoidActivation("sigmoid_output_"+i));
				cout << "A sigmoid_" << i << " layer has been added in Nueron Net\n";
				break;
			default:
				break;
		}

	}

	cout << "The Nueron Net has completed construction, start the training for 1000 epochs...\n";

	////linear layer with 2 input neuron and 30 output/hidden neurons 
	//nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
	//nn.addLayer(new ReLUActivation("relu_1"));
	////linear layer with 30 input neurons and 1 output neuron
	//nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
	//nn.addLayer(new SigmoidActivation("sigmoid_output"));

	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			//get training batches as input for forward
			Y = nn.forward(dataset.getBatches().at(batch));
			//get label/target batches as input for backprop
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / dataset.getNumOfBatches()
						<< std::endl;
		}
	}

	// compute accuracy
	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	Y.copyDeviceToHost();

	float accuracy = computeAccuracy(
			Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	std::cout 	<< "Accuracy: " << accuracy << std::endl;

	return true;
	}

//count number of correctly predicted values and divide it by the size of output vector
float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}

// Testing purpose in visual studio
// void main() {
// 	map<int, int*> testdata;
// 	int layer1[] = { 0,2,30 };
// 	testdata[0] = layer1;
// 	int layer2[] = { 1 };
// 	testdata[1] = layer2;
// 	int layer3[] = { 0,30,1 };
// 	testdata[2] = layer3;
// 	int layer4[] = { 2 };
// 	testdata[3] = layer4;

// 	int test = mainTest(testdata);
// 	cout << "final result is" << test;
// }

