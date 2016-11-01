/*
	Noel Lopes is a Professor at the Polytechnic of Guarda, Portugal
	and a Researcher at the CISUC - University of Coimbra, Portugal
	Copyright (C) 2009-2015 Noel de Jesus Mendonça Lopes

	This file is part of GPUMLib.

	GPUMLib is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <exception>
#include "MBPutils.h"
#include "../common/OutputFile.h"

void SaveSelectiveInputWeights(OutputFile & f, HostArray<cudafloat> & weights, HostArray<cudafloat> & bias) {
	size_t numWeights = weights.Length();

	for (size_t w = 0; w < numWeights; w++) {
		if (weights[w] != CUDA_VALUE(0.0) || bias[w] != CUDA_VALUE(0.0)) {
			f.WriteLine(bias[w]);
			f.WriteLine("0.0"); //delta
			f.WriteLine("0.0"); //deltaWithoutLearningMomentum
			f.WriteLine(INITIAL_LEARNING_RATE);

			f.WriteLine(weights[w]);
			f.WriteLine("0.0"); //delta
			f.WriteLine("0.0"); //deltaWithoutLearningMomentum
			f.WriteLine(INITIAL_LEARNING_RATE);
		}
	}
}

void SaveWeights(OutputFile & f, HostArray<cudafloat> & weights) {
	size_t numWeights = weights.Length();

	for (size_t w = 0; w < numWeights; w++) {
		f.WriteLine(weights[w]);
		f.WriteLine("0.0"); //delta
		f.WriteLine("0.0"); //deltaWithoutLearningMomentum
		f.WriteLine(INITIAL_LEARNING_RATE);
	}
}

void SaveMBPNetwork(const char * filename, BackPropagation * network, const char * trainfilename, const char * testfilename, float rmsStop, int epochs, bool MBP) {
	OutputFile f(filename);

	const char * MBPVersion = "Multiple Back-Propagation Version 2.2.1";

	f.WriteLine(MBPVersion);
	f.WriteLine("Multiple Back-Propagation can be freely obtained at http://dit.ipg.pt/MBP - This file was generated by the ATS.");
	f.WriteLine(trainfilename);
	f.WriteLine(testfilename);

	f.WriteLine("32"); // priority normal
	f.WriteLine("0"); // update screen

	f.WriteLine("1"); // delta bar delta
	f.WriteLine(network->GetUpStepSizeFactor());
	f.WriteLine(network->GetDownStepSizeFactor());
	f.WriteLine(network->GetMaxStepSize());

	f.WriteLine(network->GetRobustLearning());
	f.WriteLine(network->GetRobustFactor());
	f.WriteLine(1.0 + network->GetMaxPercentageRMSGrow()); // rmsGrowToApplyRobustLearning

	f.WriteLine("0.0"); // weightDecay

	f.WriteLine("0"); // autoUpdateLearning
	f.WriteLine("0"); // autoUpdateMomentum

	f.WriteLine("0.01"); //percentIncDecLearnRate
	f.WriteLine("0.01"); //percentIncDecMomentum
	f.WriteLine("0.01"); //percentIncDecSpaceLearnRate
	f.WriteLine("0.01"); //percentIncDecSpaceMomentum

	f.WriteLine(INITIAL_LEARNING_RATE); //mainNetLearningMomentumInformation.learningRate.value
	f.WriteLine("1000"); //mainNetLearningMomentumInformation.learningRate.decayEpochs
	f.WriteLine("1"); //mainNetLearningMomentumInformation.learningRate.decayPercentage

	f.WriteLine(network->GetMomentum()); //mainNetLearningMomentumInformation.momentum.value
	f.WriteLine("0"); //mainNetLearningMomentumInformation.momentum.decayEpochs
	f.WriteLine("0"); //mainNetLearningMomentumInformation.momentum.decayPercentage

	f.WriteLine(INITIAL_LEARNING_RATE); //spaceNetLearningMomentumInformation.learningRate.value
	f.WriteLine("1000"); //spaceNetLearningMomentumInformation.learningRate.decayEpochs
	f.WriteLine("1"); //spaceNetLearningMomentumInformation.learningRate.decayPercentage

	f.WriteLine(network->GetMomentum()); //spaceNetLearningMomentumInformation.momentum.value
	f.WriteLine("0"); //spaceNetLearningMomentumInformation.momentum.decayEpochs
	f.WriteLine("0"); //spaceNetLearningMomentumInformation.momentum.decayPercentage

	f.WriteLine("0"); //epochsStop
	f.WriteLine(rmsStop);
	f.WriteLine(epochs); // numberEpochsToStop

	f.WriteLine("0.0"); //spaceRmsStop

	f.WriteLine("1"); //batchTraining
	f.WriteLine("0"); //randomizePatterns

	f.WriteLine((MBP) ? 3 : 0); //Network Type

	//main network
	f.Write(network->GetNumberInputs());

	int numLayers = network->GetNumberLayers();
	for(int l = 0; l < numLayers; l++) {
		f.Write("-");
		f.Write(network->GetNumberNeurons(l));
	}
	f.WriteLine();

	//space network additional layers
	f.WriteLine("");

	// layers information
	for(int l = 0; l < numLayers; l++) {
		int numNeurons = network->GetNumberNeurons(l);

		f.WriteLine((l == 0 && MBP) ? numNeurons : 0); // NeuronsWithSelectiveActivation

		for(int n = 0; n < numNeurons; n++) {
			f.WriteLine("0"); // ActivationFunction
			f.WriteLine("1.0"); //ActivationFunctionParameter
		}
	}

	#ifndef GPUMLIB_DBN
	// space layers information
	if (MBP) {
		int numNeurons = network->GetNumberNeurons(0);
		for(int n = 0; n < numNeurons; n++) {
			f.WriteLine("0"); // ActivationFunction
			f.WriteLine("1.0"); //ActivationFunctionParameter
		}
	}
	#endif

	f.WriteLine("0"); //ConnectInputLayerWithOutputLayer main
	f.WriteLine("0"); //ConnectInputLayerWithOutputLayer space

	if (network->HasSelectiveInputs()) {
		HostArray<cudafloat> weights = network->GetSelectiveInputWeights();
		HostArray<cudafloat> bias = network->GetSelectiveInputBias();

		SaveSelectiveInputWeights(f, weights, bias);
	}

	for(int l = 0; l < numLayers; l++) {
		HostArray<cudafloat> weights = network->GetLayerWeights(l);
		SaveWeights(f, weights);
	}

	#ifndef GPUMLIB_DBN
	if (MBP) {
		MultipleBackPropagation * mbpnet = static_cast<MultipleBackPropagation *>(network);

		if (mbpnet->HasSelectiveInputs()) {
			HostArray<cudafloat> weights = mbpnet->GetSelectiveInputWeightsSpaceNetwork();
			HostArray<cudafloat> bias = mbpnet->GetSelectiveInputBiasSpaceNetwork();

			SaveSelectiveInputWeights(f, weights, bias);
		}

		HostArray<cudafloat> weights = mbpnet->GetLayerWeightsSpaceNetwork(0);
		SaveWeights(f, weights);
	}
	#endif

	f.WriteLine("0"); //epoch must be 0
	f.WriteLine("0"); //rmsInterval
	f.WriteLine("0"); //trainingTime
}

#ifdef MNIST_SMALL_DATASET
void AnalyzeTestData(BackPropagation * network, HostMatrix<cudafloat> & inputs, HostMatrix<cudafloat> & desiredOutputs, ConfusionMatrix & cm, ConfusionMatrix & cm_small) {
	cm_small.Reset();
#else
void AnalyzeTestData(BackPropagation * network, HostMatrix<cudafloat> & inputs, HostMatrix<cudafloat> & desiredOutputs, ConfusionMatrix & cm) {
#endif
	HostMatrix<cudafloat> outputs = network->GetOutputs(inputs);

	cm.Reset();

	int nOutputs = (int)outputs.Columns();
	int patterns = (int)outputs.Rows();

	cout << endl;

	for(int p = 0; p < patterns; p++) {
		int predicted = -1;
		int correct = -1;

		double max = -1.0;

		if(nOutputs > 1) {
			for (int o = 0; o < nOutputs; o++) {
				if (desiredOutputs(p, o) >= CUDA_VALUE(0.5)) correct = o;

				double output = outputs(p, o);
				if (output > max) {
					predicted = o;
					max = output;
				}
			}
		} else {
			predicted = (outputs(p, 0) >= 0.5) ? 1 : 0;
			correct = (desiredOutputs(p, 0) >= 0.5) ? 1 : 0;
		}

		if (correct < 0) {
			throw exception();
		} else {
			cm.Classify(correct, predicted);

			#ifdef MNIST_SMALL_DATASET
			if (p >= patterns - 10000) cm_small.Classify(correct, predicted);
			#endif
		}
	}
}
