/*
	Ricardo Quintas is an MSc Student at the University of Coimbra, Portugal
	Copyright (C) 2009, 2010 Ricardo Quintas

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
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
	*/

#include <cuda_runtime.h>

#include <iostream> 
#include <sstream>
#include <stdio.h>
#include <string.h>

#include "../../memory/HostMatrix.h"

#include "../Dataset/Dataset.h"
#include "../../RAN/ResourceAllocatingNetwork.h"
#include "../../RBF/utils.h"
#include "../common/CudaInit.h"

#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

#define DEFAULT_KFOLDS 5
#define DEFAULT_SCALING_FACTOR 1.0f

#define SCALE_OF_INTEREST 5
#define DESIRED_ACCURACY 0.3

bool InitCUDA(void) {
	int count = 0;

	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	cudaSetDeviceFlags(cudaDeviceMapHost);

	CudaDevice device;
	if (!device.SupportsCuda()) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}

	device.ShowInfo();

	//printf("CUDA initialized.\n");
	return true;
}

/*******/
void writeHeader(HostMatrix<float> &Input, int number_classes, int seed, int numberKfolds) {

	cout << "\n=== Run information ===\n\n";

	cout << std::left << setw(50) << "Random Seed" << std::left << setw(20) << seed << endl;
	//cout << std::left << setw(50) << "Number of Basis Functions" << std::left << setw(20) << networkSize << endl;
	//cout << std::left << setw(50) << "Number of Neighbours for Width Estimation" << std::left << setw(20) << rneighbours << endl;
	cout << std::left << setw(50) << "Number of Folds" << std::left << setw(20) << numberKfolds << endl;
	cout << std::left << setw(50) << "Number of Attributes" << std::left << setw(20) << Input.Columns() << endl;
	cout << std::left << setw(50) << "Number of Classes" << std::left << setw(20) << number_classes << endl;
	cout << std::left << setw(50) << "Number of Instances" << std::left << setw(20) << Input.Rows() << endl;

}


void writeFooter(float center_time, float width_time, float weight_time, float scaling_time, unsigned int training_time, unsigned int testing_time, unsigned int time_total) {

	std::cout << "\nCenter Selection: " << (center_time) / 1000
		<< "\nWidth Selection: " << (width_time) / 1000
		<< "\nWeight Selection: " << (weight_time) / 1000 << "\n";

	std::cout << endl << "Total Time: " << time_total / 1000 << std::endl;

}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cout << "Incorrect number of parameters. Usage:" << endl;
		cout << "RAN <data filename> [<number of folds (default " << DEFAULT_KFOLDS << ")>][<scaling factor for the neuron width(default " << DEFAULT_SCALING_FACTOR << ")>]" << endl;
		return 0;
	}

	string file_name = argv[1];

	int numberKfolds = DEFAULT_KFOLDS;
	if (argc >= 3) numberKfolds = atoi(argv[2]);

	float scalingFactor = DEFAULT_SCALING_FACTOR;
	if (argc >= 4) scalingFactor = atoi(argv[3]);

	HostMatrix<float> X;
	HostMatrix<float> X_test;
	HostMatrix<float> Y;
	HostMatrix<float> Y_test;

	HostMatrix<float> Input;
	HostMatrix<float> Target;

	std::map<string, int> Classes;
	std::map<int, string> ClassesLookup;


	readFile(file_name, Input, Target, Classes, ClassesLookup);

	int kfold = 1;

	int correct_instances = 0;
	int incorrect_instances = 0;
	int total_instances = 0;

	int **confusionMatrix;


	confusionMatrix = (int**)malloc(sizeof(int*)*Classes.size());

	for (int i = 0; i < (int)Classes.size(); i++) {
		confusionMatrix[i] = (int*)malloc(sizeof(int)*Classes.size());
		memset(confusionMatrix[i], 0, sizeof(int)*Classes.size());
	}


	//float Pet_mean = 0;
	//float Ped_mean = 0;

	unsigned int seed = (unsigned)time(0);

	/***************RUN INFORMATION*************/

	writeHeader(Input, (int)Classes.size(), seed, numberKfolds);

	/*******************************************/


	if (!InitCUDA()) {
		return 1;
	}

	culaStatus status = culaInitialize();


	std::cout << "Starting " << std::endl;

	float center_time = 0;
	float width_time = 0;
	float weight_time = 0;
	float scaling_time = 0;

	unsigned int time_total = 0;
	unsigned int testing_time = 0;
	unsigned int training_time = 0;

	clock_t initialTimeTotal = clock();

	do {
		//clock_t sdata_time = clock();

		X = crossvalidationTrain(Input, numberKfolds, kfold);
		X_test = crossvalidationTest(Input, numberKfolds, kfold);
		Y = crossvalidationTrain(Target, numberKfolds, kfold);
		Y_test = crossvalidationTest(Target, numberKfolds, kfold);

		HostMatrix<float> Weights;
		HostMatrix<float> Centers;

		/*Train Network*/

		clock_t initialTime = clock();

		//float scale_of_interest_min = cudafloat(0.07);
		float scale_of_interest_max;
		//float decay = cudafloat(0.87);
		float desired_accuracy;
		//float alpha = cudafloat(0.02);
		float overlap_factor;

		scale_of_interest_max = SCALE_OF_INTEREST;
		overlap_factor = scalingFactor;
		desired_accuracy = cudafloat(DESIRED_ACCURACY);

		int *used;
		used = (int*)malloc(sizeof(int)*X.Rows());

		for (int i = 0; i < (int)X.Rows(); i++) {
			used[i] = i;
		}

		srand(seed);
		for (int i = 0; i < (int)X.Rows(); i++) {
			int r = rand() % X.Rows();
			int t = used[r];
			used[r] = used[i];
			used[i] = t;
		}


		DeviceMatrix<float> dX(X);
		DeviceMatrix<float> dY(Y);
		DeviceMatrix<float> dX_test(X_test);
		DeviceMatrix<float> dY_test(Y_test);

		int NumClasses = (int)Classes.size();

		HostMatrix<float>TargetArr(Y.Rows(), NumClasses);
		memset(TargetArr.Pointer(), 0, sizeof(float)*TargetArr.Elements());

		for (int i = 0; i < (int)Y.Rows(); i++) {
			TargetArr(i, (int)(Y(i, 0) - 1)) = 1;
		}
		DeviceMatrix<float>dTargetArr(TargetArr);



		ResourceAllocatingNetwork RAN(scale_of_interest_max, desired_accuracy, overlap_factor, (int)X.Rows(), (int)X.Columns(), NumClasses);
		RAN.FindMaxWidth(dX, dY);

		for (int i = 0; i < (int)dX.Rows(); i++) {
			int idx = used[i];
			std::cerr << "T ";

			RAN.Train(&(dX.Pointer()[idx*dX.Columns()]), (int)dX.Columns(), Y(idx, 0), &(dTargetArr.Pointer()[idx*dTargetArr.Columns()]));
			std::cerr << i << " " << RAN.GetNumCenters() << std::endl;

		}


		free(used);

		std::cout << "Size " << RAN.GetNumCenters() << std::endl;

		std::cout << "Finished " << std::endl;

		training_time = (clock() - initialTime);

		center_time += RAN.center_time;
		width_time += RAN.width_time;
		weight_time += RAN.weight_time;
		scaling_time += RAN.scaling_time;


		/*Test Network*/

		initialTime = clock();

		std::cout << "Testing" << std::endl;

		HostMatrix<float> out_test(X_test.Rows(), 1);


		/*Classification problem*/

		std::cout << "Testing " << std::endl;

		for (int i = 0; i < (int)X_test.Rows(); i++) {

			float* out = RAN.CalculateNetworkActivation(&(dX_test.Pointer()[i*dX_test.Columns()]), (int)dX_test.Columns());

			float* result = (float*)malloc(sizeof(float)*Classes.size());
			cudaMemcpy(result, out, sizeof(float)*Classes.size(), cudaMemcpyDeviceToHost);


			float max = 0;
			float out_class = 0;
			for (int j = 0; j < (int)Classes.size(); j++) {
				if (result[j] > max) {
					out_class = (cudafloat)j;
					max = (cudafloat)result[j];
				}
			}

			out_test(i, 0) = out_class + 1;

			std::cout << out_test(i, 0) << " ";



			std::cout << std::endl;

			for (int i = 0; i < (int)out_test.Rows(); i++) {

				out_test(i, 0) = (cudafloat)round(out_test(i, 0));

				if (out_test(i, 0) <= 0) out_test(i, 0) = 1;

				if (out_test(i, 0) > Classes.size()) out_test(i, 0) = (cudafloat)Classes.size();

			}

			correct_instances += (int)out_test.Rows() - error_calc(Y_test, out_test);
			incorrect_instances += error_calc(Y_test, out_test);
			total_instances += (int)out_test.Rows();

			/*Add values to Confusion Matrix*/
			for (int i = 0; i < (int)Y_test.Rows(); i++) {
				confusionMatrix[((int)Y_test(i, 0)) - 1][((int)out_test(i, 0)) - 1] = confusionMatrix[((int)Y_test(i, 0)) - 1][((int)out_test(i, 0)) - 1] + 1;
			}

		}

		testing_time = (clock() - initialTime);

		/*Increment fold number, for use in crossvalidation*/
		kfold++;
	} while (kfold <= numberKfolds);

	time_total = (clock() - initialTimeTotal);

	/*****************MEASURES****************/

	/*Classification problem*/
	measures(correct_instances, total_instances, incorrect_instances, confusionMatrix, Classes, ClassesLookup);

	writeFooter(center_time, width_time, weight_time, scaling_time, training_time, testing_time, time_total);

	cudaThreadExit();
	culaShutdown();

	return 0;
}


