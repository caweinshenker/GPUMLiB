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

#include "rbm_kernels.h"

namespace GPUMLib {
	__global__ void cuda_rbm_init_bias_and_deltas(cudafloat * bias, cudafloat initialBias, cudafloat * lastDeltaW, cudafloat * lastDeltaB, cudafloat * lastDeltaWithoutLearningMomentumW, cudafloat * lastDeltaWithoutLearningMomentumB, cudafloat * learningRateW, cudafloat * learningRateB, cudafloat initialLearningRate, int weights, int J) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < weights) {
			lastDeltaW[idx] = 0;

			learningRateW[idx] = initialLearningRate;
			lastDeltaWithoutLearningMomentumW[idx] = 0;

			if (idx < J) {
				bias[idx] = initialBias;
				lastDeltaB[idx] = 0;
				lastDeltaWithoutLearningMomentumB[idx] = 0;
				learningRateB[idx] = initialLearningRate;
			}
		}
	}	

	__global__ void cuda_rbm_init_input_bias_and_deltas(cudafloat * v, cudafloat * bias, cudafloat * lastDeltaA, cudafloat * lastDeltaWithoutLearningMomentumA, cudafloat * learningRateA, cudafloat initialLearningRate, int I, int samples) {
		int input = blockIdx.x * blockDim.x + threadIdx.x;

		cudafloat sum = 0;

		if (input < I) {
			for (int s = 0; s < samples; s++) sum += v[s * I + input];

			cudafloat pi = sum / samples;
			pi = Log(pi / (1 - pi));
			bias[input] = pi;

			lastDeltaA[input] = 0;

			lastDeltaWithoutLearningMomentumA[input] = 0;
			learningRateA[input] = initialLearningRate;
		}
	}
}

extern "C" cudaError_t gpumlib_cuda_rbm_init_bias_and_deltas(unsigned int gridDim, unsigned int blockDim, cudafloat * bias, cudafloat initialBias, cudafloat * lastDeltaW, cudafloat * lastDeltaB, cudafloat * lastDeltaWithoutLearningMomentumW, cudafloat * lastDeltaWithoutLearningMomentumB, cudafloat * learningRateW, cudafloat * learningRateB, cudafloat initialLearningRate, int weights, int J) {
	GPUMLib::cuda_rbm_init_bias_and_deltas<<<gridDim, blockDim>>>(bias, initialBias, lastDeltaW, lastDeltaB, lastDeltaWithoutLearningMomentumW, lastDeltaWithoutLearningMomentumB, learningRateW, learningRateB, initialLearningRate, weights, J);

	return cudaGetLastError();
}


extern "C" cudaError_t gpumlib_cuda_rbm_init_input_bias_and_deltas(unsigned int gridDim, unsigned int blockDim, cudafloat * v, cudafloat * bias, cudafloat * lastDeltaA, cudafloat * lastDeltaWithoutLearningMomentumA, cudafloat * learningRateA, cudafloat initialLearningRate, int I, int samples) {
	GPUMLib::cuda_rbm_init_input_bias_and_deltas<<<gridDim, blockDim>>>(v, bias, lastDeltaA, lastDeltaWithoutLearningMomentumA, learningRateA, initialLearningRate, I, samples);

	return cudaGetLastError();
}
