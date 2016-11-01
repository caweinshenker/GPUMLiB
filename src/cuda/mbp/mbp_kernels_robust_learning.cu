/*
	Noel Lopes is a Professor at the Polytechnic of Guarda, Portugal
	and a researcher at the CISUC - University of Coimbra, Portugal
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
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
	*/

#include "mbp_kernels.h"

namespace GPUMLib {

	__global__ void cuda_mbp_robust_learning(cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, int layers, int * numberWeights, cudafloat ** weights, cudafloat ** bestWeights, cudafloat ** learningRate, cudafloat r, cudafloat ** lastDeltaWithoutLearningMomentum, cudafloat ** lastDelta) {
		__shared__ cudafloat rms;
		__shared__ cudafloat bRMS;

		rms = *rmsF;
		bRMS = *bestRMS;

		if (rms < bRMS) {
			for (int l = 0; l < layers; l++) {
				if (threadIdx.x < numberWeights[l]) bestWeights[l][threadIdx.x] = weights[l][threadIdx.x];
			}

			if (threadIdx.x == 0) *bestRMS = rms;
		} else if (rms >= bRMS * maxErrorGrowth) {
			for (int l = 0; l < layers; l++) {
				if (threadIdx.x < numberWeights[l]) {
					weights[l][threadIdx.x] = bestWeights[l][threadIdx.x];

					learningRate[l][threadIdx.x] *= r;

					lastDeltaWithoutLearningMomentum[l][threadIdx.x] = CUDA_VALUE(0.0);
					lastDelta[l][threadIdx.x] = CUDA_VALUE(0.0);
				}
			}
		}
	}

}

extern "C" cudaError_t gpumlib_cuda_mbp_robust_learning(dim3 blockDim, cudaStream_t stream, cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, int layers, int * numberWeights, cudafloat ** weights, cudafloat ** bestWeights, cudafloat ** learningRate, cudafloat r, cudafloat ** lastDeltaWithoutLearningMomentum, cudafloat ** lastDelta) {
	GPUMLib::cuda_mbp_robust_learning<<<1, blockDim, 0, stream>>>(rmsF, bestRMS, maxErrorGrowth, layers, numberWeights, weights, bestWeights, learningRate, r, lastDeltaWithoutLearningMomentum, lastDelta);

	return cudaGetLastError();
}