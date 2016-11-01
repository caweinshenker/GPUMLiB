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

#include "../reduction/sum_warp.h"
#include "mbp_kernels.h"

#define BIAS 0

#define INPUT blockIdx.x
#define NUM_INPUTS_INCLUDING_BIAS gridDim.x
#define NUM_INPUTS (NUM_INPUTS_INCLUDING_BIAS - 1)

#define NEURON blockIdx.y
#define NUM_NEURONS gridDim.y

namespace GPUMLib {

	template <int blockSize> __global__ void CorrectLayerWeights(cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * inputs, cudafloat * localGradient, cudafloat * weights, cudafloat * learningRate, cudafloat * lastDeltaWithoutLearningMomentum, cudafloat * lastDelta, cudafloat maxStepSize, cudafloat u, cudafloat d, cudafloat r, cudafloat momentum, int numberPatterns) {
		extern __shared__ cudafloat deltas[];

		if (bestRMS != nullptr) {
			__shared__ cudafloat rms;
			__shared__ cudafloat bRMS;

			if (threadIdx.x == 0 && threadIdx.y == 0) {
				rms = *rmsF;
				bRMS = *bestRMS;
			}
			__syncthreads();

			if (rms >= bRMS * maxErrorGrowth) return;
		}

		deltas[threadIdx.x] = CUDA_VALUE(0.0);
		for (int p = threadIdx.x; p < numberPatterns; p += blockDim.x) {
			cudafloat delta = localGradient[p * NUM_NEURONS + NEURON];
			if (INPUT > BIAS) delta *= inputs[p * NUM_INPUTS + (INPUT - 1)];

			deltas[threadIdx.x] += delta;
		}
		__syncthreads();

		SumBeforeWarp<blockSize>(deltas);

		if (threadIdx.x < 32) {
			SumWarp<blockSize>(deltas);

			if (threadIdx.x == 0) {
				int connection = NEURON * NUM_INPUTS_INCLUDING_BIAS + INPUT;

				cudafloat delta = deltas[0] / numberPatterns;
				cudafloat learnRate = learningRate[connection];

				cudafloat factor = SAME_DIRECTION(lastDeltaWithoutLearningMomentum[connection], delta) ? u : d;
				learnRate *= factor;
				if (learnRate > maxStepSize) learnRate = maxStepSize;
				learningRate[connection] = learnRate;

				lastDeltaWithoutLearningMomentum[connection] = delta;

				delta += momentum * lastDelta[connection];
				lastDelta[connection] = delta;

				cudafloat w = weights[connection] + (learnRate * delta);
				if (IsInfOrNaN(w)) {
					lastDelta[connection] = CUDA_VALUE(0.0);
					lastDeltaWithoutLearningMomentum[connection] = CUDA_VALUE(0.0);
					if (bestRMS != nullptr) {
						learnRate *= r;
						learningRate[connection] = learnRate;
					}
				} else {
					weights[connection] = w;
				}
			}
		}
	}

	void KernelCorrectLayerWeights(cudaStream_t stream, dim3 & gridSize, int blockSize, cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * inputs, cudafloat * localGradient, cudafloat * weights, cudafloat * learningRate, cudafloat * lastDeltaWithoutLearningMomentum, cudafloat * lastDelta, cudafloat maxStepSize, cudafloat u, cudafloat d, cudafloat r, cudafloat momentum, int numberPatterns) {
		switch (blockSize) {
			case 1024:
				CorrectLayerWeights<1024><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 512:
				CorrectLayerWeights<512><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 256:
				CorrectLayerWeights<256><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 128:
				CorrectLayerWeights<128><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 64:
				CorrectLayerWeights<64><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 32:
				CorrectLayerWeights<32><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 16:
				CorrectLayerWeights<16><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 8:
				CorrectLayerWeights<8><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 4:
				CorrectLayerWeights<4><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 2:
				CorrectLayerWeights<2><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
			case 1:
				CorrectLayerWeights<1><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, r, momentum, numberPatterns);
				break;
		}
	}

}
