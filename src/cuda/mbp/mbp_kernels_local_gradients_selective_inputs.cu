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

#define SAMPLE blockIdx.x

namespace GPUMLib {

	__global__ void cuda_mbp_calculate_local_gradients_selective_inputs(cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * inputs, cudafloat * selectiveNeuronsWeights, cudafloat * selectiveNeuronsBias, cudafloat * weights, cudafloat * localGradientNextLayer, int neuronsNextLayer, int neurons, cudafloat * localGradient) {
		extern __shared__ cudafloat lg[];

		// if robust learning, check if the RMS has increased too much relatively to the best RMS found so far and if so abort the execution of the kernel
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

		cudafloat * lgNextLayer = (lg + (blockDim.y * blockDim.x));

		int threadId = (threadIdx.y * blockDim.x + threadIdx.x);

		for (int neuron = threadIdx.y; neuron < neurons + threadIdx.y; neuron += blockDim.y) {
			lg[threadId] = 0;

			for (int outputNeuron = threadIdx.x; outputNeuron < neuronsNextLayer + threadIdx.x; outputNeuron += blockDim.x) {
				if (threadIdx.y == 0 && outputNeuron < neuronsNextLayer) {
					lgNextLayer[threadIdx.x] = localGradientNextLayer[SAMPLE * neuronsNextLayer + outputNeuron];
				}
				__syncthreads();

				if (outputNeuron < neuronsNextLayer && neuron < neurons) {
					int connection = outputNeuron * (neurons + 1) + neuron + 1;
					lg[threadId] += weights[connection] * lgNextLayer[threadIdx.x];
				}
				__syncthreads();
			}

			int numberElemSum = blockDim.x;
			for (int sumUpTo = (numberElemSum >> 1); numberElemSum > 1; sumUpTo = (numberElemSum >> 1)) {
				int nextNumberElemSum = sumUpTo;
				if (numberElemSum & 1) nextNumberElemSum++;

				if (threadIdx.x < sumUpTo) lg[threadId] += lg[threadId + nextNumberElemSum];

				numberElemSum = nextNumberElemSum;

				__syncthreads();
			}

			if (threadIdx.x == 0 && neuron < neurons) {
				cudafloat lgn = 0;

				int n = SAMPLE * neurons + neuron;

				cudafloat i = inputs[n];

				if (!IsInfOrNaN(i)) {
					cudafloat w = selectiveNeuronsWeights[neuron];
					cudafloat b = selectiveNeuronsBias[neuron];

					if (w != 0 || b != 0) { // input may have missing values
						cudafloat coshfx = CUDA_COSH(i * w + b);
						lgn = lg[threadId] / (coshfx * coshfx); // derivate = 1 / (coshfx * coshfx)
					}
				}

				localGradient[n] = lgn;
			}
		}
	}

}

extern "C" cudaError_t gpumlib_cuda_mbp_calculate_local_gradients_selective_inputs(unsigned int gridDim, dim3 blockDim, unsigned int sharedMemSize, cudaStream_t stream, cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * inputs, cudafloat * selectiveNeuronsWeights, cudafloat * selectiveNeuronsBias, cudafloat * weights, cudafloat * localGradientNextLayer, int neuronsNextLayer, int neurons, cudafloat * localGradient) {
	GPUMLib::cuda_mbp_calculate_local_gradients_selective_inputs<<<gridDim, blockDim, sharedMemSize, stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, selectiveNeuronsWeights, selectiveNeuronsBias, weights, localGradientNextLayer, neuronsNextLayer, neurons, localGradient);

	return cudaGetLastError();
}

