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

	__global__ void cuda_mbp_calculate_local_gradients(cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * outputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * localGradientNextLayer, int neuronsNextLayer, int neurons, cudafloat * localGradient, cudafloat * localGradientSpaceNet) {
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
				int n = SAMPLE * neurons + neuron;

				cudafloat Fh = outputs[n];

				cudafloat lgn = lg[threadId];

				if (m != nullptr) {
					int nSelAct = SAMPLE * totalNeuronsWithSelectiveActivation + neuron + mOffset;

					cudafloat M = m[nSelAct];
					if (M == CUDA_VALUE(0.0)) {
						localGradientSpaceNet[nSelAct] = CUDA_VALUE(0.0);
					} else {
						Fh = Fh / M;
						localGradientSpaceNet[nSelAct] = lgn * Fh * CUDA_SIGMOID_DERIVATE(M);
					}
					lgn *= M;
				}

				localGradient[n] = lgn * CUDA_SIGMOID_DERIVATE(Fh);
			}
		}
	}

}

extern "C" cudaError_t gpumlib_cuda_mbp_calculate_local_gradients(unsigned int gridDim, dim3 blockDim, unsigned int sharedMemSize, cudaStream_t stream, cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * outputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * localGradientNextLayer, int neuronsNextLayer, int neurons, cudafloat * localGradient, cudafloat * localGradientSpaceNet) {
	GPUMLib::cuda_mbp_calculate_local_gradients<<<gridDim, blockDim, sharedMemSize, stream>>>(rmsF, bestRMS, maxErrorGrowth, outputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, localGradientNextLayer, neuronsNextLayer, neurons, localGradient, localGradientSpaceNet);

	return cudaGetLastError();
}
