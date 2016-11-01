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

	__global__ void cuda_mbp_fire_selective_inputs(cudafloat * inputs, cudafloat * weights, cudafloat * bias, cudafloat * outputs, int numNeurons) {
		for (int n = threadIdx.x; n < numNeurons; n += blockDim.x) {
			int idx = SAMPLE * numNeurons + n;

			cudafloat o = inputs[idx];

			if (IsInfOrNaN(o)) {
				o = CUDA_VALUE(0.0);
			} else {
				cudafloat w = weights[n];
				cudafloat b = bias[n];

				if (w != CUDA_VALUE(0.0) || b != CUDA_VALUE(0.0)) { // input may have missing values
					o = CUDA_TANH(o * w + b);
				}
			}

			outputs[idx] = o;
		}
	}

}

extern "C" cudaError_t gpumlib_cuda_mbp_fire_selective_inputs(unsigned int gridDim, dim3 blockDim, cudaStream_t stream, cudafloat * inputs, cudafloat * weights, cudafloat * bias, cudafloat * outputs, int numNeurons) {
	GPUMLib::cuda_mbp_fire_selective_inputs<<<gridDim, blockDim, 0, stream>>>(inputs, weights, bias, outputs, numNeurons);

	return cudaGetLastError();
}
