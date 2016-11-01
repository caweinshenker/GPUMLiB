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
#include "../reduction/sum_warp.h"

#define NEURON blockIdx.y
#define NUM_NEURONS gridDim.y

#define SAMPLE blockIdx.x

namespace GPUMLib {
	template <int blockSize> __global__ void cuda_rbm_compute_status_hidden_units(cudafloat * v, cudafloat * weights, cudafloat * b, cudafloat * h, float * randomValues, int I) {
		extern __shared__ cudafloat iw[];

		iw[threadIdx.x] = 0;
		for (int i = threadIdx.x; i < I; i += blockDim.x) {
			iw[threadIdx.x] += v[SAMPLE * I + i] * weights[NEURON * I + i];
		}
		__syncthreads();

		if (blockSize >= 1024) {
			if (threadIdx.x < 512) iw[threadIdx.x] += iw[threadIdx.x + 512];
			__syncthreads();
		}

		if (blockSize >= 512) {
			if (threadIdx.x < 256) iw[threadIdx.x] += iw[threadIdx.x + 256];
			__syncthreads();
		}

		if (blockSize >= 256) {
			if (threadIdx.x < 128) iw[threadIdx.x] += iw[threadIdx.x + 128];
			__syncthreads();
		}

		if (blockSize >= 128) {
			if (threadIdx.x < 64) iw[threadIdx.x] += iw[threadIdx.x + 64];
			__syncthreads();
		}

		__shared__ cudafloat output;
		if (threadIdx.x < 32) {
			SumWarp<blockSize>(iw);

			if (threadIdx.x == 0) {
				output = CUDA_SIGMOID(iw[0] + b[NEURON]);
				int idx = SAMPLE * NUM_NEURONS + NEURON;
				if (randomValues != nullptr) output = (output > randomValues[idx]) ? 1 : 0;
				h[idx] = output;
			}
		}
	}

	template <int blockSize> __global__ void cuda_rbm_compute_status_visible_units(cudafloat * h, cudafloat * weights, cudafloat * a, cudafloat * v, float * randomValues, int J) {
		extern __shared__ cudafloat sum[];

		sum[threadIdx.x] = 0;
		for (int j = threadIdx.x; j < J; j += blockDim.x) {
			sum[threadIdx.x] += h[SAMPLE * J + j] * weights[j * NUM_NEURONS + NEURON];
		}
		__syncthreads();

		if (blockSize >= 1024) {
			if (threadIdx.x < 512) sum[threadIdx.x] += sum[threadIdx.x + 512];
			__syncthreads();
		}

		if (blockSize >= 512) {
			if (threadIdx.x < 256) sum[threadIdx.x] += sum[threadIdx.x + 256];
			__syncthreads();
		}

		if (blockSize >= 256) {
			if (threadIdx.x < 128) sum[threadIdx.x] += sum[threadIdx.x + 128];
			__syncthreads();
		}

		if (blockSize >= 128) {
			if (threadIdx.x < 64) sum[threadIdx.x] += sum[threadIdx.x + 64];
			__syncthreads();
		}

		if (threadIdx.x < 32) {
			SumWarp<blockSize>(sum);

			if (threadIdx.x == 0) {
				cudafloat output = CUDA_SIGMOID(sum[0] + a[NEURON]);

				int idx = SAMPLE * NUM_NEURONS + NEURON;
				if (randomValues != nullptr) output = (output > randomValues[idx]) ? 1 : 0;
				v[idx] = output;
			}
		}
	}
}

extern "C" cudaError_t gpumlib_cuda_rbm_compute_status_hidden_units(dim3 gridDim, unsigned int blockDim, cudafloat * v, cudafloat * weights, cudafloat * b, cudafloat * h, float * randomValues, int I) {
	const unsigned int sharedMemSize = blockDim * sizeof(cudafloat);

	switch (blockDim) {
	case 1024:
		GPUMLib::cuda_rbm_compute_status_hidden_units<1024><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 512:
		GPUMLib::cuda_rbm_compute_status_hidden_units<512><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 256:
		GPUMLib::cuda_rbm_compute_status_hidden_units<256><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 128:
		GPUMLib::cuda_rbm_compute_status_hidden_units<128><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 64:
		GPUMLib::cuda_rbm_compute_status_hidden_units<64><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 32:
		GPUMLib::cuda_rbm_compute_status_hidden_units<32><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 16:
		GPUMLib::cuda_rbm_compute_status_hidden_units<16><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 8:
		GPUMLib::cuda_rbm_compute_status_hidden_units<8><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 4:
		GPUMLib::cuda_rbm_compute_status_hidden_units<4><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 2:
		GPUMLib::cuda_rbm_compute_status_hidden_units<2><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	case 1:
		GPUMLib::cuda_rbm_compute_status_hidden_units<1><<<gridDim, blockDim, sharedMemSize>>>(v, weights, b, h, randomValues, I);
		break;
	}

	return cudaGetLastError();
}

extern "C" cudaError_t gpumlib_cuda_rbm_compute_status_visible_units(dim3 gridDim, unsigned int blockDim, cudafloat * h, cudafloat * weights, cudafloat * a, cudafloat * v, float * randomValues, int J) {
	const unsigned int sharedMemSize = blockDim * sizeof(cudafloat);

	switch (blockDim) {
	case 1024:
		GPUMLib::cuda_rbm_compute_status_visible_units<1024><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 512:
		GPUMLib::cuda_rbm_compute_status_visible_units<512><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 256:
		GPUMLib::cuda_rbm_compute_status_visible_units<256><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 128:
		GPUMLib::cuda_rbm_compute_status_visible_units<128><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 64:
		GPUMLib::cuda_rbm_compute_status_visible_units<64><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 32:
		GPUMLib::cuda_rbm_compute_status_visible_units<32><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 16:
		GPUMLib::cuda_rbm_compute_status_visible_units<16><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 8:
		GPUMLib::cuda_rbm_compute_status_visible_units<8><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 4:
		GPUMLib::cuda_rbm_compute_status_visible_units<4><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 2:
		GPUMLib::cuda_rbm_compute_status_visible_units<2><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	case 1:
		GPUMLib::cuda_rbm_compute_status_visible_units<1><<<gridDim, blockDim, sharedMemSize>>>(h, weights, a, v, randomValues, J);
		break;
	}

	return cudaGetLastError();
}