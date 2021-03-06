/*
	 Noel Lopes is a Professor at the Polytechnic of Guarda, Portugal
	 and a Researcher at the CISUC - University of Coimbra, Portugal
	 Copyright (C) 2009-2015 Noel de Jesus Mendonça Lopes

	 Siti Mariyam Shamsuddin is a Professor at the Faculty of Computing
	 Universiti Teknologi Malaysia (UTM), Malaysia and researcher at
	 UTM Big Data Centre, Malaysia

	 Shafaatunnur Hasan is a full time researcher at UTM Big Data Centre,
	 Malaysia

	 Copyright (C) 2009-2015 Noel de Jesus Mendonça Lopes
	                         Siti Mariyam Shamsuddin
	                         Shafaatunnur Hasan

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

#include "som_kernels.h"

__global__ void ComputeDistancesSOMkernel(cudafloat * inputData, cudafloat * weights, int vector, int features, cudafloat * distances) {
	extern __shared__ cudafloat sdist [];

	int w = (blockIdx.y * gridDim.x + blockIdx.x);

	cudafloat distance = 0.0;

	for (int f = threadIdx.x; f < features; f += blockDim.x) {
		cudafloat fdist = inputData[vector * features + f] - weights[w * features + f];
		distance += fdist * fdist;
	}

	sdist[threadIdx.x] = distance;

	__syncthreads();

	// reduction
	for (int dist = blockDim.x; dist >= 2;) {
		dist /= 2;
		if (threadIdx.x < dist) {
			sdist[threadIdx.x] += sdist[threadIdx.x + dist];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		distances[w] = sqrt(sdist[0]);
	}
}

cudaError_t ComputeDistancesSOM(dim3 gridSize, int blockSize, cudafloat * inputData, cudafloat * weights, int vector, int numberFeatures, cudafloat * distances) {
	ComputeDistancesSOMkernel<<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(inputData, weights, vector, numberFeatures, distances);

	return cudaGetLastError();
}

__global__ void UpdateWeightsSOMkernel(int * bmu, int * mapView, int mapx, int mapy, cudafloat * inputData, int vector, int features, int target, cudafloat neighbourhoodRadiusSquare, cudafloat * weights, cudafloat learningRate) {
	__shared__ int winx;
	__shared__ int winy;

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		winx = *bmu % mapx;
		winy = *bmu / mapx;
		mapView[*bmu] = target;
	}
	__syncthreads();

	for (int y = threadIdx.z; y < mapy; y += blockDim.z) {
		cudafloat dy = winy - y;

		for (int x = threadIdx.y; x < mapx; x += blockDim.y) {
			cudafloat dx = winx - x;

			cudafloat distance = dx * dx + dy * dy;

			cudafloat influence = exp(-distance / (2 * neighbourhoodRadiusSquare));

			if (distance < neighbourhoodRadiusSquare) {
				for (int f = threadIdx.x; f < features; f += blockDim.x) {
					int idx = (y * mapx + x) * features + f;

					weights[idx] += learningRate * influence * (inputData[vector * features + f] - weights[idx]);
				}
			}
		}
	}
}

cudaError_t UpdateWeightsSOM(dim3 blockSize, int * bmu, int * mapView, int mapx, int mapy, cudafloat * inputData, int vector, int features, int target, cudafloat neighbourhoodRadiusSquared, cudafloat * weights, cudafloat learningRate) {
	UpdateWeightsSOMkernel<<<1, blockSize>>>(bmu, mapView, mapx, mapy, inputData, vector, features, target, neighbourhoodRadiusSquared, weights, learningRate);

	return cudaGetLastError();
}

__global__ void NormalizeWeightsSOMkernel(int mapx, int mapy, int features, cudafloat * weights) {
	extern __shared__ cudafloat snorm[];

	int idx = (blockIdx.y * gridDim.x + blockIdx.x) * features;

	cudafloat norm = 0.0;

	for (int f = threadIdx.x; f < features; f += blockDim.x) {
		cudafloat weight = weights[idx + f];
		norm += weight * weight;
	}

	snorm[threadIdx.x] = norm;

	__syncthreads();

	// reduction
	for (int dist = blockDim.x; dist >= 2;) {
		dist /= 2;
		if (threadIdx.x < dist) {
			snorm[threadIdx.x] += snorm[threadIdx.x + dist];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		*snorm = CUDA_RSQRT(*snorm);
	}

	__syncthreads();

	for (int f = threadIdx.x; f < features; f += blockDim.x) {
		weights[idx + f] *= *snorm;
	}
}

cudaError_t NormalizeWeightsSOM(dim3 gridSize, int blockSize, int mapx, int mapy, int features, cudafloat * weights) {
	NormalizeWeightsSOMkernel<<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(mapx, mapy, features, weights);

	return cudaGetLastError();
}
