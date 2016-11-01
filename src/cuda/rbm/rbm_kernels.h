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

#ifndef GPUMLIB_RBM_KERNELS_H
#define GPUMLIB_RBM_KERNELS_H

#include <cuda_runtime.h>

#include "../definitions.h"

extern "C" cudaError_t gpumlib_cuda_rbm_init_bias_and_deltas(unsigned int gridDim, unsigned int blockDim, cudafloat * bias, cudafloat initialBias, cudafloat * lastDeltaW, cudafloat * lastDeltaB, cudafloat * lastDeltaWithoutLearningMomentumW, cudafloat * lastDeltaWithoutLearningMomentumB, cudafloat * learningRateW, cudafloat * learningRateB, cudafloat initialLearningRate, int weights, int J);
extern "C" cudaError_t gpumlib_cuda_rbm_init_input_bias_and_deltas(unsigned int gridDim, unsigned int blockDim, cudafloat * v, cudafloat * bias, cudafloat * lastDeltaA, cudafloat * lastDeltaWithoutLearningMomentumA, cudafloat * learningRateA, cudafloat initialLearningRate, int I, int samples);
extern "C" cudaError_t gpumlib_cuda_rbm_compute_status_hidden_units_small(unsigned int gridDim, dim3 blockDim, unsigned int sharedMemSize, cudafloat * v, cudafloat * weights, cudafloat * b, cudafloat * h, float * randomValues);
extern "C" cudaError_t gpumlib_cuda_rbm_compute_status_visible_units_small(unsigned int gridDim, dim3 blockDim, unsigned int sharedMemSize, cudafloat * h, cudafloat * weights, cudafloat * a, cudafloat * v, float * randomValues);
extern "C" cudaError_t gpumlib_cuda_rbm_compute_status_hidden_units(dim3 gridDim, unsigned int blockDim, cudafloat * v, cudafloat * weights, cudafloat * b, cudafloat * h, float * randomValues, int I);
extern "C" cudaError_t gpumlib_cuda_rbm_compute_status_visible_units(dim3 gridDim, unsigned int blockDim, cudafloat * h, cudafloat * weights, cudafloat * a, cudafloat * v, float * randomValues, int J);
extern "C" cudaError_t gpumlib_cuda_rbm_correct_weights(dim3 gridDim, dim3 blockDim, cudafloat * v_data, cudafloat * h_data, cudafloat * v_recon, cudafloat * h_recon, int samples, cudafloat * learningRateW, cudafloat * lastDeltaWithoutLearningMomentumW, cudafloat * lastDeltaW, cudafloat * learningRateB, cudafloat * lastDeltaWithoutLearningMomentumB, cudafloat * lastDeltaB, cudafloat * learningRateA, cudafloat * lastDeltaWithoutLearningMomentumA, cudafloat * lastDeltaA, cudafloat u, cudafloat d, cudafloat momentum, cudafloat * weights, cudafloat * b, cudafloat * a, cudafloat * errors, int I, int J);

#endif // GPUMLIB_RBM_KERNELS_H
