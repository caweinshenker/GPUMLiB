/*
	João Goncalves is a MSc Student at the University of Coimbra, Portugal
	Copyright (C) 2012 Joao Goncalves

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

#ifndef GPUMLIB_SVM_KERNELS_H
#define GPUMLIB_SVM_KERNELS_H

//GPUMLib stuff
#include <cuda.h>

#include "svm_kernel_type.h"
#include "../definitions.h"
#include "../../memory/DeviceArray.h"
#include "../../memory/DeviceMatrix.h"

namespace GPUMLib {
	void cudaCalculateB_1stPass(cudaStream_t stream,
								int blocks, int blockSize,
								cudafloat * offsets,
								cudafloat * results,
								int n_svs);

	void cudaCalculateB_FinalPass(cudaStream_t stream, int blockSize, cudafloat * input_floats, int input_size);
	
	void cudaFirstOrderHeuristic1stPass(cudaStream_t stream, int blocks, int blockSize,
										cudafloat * f, cudafloat * alphas,
										int * y, cudafloat * minimuns, int * min_indices,
										cudafloat * maximuns, int * max_indices, int input_size,
										cudafloat constant_epsilon, cudafloat constant_c);

	void cudaFirstOrderHeuristicFinalPass(cudaStream_t stream, int blockSize,
										  cudafloat * minimuns_input, int * min_indices_input,
										  cudafloat * maximuns_input, int * max_indices_input,
										  int input_size);

	void cudaUpdateAlphasAdvanced(cudaStream_t stream, GPUMLib::svm_kernel_type kernel_type,
								  GPUMLib::DeviceMatrix<cudafloat> &d_x, GPUMLib::DeviceArray<cudafloat> &d_alphas,
								  GPUMLib::DeviceArray<int> &d_y, cudafloat constant_c_negative, cudafloat constant_c_positive,
								  GPUMLib::DeviceArray<cudafloat> &d_kernel_args, int training_dataset_size, int ndims);

	void cudaUpdateKKTConditions(cudaStream_t stream, GPUMLib::svm_kernel_type kernel_type, int n_blocks, int blocksize,
								 GPUMLib::DeviceArray<cudafloat> &d_f, GPUMLib::DeviceArray<int> &d_y,
								 GPUMLib::DeviceMatrix<cudafloat> &d_x, GPUMLib::DeviceArray<cudafloat> &d_kernel_args,
								 int training_dataset_size, int ndims);

	void cudaInitializeSMO(int n_blocks, int n_threads_per_block, cudafloat * alphas, cudafloat * f, int * classes, int _nsamples);

	void CopyLowHighIndicesToDeviceMemory(int & i_low, int & i_high, cudafloat & h_b_low, cudafloat & h_b_high);
	void CopyLowHighOffsetsFromDeviceMemory(cudaStream_t & stream_memory_transaction, int & i_low, int & i_high, cudafloat & h_b_low, cudafloat & h_b_high);

	void cudaCalculateOffsetsUsingModel(svm_kernel_type kernel_type, int n_blocks, int n_threads_per_block, cudaStream_t stream_bias_calculus, cudafloat * results, cudafloat * model, int nsvs, int ndims, cudafloat * kernel_args);
	void CopyBiasToHost(cudafloat & h_b);
	void cudaClassifyDataSet(svm_kernel_type kernel_type, int n_blocks, int n_threads_per_block, int * results, cudafloat * dataset, int dataset_size, cudafloat * model, int nsvs, cudafloat b, int ndims, cudafloat * kernel_args);
}

#endif // GPUMLIB_SVM_KERNELS_H
