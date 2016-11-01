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

#ifndef GPUMLIB_REDUCTION_KERNELS_H
#define GPUMLIB_REDUCTION_KERNELS_H

#include "../definitions.h"

namespace GPUMLib {

	//! \addtogroup reduction Reduction framework
	//! @{

	//! Kernel to sum an array. For small arrays use KernelSumSmallArray instead.
	//! \param[in] stream CUDA stream
	//! \param[in] blocks Number of thread blocks 
	//! \param[in] blockSize Block size (number of threads per block)
	//! \param[in] inputs Values to be summed
	//! \param[out] outputs Array that will contain the partial sums of each block
	//! \param[in] numInputs Number of inputs
	//! \sa KernelSumSmallArray, SIZE_SMALL_CUDA_VECTOR
	void KernelSum(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * outputs, int numInputs);

	//! Kernel to sum a small array, multiply the result by a given factor and place the result in the output.
	//! \param[in] stream CUDA stream
	//! \param[in] blockSize Block size (number of threads per block)
	//! \param[in] inputs Values to be summed
	//! \param[out] output Pointer to the location that will contain the sum output
	//! \param[in] numInputs Number of inputs
	//! \param[in] multiplyFactor Multiply factor (optional, by default 1.0)
	//! \sa KernelSum, SIZE_SMALL_CUDA_VECTOR
	void KernelSumSmallArray(cudaStream_t stream, int blockSize, cudafloat * inputs, cudafloat * output, int numInputs, cudafloat multiplyFactor);

	//! Kernel to compute the minimum of an array. 
	//! \param[in] stream CUDA stream
	//! \param[in] blocks Number of thread blocks
	//! \param[in] blockSize Block size (number of threads per block)
	//! \param[in] inputs input array
	//! \param[out] output Pointer to the location that will contain the minimum
	//! \param[in] numInputs Number of inputs
	void KernelMin(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int numInputs);

	//! Kernel to compute the minimum of an array and its index within the array. 
	//! \param[in] stream CUDA stream
	//! \param[in] blocks Number of thread blocks
	//! \param[in] blockSize Block size (number of threads per block)
	//! \param[in] inputs input array
	//! \param[out] output Pointer to the location that will contain the minimum
	//! \param[out] minIndexes Pointer to the location that will contain the index of one of the minimums
	//! \param[in] numInputs Number of inputs
	//! \param[in] indexes Buffer used to tempory store the indexes. Must have the same size of the inputs array.
	void KernelMinIndexes(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int * minIndexes, int numInputs, int * indexes);

	//! Kernel to compute the maximum of an array. 
	//! \param[in] stream CUDA stream
	//! \param[in] blocks Number of thread blocks
	//! \param[in] blockSize Block size (number of threads per block)
	//! \param[in] inputs input array
	//! \param[out] output Pointer to the location that will contain the maximum
	//! \param[in] numInputs Number of inputs
	void KernelMax(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int numInputs);

	//! Kernel to compute the maximum of an array and its index within the array. 
	//! \param[in] stream CUDA stream
	//! \param[in] blocks Number of thread blocks
	//! \param[in] blockSize Block size (number of threads per block)
	//! \param[in] inputs input array
	//! \param[out] output Pointer to the location that will contain the maximum
	//! \param[out] maxIndexes Pointer to the location that will contain the index of one of the maximums
	//! \param[in] numInputs Number of inputs
	//! \param[in] indexes Buffer used to tempory store the indexes. Must have the same size of the inputs array.
	void KernelMaxIndexes(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int * maxIndexes, int numInputs, int * indexes);

}

#endif // GPUMLIB_REDUCTION_KERNELS_H
