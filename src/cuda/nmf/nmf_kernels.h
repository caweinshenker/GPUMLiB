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

#ifndef GPUMLIB_NMF_KERNELS_H
#define GPUMLIB_NMF_KERNELS_H

#include <cuda_runtime.h>
#include "../definitions.h"

namespace GPUMLib {

	//! \addtogroup nmfkernels Non-negative Matrix Factorization kernels
	//! @{

	//! Small value added to the denominator of a fraction to prevent division by zero.
#define SMALL_VALUE_TO_ADD_DENOMINATOR (CUDA_VALUE(10e-9))


	//! Kernel used to determine the quality improvement (regarding the approximation obtained) caused by the previous iteration
	//! \param[in] blockSize Block size
	//! \param[in] V matrix V
	//! \param[in] WH WH (approximation of the V matix) matrix
	//! \param[in] n number of elements of the matrix V (and WH)
	//! \param[out] quality Quality improvement
	//! \sa QualityImprovement
	void KernelNMFquality(int blockSize, cudafloat * V, cudafloat * WH, int n, cudafloat * quality);


	// NMFadditiveEuclidian kernels




	// NMF_MultiplicativeDivergence kernels

#define BLOCK_SIZE_NMF (16)
#define BLOCK_MULTIPLIER_NMF (2)

	//! Calculates the sums of all rows for each column of W and places the results in a array. Used by the NMF_MultiplicativeDivergence class.
	//! \param[in] blockSize Number of threads per block. Must be a multiple of 2 and can not exceed the maximum number of threads per block.
	//! \param[in] W Matrix W.
	//! \param[in] n number of W rows.
	//! \param[in] r number of W columns.
	//! \param[out] sumW will contain the sums of the rows for each column.
	//! \sa NMF_MultiplicativeDivergence
	void KernelSumW(int blockSize, cudafloat * W, int n, int r, cudafloat * sumW);

	//! Calculates the sums of all columns for each row of H and places the results in a array. Used by the NMF_MultiplicativeDivergence class.
	//! \param[in] blockSize Number of threads per block. Must be a multiple of 2 and can not exceed the maximum number of threads per block.
	//! \param[in] H Matrix H.
	//! \param[in] r number of H rows.
	//! \param[in] m number of H columns.
	//! \param[out] sumH will contain the sums of the columns for each row.
	//! \sa NMF_MultiplicativeDivergence
	void KernelSumH(int blockSize, cudafloat * H, int r, int m, cudafloat * sumH);


	//! @}

}

// TODO: UPDATE DOCUMENTATION
// Kernel used by the NMF_MultiplicativeEuclidianDistance class to update matrices W and H
// \param[in] nm numerator matrix
// \param[in] dm denominator matrix
// \param[in, out] m matrix being updated
// \param[in] elements number of elements (rows * columns) of the matrices
// \sa NMF_MultiplicativeEuclidianDistance
extern "C" cudaError_t gpumlib_cuda_nmf_multiplicative_euclidean_update_matrix(unsigned int gridDim, unsigned int blockDim, cudafloat * nm, cudafloat * dm, cudafloat * m, int elements);

// TODO: UPDATE DOCUMENTATION
// Updates the matrix W. Used by the NMF_MultiplicativeDivergence class.
// \param[in, out] W Matrix W.
// \param[in] H Matrix H.
// \param[in] V Matrix V.
// \param[in] WH WH matrix approximation (of V).
// \param[in] sumH sums of the columns for each row.
// \param[in] n number of rows of V.
// \param[in] m number of columns of V.
// \param[in] r number of columns of W (and rows of H).
// \sa KernelSumH, NMF_MultiplicativeDivergence
extern "C" cudaError_t gpumlib_cuda_nmf_multiplicative_divergence_update_matrix_w(dim3 gridDim, dim3 blockDim, cudafloat * W, cudafloat * H, cudafloat * V, cudafloat * WH, cudafloat * sumH, int n, int m, int r);

// TODO: UPDATE DOCUMENTATION
//! Updates the matrix H. Used by the NMF_MultiplicativeDivergence class.
//! \param[in, out] H Matrix H.
//! \param[in] W Matrix W.
//! \param[in] V Matrix V.
//! \param[in] WH WH matrix approximation (of V).
//! \param[in] sumW sums of the rows for each column.
//! \param[in] n number of rows of V.
//! \param[in] m number of columns of V.
//! \param[in] r number of rows of H (and columns of W).
//! \sa KernelSumW, NMF_MultiplicativeDivergence
extern "C" cudaError_t gpumlib_cuda_nmf_multiplicative_divergence_update_matrix_h(dim3 gridDim, dim3 blockDim, cudafloat * H, cudafloat * W, cudafloat * V, cudafloat * WH, cudafloat * sumW, int n, int m, int r);

// TODO: UPDATE DOCUMENTATION
extern "C" cudaError_t gpumlib_cuda_nmf_additive_euclidean_update_matrix(unsigned int gridDim, unsigned int blockDim, cudafloat * X, cudafloat * deltaX1, cudafloat * deltaX2, int elements);

// TODO: UPDATE DOCUMENTATION
// Updates the matrix H. Used by the NMF_AdditiveDivergence class.
// \param[in, out] H Matrix H.
// \param[in] W Matrix W.
// \param[in] V Matrix V.
// \param[in] WH WH matrix approximation (of V).
// \param[in] sumW sums of the rows for each column.
// \param[in] n number of rows of V.
// \param[in] m number of columns of V.
// \param[in] r number of rows of H (and columns of W).
// \sa KernelSumW, NMF_AdditiveDivergence
extern "C" cudaError_t gpumlib_cuda_nmf_additive_divergence_update_matrix_h(dim3 gridDim, dim3 blockDim, cudafloat * H, cudafloat * W, cudafloat * V, cudafloat * WH, cudafloat * sumW, int n, int m, int r);

// TODO: UPDATE DOCUMENTATION
//! Updates the matrix W. Used by the NMF_AdditiveDivergence class.
//! \param[in, out] W Matrix W.
//! \param[in] H Matrix H.
//! \param[in] V Matrix V.
//! \param[in] WH WH matrix approximation (of V).
//! \param[in] sumH sums of the columns for each row.
//! \param[in] n number of rows of V.
//! \param[in] m number of columns of V.
//! \param[in] r number of columns of W (and rows of H).
//! \sa KernelSumH, NMF_AdditiveDivergence
extern "C" cudaError_t gpumlib_cuda_nmf_additive_divergence_update_matrix_w(dim3 gridDim, dim3 blockDim, cudafloat * W, cudafloat * H, cudafloat * V, cudafloat * WH, cudafloat * sumH, int n, int m, int r);

#endif // GPUMLIB_NMF_KERNELS_H
