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

#ifndef GPUMLIB_REDUCTION_H
#define GPUMLIB_REDUCTION_H

#include <cuda_runtime.h>
#include <cmath>

#include "reduction_kernels.h"
#include "../definitions.h"
#include "../../common/Utilities.h"
#include "../../memory/CudaArray.h"
#include "../../memory/DeviceMatrix.h"
#include "../../memory/DeviceAccessibleVariable.h"

namespace GPUMLib {

	//! \addtogroup reduction Reduction framework
	//! @{

	//! Provides reduction functions (Sum, Average, Max, Min, ...).
	class Reduction {
	private:
		void static Sum(cudafloat * inputs, cudafloat * output, int numInputs, cudafloat multiplyFactor, cudaStream_t stream);

		void static MinIndex(cudafloat * inputs, cudafloat * output, int * minIndex, int numInputs, cudaStream_t stream);
		void static Min(cudafloat * inputs, cudafloat * output, int numInputs, cudaStream_t stream);

		void static Max(cudafloat * inputs, cudafloat * output, int numInputs, cudaStream_t stream);
		void static MaxIndex(cudafloat * inputs, cudafloat * output, int * minIndex, int numInputs, cudaStream_t stream);

	public:
		//! Temporary buffer used for the reduction tasks. Programmers may take advantage of it for other tasks (hence, it is declared as public).
		static DeviceArray<cudafloat> temporaryBuffer;

		//! Sums all the elements of an input array, multiplies the sum by a given factor and places the result in the output
		//! \param[in] inputs Values to be summed
		//! \param[out] output Pointer to the memory address that will contain the sum output
		//! \param[in] multiplyFactor Multiply factor (optional, by default 1.0)
		//! \param[in] stream CUDA stream (optional)
		void static Sum(DeviceArray<cudafloat> & inputs, cudafloat * output, cudafloat multiplyFactor = CUDA_VALUE(1.0), cudaStream_t stream = nullptr) {
			Sum(inputs.Pointer(), output, (int)inputs.Length(), multiplyFactor, stream);
		}

		//! Sums all the elements of an input array, multiplies the sum by a given factor and places the result in the output
		//! \param[in] inputs Values to be summed
		//! \param[out] output Array that will contain the sum output (in position 0)
		//! \param[in] multiplyFactor Multiply factor (optional, by default 1.0)
		//! \param[in] stream CUDA stream (optional)
		void static Sum(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudafloat multiplyFactor = CUDA_VALUE(1.0), cudaStream_t stream = nullptr) {
			Sum(inputs.Pointer(), output.Pointer(), (int)inputs.Length(), multiplyFactor, stream);
		}

		//! Averages the elements of an input array, placing the result in the output
		//! \param[in] inputs input array for which we want to compute the average
		//! \param[out] output Array that will contain the average (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Average(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			double multiplyFactor = 1.0 / inputs.Length();
			Sum(inputs.Pointer(), output.Pointer(), (int)inputs.Length(), (cudafloat)multiplyFactor, stream);
		}

		//! Computes the minimum of an input array, placing the result in the output
		//! \param[in] inputs input array for which we want to compute the minimum
		//! \param[out] output Array that will contain the minimum (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Min(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			Min(inputs.Pointer(), output.Pointer(), (int)inputs.Length(), stream);
		}

		//! Computes the minimum of an input matrix, placing the result in the output
		//! \param[in] inputs input matrix for which we want to compute the minimum
		//! \param[out] output Array that will contain the minimum (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Min(DeviceMatrix<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			Min(inputs.Pointer(), output.Pointer(), (int)inputs.Elements(), stream);
		}

		//! Computes the minimum of an input array as well as its index within the array
		//! \param[in] inputs input array for which we want to compute the minimum
		//! \param[out] min Array that will contain the minimum (in position 0)
		//! \param[out] minIndex Array that will contain the index of the minimum within the array (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static MinIndex(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & min, DeviceArray<int> & minIndex, cudaStream_t stream = nullptr) {
			MinIndex(inputs.Pointer(), min.Pointer(), minIndex.Pointer(), (int)inputs.Length(), stream);
		}

		//! Computes the minimum of an input matrix as well as its (1-D) index within the matrix
		//! \param[in] inputs input matrix for which we want to compute the minimum
		//! \param[out] min Array that will contain the minimum (in position 0)
		//! \param[out] minIndex Array that will contain the index of the minimum within the array (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static MinIndex(DeviceMatrix<cudafloat> & inputs, DeviceArray<cudafloat> & min, DeviceArray<int> & minIndex, cudaStream_t stream = nullptr) {
			MinIndex(inputs.Pointer(), min.Pointer(), minIndex.Pointer(), (int)inputs.Elements(), stream);
		}

		//! Computes the position (1-D index) of the minimum of an input matri
		//! \param[in] inputs input matrix for which we want to compute the minimum
		//! \param[out] minIndex Array that will contain the index of the minimum within the array (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static MinIndex(DeviceMatrix<cudafloat> & inputs, DeviceArray<int> & minIndex, cudaStream_t stream = nullptr) {
			MinIndex(inputs.Pointer(), nullptr, minIndex.Pointer(), (int)inputs.Elements(), stream);
		}

		//! Computes the maximum of an input array, placing the result in the output
		//! \param[in] inputs input array for which we want to compute the maximum
		//! \param[out] output Array that will contain the maximum (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Max(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			Max(inputs.Pointer(), output.Pointer(), (int)inputs.Length(), stream);
		}

		//! Computes the maximum of an input matrix, placing the result in the output
		//! \param[in] inputs input matrix for which we want to compute the maximum
		//! \param[out] output Array that will contain the maximum (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Max(DeviceMatrix<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			Max(inputs.Pointer(), output.Pointer(), (int)inputs.Elements(), stream);
		}

		//! Computes the maximum of an input array as well as its index within the array
		//! \param[in] inputs input array for which we want to compute the minimum
		//! \param[out] max Array that will contain the minimum (in position 0)
		//! \param[out] maxIndex Array that will contain the index of the minimum within the array (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static MaxIndex(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & max, DeviceArray<int> & maxIndex, cudaStream_t stream = nullptr) {
			MaxIndex(inputs.Pointer(), max.Pointer(), maxIndex.Pointer(), (int)inputs.Length(), stream);
		}

		//! Computes the maximum of an input matrix as well as its (1-D) index within the array
		//! \param[in] inputs input matrix for which we want to compute the minimum
		//! \param[out] max Array that will contain the minimum (in position 0)
		//! \param[out] maxIndex Array that will contain the index of the minimum within the array (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static MaxIndex(DeviceMatrix<cudafloat> & inputs, DeviceArray<cudafloat> & max, DeviceArray<int> & maxIndex, cudaStream_t stream = nullptr) {
			MaxIndex(inputs.Pointer(), max.Pointer(), maxIndex.Pointer(), (int)inputs.Elements(), stream);
		}
	};

	//! @}

}

#endif // GPUMLIB_REDUCTION_H
