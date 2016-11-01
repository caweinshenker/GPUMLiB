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

#ifndef GPUMLIB_DEVICE_MATRIX_H
#define GPUMLIB_DEVICE_MATRIX_H

#include <cassert>
#include <cublas.h>

#include "../cuda/definitions.h"
#include "HostMatrix.h"
#include "DeviceMemoryManager.h"

namespace GPUMLib {

	//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
	//! @{

	//! Create a matrix of any type, on the device, that automatically manages the memory used to hold its elements
	template <class Type> class DeviceMatrix : public BaseMatrix < Type > {
	private:
		DeviceMemoryManager<Type> deviceMem;

	public:
		//! Constructs an empty matrix
		//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
		DeviceMatrix(StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(deviceMem, storingOrder) {}

		//! Constructs a matrix with a given number of rows and columns
		//! \param rows the number of rows
		//! \param columns the number of columns
		//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
		DeviceMatrix(size_t rows, size_t columns, StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(deviceMem, storingOrder) {
			this->ResizeWithoutPreservingData(rows, columns);
		}

		//! Constructs a matrix identical to the other
		//! \param other another matrix
		DeviceMatrix(const DeviceMatrix<Type> & other) : BaseMatrix<Type>(deviceMem) {
			this->AssignDeviceMatrix(other);
		}

		//! Constructs a matrix identical to an host matrix
		//! \param other host matrix
		DeviceMatrix(const HostMatrix<Type> & other) : BaseMatrix<Type>(deviceMem) {
			this->AssignHostMatrix(other);
		}

		//! Transforms this matrix into an matrix identical to an host matrix
		//! \param other host matrix
		//! \return a reference to this matrix
		//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
		//! \sa IsRowMajor
		DeviceMatrix<Type> & operator = (const HostMatrix<Type> & other) {
			this->AssignHostMatrix(other);
			return *this;
		}

		//! Transforms this matrix into an matrix identical to the other
		//! \param other other matrix
		//! \return a reference to this matrix
		//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
		//! \sa IsRowMajor
		DeviceMatrix<Type> & operator = (const DeviceMatrix<Type> & other) {
			this->AssignDeviceMatrix(other);
			return *this;
		}

		//! Constructs a matrix using the elements of a device temporary matrix (rvalue)
		//! \param temporaryMatrix temporary device matrix containing the elements
		DeviceMatrix(DeviceMatrix<Type> && temporaryMatrix) : BaseMatrix<Type>(deviceMem) {
			this->TransferOwnerShipFrom(temporaryMatrix);
		}

		//! Replaces this matrix using a temporary matrix (rvalue)
		//! \param temporaryMatrix temporary matrix
		//! \return a reference to this matrix
		DeviceMatrix<Type> & operator = (DeviceMatrix<Type> && temporaryMatrix) {
			this->TransferOwnerShipFrom(temporaryMatrix);
			return *this;
		}

		//! Gets the transposed of the matrix
		//! \return the transposed of the matrix
		//! \attention The returned matrix does not use the same method (row-major or column-major) for storing information as this matrix.
		//! \sa ReplaceByTranspose, IsRowMajor
		DeviceMatrix<Type> Transpose() {
			HostMatrix<Type> transpose(*this);
			transpose.ReplaceByTranspose();

			return transpose;
		}
	};

	//! Create a cudafloat matrix, on the device, that automatically manages the memory used to hold its elements
	template <> class DeviceMatrix<cudafloat> : public BaseMatrix < cudafloat > {
	private:
		DeviceMemoryManager<cudafloat> deviceMem;

	public:
		//! Constructs an empty matrix
		//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
		DeviceMatrix(StoringOrder storingOrder = RowMajor) : BaseMatrix<cudafloat>(deviceMem, storingOrder) {}

		//! Constructs a matrix with a given number of rows and columns
		//! \param rows the number of rows
		//! \param columns the number of columns
		//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
		DeviceMatrix(size_t rows, size_t columns, StoringOrder storingOrder = RowMajor) : BaseMatrix<cudafloat>(deviceMem, storingOrder) {
			this->ResizeWithoutPreservingData(rows, columns);
		}

		//! Constructs a matrix identical to another
		//! \param other another matrix
		DeviceMatrix(const DeviceMatrix<cudafloat> & other) : BaseMatrix<cudafloat>(deviceMem) {
			this->AssignDeviceMatrix(other);
		}

		//! Constructs a matrix identical to an host matrix
		//! \param other host matrix
		DeviceMatrix(const HostMatrix<cudafloat> & other) : BaseMatrix<cudafloat>(deviceMem) {
			this->AssignHostMatrix(other);
		}

		//! Transforms this matrix into an matrix identical to an host matrix
		//! \param other host matrix
		//! \return a reference to this matrix
		//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
		//! \sa IsRowMajor
		DeviceMatrix<cudafloat> & operator = (const HostMatrix<cudafloat> & other) {
			this->AssignHostMatrix(other);
			return *this;
		}

		//! Transforms this matrix into an matrix identical to the other
		//! \param other other matrix
		//! \return a reference to this matrix
		//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
		//! \sa IsRowMajor
		DeviceMatrix<cudafloat> & operator = (const DeviceMatrix<cudafloat> & other) {
			this->AssignDeviceMatrix(other);
			return *this;
		}

		//! Constructs a matrix using the elements of a device temporary matrix (rvalue)
		//! \param temporaryMatrix temporary device matrix containing the elements
		DeviceMatrix(DeviceMatrix<cudafloat> && temporaryMatrix) : BaseMatrix<cudafloat>(deviceMem) {
			this->TransferOwnerShipFrom(temporaryMatrix);
		}

		//! Replaces this matrix using a temporary matrix (rvalue)
		//! \param temporaryMatrix temporary matrix
		//! \return a reference to this matrix
		DeviceMatrix<cudafloat> & operator = (DeviceMatrix<cudafloat> && temporaryMatrix) {
			this->TransferOwnerShipFrom(temporaryMatrix);
			return *this;
		}

		//! Gets the transposed of the matrix
		//! \return the transposed of the matrix
		//! \attention The returned matrix does not use the same method (row-major or column-major) for storing information as this matrix.
		//! \sa ReplaceByTranspose, IsRowMajor
		DeviceMatrix<cudafloat> Transpose() {
			HostMatrix<cudafloat> transpose(*this);
			transpose.ReplaceByTranspose();

			return transpose;
		}

		//! Multiplies the matrix by its own transpose and places the result in matrix C. More specifically C = alpha * (A * transpose(A)) + beta * C. This method uses the CUBLAS library.
		//! \attention Matrix C is returned in column-major order. If beta is different than zero, Matrix C must already be in column-major order.
		//! \param C matrix C. The number of rows and columns of C must be identical to the number of rows of A.
		//! \param alpha alpha scalar parameter (default value is 1.0).
		//! \param beta beta scalar parameter (default value is 0.0). If beta is different than zero, Matrix C must already be in column-major order.
		//! \sa IsRowMajor
		//! \warning The CUBLAS library must be initialized prior to the use of this method. Use cublasInit() to initialize the CUBLAS library. Don't forget to call cublasShutdown() when you no longer need to use the CUBLAS library.
		void MultiplyBySelfTranspose(DeviceMatrix<cudafloat> & C, cudafloat alpha = 1, cudafloat beta = 0) {
			assert(C.rows <= INT_MAX && C.rows == rows && C.columns == rows);

			if (C.IsRowMajor()) {
				assert(beta == 0);
				C.storingOrder = ColumnMajor;
			}

			size_t ldAB = IsRowMajor() ? this->columns : this->rows;
			cublasSgemm(this->IsRowMajor() ? 'T' : 'N', this->IsRowMajor() ? 'N' : 'T', (int)C.rows, (int)C.columns, (int)columns, alpha, this->Pointer(), (int)ldAB, this->Pointer(), (int)ldAB, beta, C.Pointer(), (int)C.rows);
		}

		//! Multiplies matrix A by Matrix B and places the result in C. More specifically C = alpha * (A * B) + beta * C. This method uses the CUBLAS library.
		//! \attention Matrix C is returned in column-major order. If beta is different than zero, Matrix C must already be in column-major order.
		//! \attention Best performance is achieved when all matrices are in column-major order.
		//! \param A matrix A. The number of columns of A must be identical to the number of B rows.
		//! \param B matrix B. The number of rows of B must be identical to the number of A columns.
		//! \param C matrix C. The number of rows of C must be identical to the number of rows of A and the number of columns of C must be identical to the number of columns of B.
		//! \param alpha alpha scalar parameter (default value is 1.0).
		//! \param beta beta scalar parameter (default value is 0.0). If beta is different than zero, Matrix C must already be in column-major order.
		//! \sa IsRowMajor, ReplaceByTranspose, Transpose
		//! \warning The CUBLAS library must be initialized prior to the use of this method. Use cublasInit() to initialize the CUBLAS library. Don't forget to call cublasShutdown() when you no longer need to use the CUBLAS library.
		static void Multiply(DeviceMatrix<cudafloat> & A, DeviceMatrix<cudafloat> & B, DeviceMatrix<cudafloat> & C, cudafloat alpha = 1, cudafloat beta = 0) {
			assert(A.columns <= INT_MAX && C.rows <= INT_MAX && C.columns <= INT_MAX);
			assert(A.columns == B.rows && C.rows == A.rows && C.columns == B.columns);

			if (C.IsRowMajor()) {
				assert(beta == 0);
				C.storingOrder = ColumnMajor;
			}

			cublasSgemm(A.IsRowMajor() ? 'T' : 'N', B.IsRowMajor() ? 'T' : 'N', (int)C.rows, (int)C.columns, (int)A.columns, alpha, A.Pointer(), (int)(A.IsRowMajor() ? A.columns : A.rows), B.Pointer(), (int)(B.IsRowMajor() ? B.columns : B.rows), beta, C.Pointer(), (int)C.rows);
		}
	};

	//! @}

}

#endif //GPUMLIB_DEVICE_MATRIX_H
