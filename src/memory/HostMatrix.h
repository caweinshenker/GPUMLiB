﻿/*
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

#ifndef GPUMLIB_HOST_MATRIX_H
#define GPUMLIB_HOST_MATRIX_H

#include <cassert>
#include "BaseMatrix.h"
#include "HostMemoryManager.h"

namespace GPUMLib {

	//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
	//! @{

	template <class Type> class DeviceMatrix;

	//! Create a matrix of any type, on the host, that automatically manages the memory used to hold its elements
	template <class Type> class HostMatrix : public BaseMatrix < Type > {
	private:
		HostMemoryManager<Type> hostMem;

		size_t Index(size_t row, size_t column) const {
			assert(row < this->rows && column < this->columns);

			return (this->IsRowMajor()) ? row * this->columns + column : column * this->rows + row;
		}

	public:
		//! Constructs an empty matrix
		//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
		HostMatrix(StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(hostMem, storingOrder) {}

		//! Constructs a matrix with a given number of rows and columns
		//! \param rows the number of rows
		//! \param columns the number of columns
		//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
		HostMatrix(size_t rows, size_t columns, StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(hostMem, storingOrder) {
			this->ResizeWithoutPreservingData(rows, columns);
		}

		//! Constructs a matrix identical to the other
		//! \param other another matrix
		HostMatrix(const HostMatrix<Type> & other) : BaseMatrix<Type>(hostMem) {
			this->AssignHostMatrix(other);
		}

		//! Constructs a matrix identical to a device matrix
		//! \param other device matrix
		HostMatrix(const DeviceMatrix<Type> & other) : BaseMatrix<Type>(hostMem) {
			this->AssignDeviceMatrix(other);
		}

		//! Transforms this matrix into an matrix identical to the other
		//! \param other other matrix
		//! \return a reference to this matrix
		//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
		//! \sa IsRowMajor
		HostMatrix<Type> & operator = (const HostMatrix<Type> & other) {
			this->AssignHostMatrix(other);
			return *this;
		}

		//! Transforms this matrix into an matrix identical a device matrix
		//! \param other device matrix
		//! \return a reference to this matrix
		//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
		//! \sa IsRowMajor
		HostMatrix<Type> & operator = (const DeviceMatrix<Type> & other) {
			this->AssignDeviceMatrix(other);
			return *this;
		}

		//! Constructs a matrix using the elements of a temporary matrix (rvalue)
		//! \param temporaryMatrix temporary matrix containing the elements
		HostMatrix(HostMatrix<Type> && temporaryMatrix) : BaseMatrix<Type>(hostMem) {
			this->TransferOwnerShipFrom(temporaryMatrix);
		}

		//! Replaces this matrix using a temporary matrix (rvalue)
		//! \param temporaryMatrix temporary matrix
		//! \return a reference to this matrix
		HostMatrix<Type> & operator = (HostMatrix<Type> && temporaryMatrix) {
			this->TransferOwnerShipFrom(temporaryMatrix);
			return *this;
		}

		//! Gets the transposed of the matrix
		//! \return the transposed of the matrix
		//! \attention The returned matrix does not use the same method (row-major or column-major) for storing information as this matrix.
		//! \sa ReplaceByTranspose, IsRowMajor
		HostMatrix<Type> Transpose() {
			HostMatrix<Type> transpose(*this);
			transpose.ReplaceByTranspose();

			return transpose;
		}

		//! Gets a reference to an element of the matrix
		//! \param row row of the desired element
		//! \param column column of the desired element
		//! \return a reference to an element desired, based on the row and column specified
		Type & operator()(size_t row, size_t column) {
			return this->Pointer()[Index(row, column)];
		}

		//! Gets an element of the matrix
		//! \param row row of the desired element
		//! \param column column of the desired element
		//! \return the element desired, based on the row and column specified
		Type operator()(size_t row, size_t column) const {
			return this->Pointer()[Index(row, column)];
		}
	};

	//! @}

}

#endif //GPUMLIB_HOST_MATRIX_H
