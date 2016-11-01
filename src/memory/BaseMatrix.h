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
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GPUMLIB_BASE_MATRIX_H
#define GPUMLIB_BASE_MATRIX_H

#include "MemoryManager.h"

namespace GPUMLib {

//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
//! @{

//! Defines the methods for storing information in the matrix (either row-major or column-major).
typedef enum {
	RowMajor,
	ColumnMajor
} StoringOrder;

template <class Type> class CudaMatrix;

//! Base class for HostMatrix and DeviceMatrix classes (Matrix base class)
template <class Type> class BaseMatrix {
	friend class CudaMatrix<Type>;

	protected:
		StoringOrder storingOrder;
		size_t rows;
		size_t columns;

	private:
		MemoryManager<Type> * mem;

		void CompleteAssign(const BaseMatrix<Type> & other) {
			if (mem->Size() == other.Elements()) {
				this->rows = other.rows;
				this->columns = other.columns;
				this->storingOrder = other.storingOrder;
			}
		}

	public:
		//! Resizes the matrix without preserving its data
		//! \param rows the new number of rows
		//! \param columns the new number of columns
		//! \return the number of elements of the matrix after being resized.
		size_t ResizeWithoutPreservingData(size_t rows, size_t columns) {
			size_t newElements = rows * columns;

			if (mem->ResizeWithoutPreservingData(newElements) == newElements) {
				this->rows = rows;
				this->columns = columns;
			} else {
				this->rows = 0;
				this->columns = 0;
			}

			return Elements();
		}

	protected:
		BaseMatrix(MemoryManager<Type> & mem, StoringOrder storingOrder = RowMajor) {
			this->mem = &mem;
			this->storingOrder = storingOrder;
			this->rows = 0;
			this->columns = 0;
		}

		void AssignHostMatrix(const BaseMatrix<Type> & other) {
			mem->CopyDataFromHost(other.Pointer(), other.Elements());
			CompleteAssign(other);
		}

		void AssignDeviceMatrix(const BaseMatrix<Type> & other) {
			mem->CopyDataFromDevice(other.Pointer(), other.Elements());
			CompleteAssign(other);
		}

	public:
		//! Disposes the matrix.
		void Dispose() {
			mem->Dispose();
			rows = columns = 0;
		}

		//! Gets the number of rows of the matrix
		//! \return the number of rows of the matrix
		size_t Rows() const {
			return rows;
		}

		//! Gets the number of columns of the matrix
		//! \return the number of columns of the matrix
		size_t Columns() const {
			return columns;
		}

		//! Gets a pointer to the matrix data
		//! \attention Use with caution. Special attention should be given to how the matrix information is stored (row-major or column-major).
		//! \return a pointer to the matrix data
		//! \sa IsRowMajor
		Type * Pointer() const {
			return mem->Pointer();
		}

		//! Gets the number of elements contained in the matrix
		//! \return the number of elements contained in the matrix
		size_t Elements() const {
			return mem->Size();
		}

		//! Indicates if the information in the matrix is stored in row-major order.
		//! \return True if the matrix information is stored in row-major order. False if the information is stored in column-major format.
		bool IsRowMajor() const {
			return (storingOrder == RowMajor);
		}

		//! Replaces this matrix by its transpose
		//! \attention This method is very fast, however it changes the method for storing information in the matrix (row-major or column-major).
		//! \sa IsRowMajor
		void ReplaceByTranspose() {
			size_t newRows = columns;

			columns = rows;
			rows = newRows;
			storingOrder = (IsRowMajor()) ? ColumnMajor : RowMajor;
		}

		void TransferOwnerShipFrom(BaseMatrix<Type> & other) {
			if (this != &other) {
				mem->TransferOwnerShipFrom(*(other.mem));

				storingOrder = other.storingOrder;
				rows = other.rows;
				columns = other.columns;

				other.rows = 0;
				other.columns = 0;
			}
		}
};

//! @}

}

#endif //GPUMLIB_BASE_MATRIX_H
