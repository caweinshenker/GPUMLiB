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

#ifndef GPUMLIB_DEVICE_MATRIX_3D_H
#define GPUMLIB_DEVICE_MATRIX_3D_H

#include <cassert>
#include <cublas.h>

#include "HostMatrix3D.h"
#include "DeviceMemoryManager.h"

namespace GPUMLib {

	//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
	//! @{

	//! Create a 3D matrix of any type, on the device, that automatically manages the memory used to hold its elements
	template <class Type> class DeviceMatrix3D : public BaseMatrix3D < Type > {
	private:
		DeviceMemoryManager<Type> deviceMem;

	public:
		//! Constructs an empty 3D device matrix
		//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
		DeviceMatrix3D() : BaseMatrix3D<Type>(deviceMem) {}

		//! Constructs a 3D device matrix with the given dimensions
		//! \param dimX x matrix dimension
		//! \param dimY y matrix dimension
		//! \param dimZ z matrix dimension
		DeviceMatrix3D(size_t dimX, size_t dimY, size_t dimZ) : BaseMatrix3D<Type>(deviceMem) {
			this->ResizeWithoutPreservingData(dimX, dimY, dimZ);
		}

		//! Constructs a 3D matrix identical to another
		//! \param other another matrix
		DeviceMatrix3D(const DeviceMatrix3D<Type> & other) : BaseMatrix3D<Type>(deviceMem) {
			this->AssignDeviceMatrix(other);
		}

		//! Constructs a matrix identical to an host matrix
		//! \param other host matrix
		DeviceMatrix3D(const HostMatrix3D<Type> & other) : BaseMatrix3D<Type>(deviceMem) {
			this->AssignHostMatrix(other);
		}

		//! Transforms this matrix into an matrix identical to an host matrix
		//! \param other host matrix
		//! \return a reference to this matrix
		//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
		//! \sa IsRowMajor
		DeviceMatrix3D<Type> & operator = (const HostMatrix3D<Type> & other) {
			this->AssignHostMatrix(other);
			return *this;
		}

		//! Transforms this matrix into an matrix identical to the other
		//! \param other other matrix
		//! \return a reference to this matrix
		//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
		//! \sa IsRowMajor
		DeviceMatrix3D<Type> & operator = (const DeviceMatrix3D<Type> & other) {
			this->AssignDeviceMatrix(other);
			return *this;
		}

		//! Constructs a matrix using the elements of a device temporary matrix (rvalue)
		//! \param temporaryMatrix temporary device matrix containing the elements
		DeviceMatrix3D(DeviceMatrix3D<Type> && temporaryMatrix) : BaseMatrix3D<Type>(deviceMem) {
			this->TransferOwnerShipFrom(temporaryMatrix);
		}

		//! Replaces this matrix using a temporary matrix (rvalue)
		//! \param temporaryMatrix temporary matrix
		//! \return a reference to this matrix
		DeviceMatrix3D<Type> & operator = (DeviceMatrix3D<Type> && temporaryMatrix) {
			this->TransferOwnerShipFrom(temporaryMatrix);
			return *this;
		}
	};

	//! @}

}

#endif //GPUMLIB_DEVICE_MATRIX_3D_H
