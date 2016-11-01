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

#ifndef GPUMLIB_BASE_ARRAY_H
#define GPUMLIB_BASE_ARRAY_H

#include "MemoryManager.h"

namespace GPUMLib {

//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
//! @{

template <class Type> class CudaArray;

//! Base class for HostArray and DeviceArray classes (Array base class)
template <class Type> class BaseArray {
	friend class CudaArray<Type>;

	private:
		MemoryManager<Type> * mem;

	protected:
		BaseArray(MemoryManager<Type> & mem) {
			this->mem = &mem;
		}

	public:
		//! Disposes the array.
		void Dispose() {
			mem->Dispose();
		}

		//! Gets the length of the array.
		//! You can use this function to check if the array was effectively allocated.
		//! \return the number of elements of the array
		size_t Length() const {
			return mem->Size();
		}

		//! Gets a pointer to the array data
		//! \attention Use with caution
		//! \return a pointer to the array data
		Type * Pointer() const {
			return mem->Pointer();
		}

		//! Resizes the array without preserving its data
		//! \param size new size of the array
		//! \return the number of elements of the array after being resized.
		size_t ResizeWithoutPreservingData(size_t size) {
			return mem->ResizeWithoutPreservingData(size);
		}
};

//! @}

}

#endif //GPUMLIB_BASE_ARRAY_H
