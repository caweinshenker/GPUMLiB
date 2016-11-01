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

#ifndef GPUMLIB_HOST_ARRAY_H
#define GPUMLIB_HOST_ARRAY_H

#include <cassert>
#include "BaseArray.h"
#include "HostMemoryManager.h"

namespace GPUMLib {

	template <class Type> class DeviceArray;

	//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
	//! @{

	//! Create an array of any type, on the host, that automatically manages the memory used to hold its elements
	template <class Type> class HostArray : public BaseArray < Type > {
	private:
		HostMemoryManager<Type> hostMem;

	public:
		//! Constructs an array with no elements
		HostArray() : BaseArray<Type>(hostMem) {}

		//! Constructs an array with size elements
		//! \param size number of elements of the array
		explicit HostArray(size_t size) : BaseArray<Type>(hostMem) {
			hostMem.Alloc(size);
		}

		//! Constructs an array with the same elements as another array
		//! \param originalArray original device array from where to copy the elements
		HostArray(const DeviceArray<Type> & originalArray) : BaseArray<Type>(hostMem) {
			hostMem.CopyDataFromDevice(originalArray.Pointer(), originalArray.Length());
		}

		//! Constructs an array with the same elements as another array
		//! \param originalArray original array from where to copy the elements
		HostArray(const HostArray<Type> & originalArray) : BaseArray<Type>(hostMem) {
			hostMem.CopyDataFromHost(originalArray.Pointer(), originalArray.Length());
		}

		//! Transforms this array into an array identical to another array
		//! \param originalArray original device array from where to copy the elements
		//! \return a reference to this array
		HostArray<Type> & operator = (const DeviceArray<Type> & originalArray) {
			hostMem.CopyDataFromDevice(originalArray.Pointer(), originalArray.Length());
			return *this;
		}

		//! Transforms this array into an array identical to another array
		//! \param originalArray original array from where to copy the elements
		//! \return a reference to this array
		HostArray<Type> & operator = (const HostArray<Type> & originalArray) {
			hostMem.CopyDataFromHost(originalArray.Pointer(), originalArray.Length());
			return *this;
		}

		//! Releases its own resources (elements) and obtains ownership of another array resources.
		//! The other array will no longer have any elements.
		//! In other words, it moves the elements from one array to another.
		//! \param other array containing the elements to be moved.
		void TransferOwnerShipFrom(HostArray<Type> & other) {
			if (this != &other) hostMem.TransferOwnerShipFrom(other.hostmem);
		}

		//! Constructs an array using the elements of a temporary array (rvalue)
		//! \param temporaryArray temporary array containing the elements
		HostArray(HostArray<Type> && temporaryArray) : BaseArray<Type>(hostMem) {
			hostMem.TransferOwnerShipFrom(temporaryArray.hostMem);
		}

		//! Replaces the elements of this array by the elements of a temporary array (rvalue)
		//! \param temporaryArray temporary array containing the elements
		//! \return a reference to this array
		HostArray<Type> & operator = (HostArray<Type> && temporaryArray) {
			hostMem.TransferOwnerShipFrom(temporaryArray.hostMem);
			return *this;
		}

		//! Gets a reference to an element of the array
		//! \param element position of the desired element
		//! \return a reference to an element desired
		Type & operator [] (size_t element) {
			assert(element < this->Length());
			return this->Pointer()[element];
		}

		//! Gets an element of the array
		//! \param element position of the desired element
		//! \return the element desired
		Type operator [] (size_t element) const {
			assert(element < this->Length());
			return this->Pointer()[element];
		}
	};

	//! @}

}

#endif //GPUMLIB_HOST_ARRAY_H
