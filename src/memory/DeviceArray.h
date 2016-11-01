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

#ifndef GPUMLIB_DEVICE_ARRAY_H
#define GPUMLIB_DEVICE_ARRAY_H

#include "HostArray.h"
#include "DeviceMemoryManager.h"

namespace GPUMLib {

	//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
	//! @{

	//! Create an array of any type, on the device, that automatically manages the memory used to hold its elements
	template <class Type> class DeviceArray : public BaseArray < Type > {
	private:
		DeviceMemoryManager<Type> deviceMem;

	public:
		//! Constructs an array with no elements
		DeviceArray() : BaseArray<Type>(deviceMem) {}

		//! Constructs an array with size elements
		//! \param size number of elements of the array
		explicit DeviceArray(size_t size) : BaseArray<Type>(deviceMem) {
			deviceMem.Alloc(size);
		}

		//! Constructs a device array with the same elements as an host array
		//! \param originalArray host array from where to copy the elements
		DeviceArray(const HostArray<Type> & originalArray) : BaseArray<Type>(deviceMem) {
			deviceMem.CopyDataFromHost(originalArray.Pointer(), originalArray.Length());
		}

		//! Constructs a device array with the same elements as another device array
		//! \param originalArray array from where to copy the elements
		DeviceArray(const DeviceArray<Type> & originalArray) : BaseArray<Type>(deviceMem) {
			deviceMem.CopyDataFromDevice(originalArray.Pointer(), originalArray.Length());
		}

		//! Constructs a device array with the same elements as those in an host array
		//! \param originalArray array data from where to copy the elements
		//! \param size number of elements to copy
		DeviceArray(const Type * originalArray, size_t size) : BaseArray<Type>(deviceMem) {
			deviceMem.CopyDataFromHost(originalArray, size);
		}

		//! Transforms this array into an array identical to another array
		//! \param originalArray array from where to copy the elements
		//! \return a reference to this array
		DeviceArray<Type> & operator = (const DeviceArray<Type> & originalArray) {
			deviceMem.CopyDataFromDevice(originalArray.Pointer(), originalArray.Length());
			return *this;
		}

		//! Transforms this array into an array with the same data as an host array
		//! \param originalArray array from where to copy the elements
		//! \return a reference to this array
		DeviceArray<Type> & operator = (const HostArray<Type> & originalArray) {
			deviceMem.CopyDataFromHost(originalArray.Pointer(), originalArray.Length());
			return *this;
		}

		//! Releases its own resources (elements) and obtains ownership of another array resources.
		//! The other array will no longer have any elements.
		//! In other words, it moves the elements from one device array to another.
		//! \param other array containing the elements to be moved.
		void TransferOwnerShipFrom(DeviceArray<Type> & other) {
			if (this != &other) deviceMem.TransferOwnerShipFrom(other.deviceMem);
		}

		//! Constructs a device array using the elements of a temporary device array (rvalue)
		//! \param temporaryArray temporary array containing the elements
		DeviceArray(DeviceArray<Type> && temporaryArray) : BaseArray<Type>(deviceMem) {
			deviceMem.TransferOwnerShipFrom(temporaryArray.deviceMem);
		}

		//! Replaces the elements of this device array by the elements of a temporary device array (rvalue)
		//! \param temporaryArray temporary array containing the elements
		//! \return a reference to this array
		DeviceArray<Type> & operator = (DeviceArray<Type> && temporaryArray) {
			deviceMem.TransferOwnerShipFrom(temporaryArray.deviceMem);
			return *this;
		}
	};

	//! @}

}

#endif //GPUMLIB_DEVICE_ARRAY_H
