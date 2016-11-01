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

#ifndef GPUMLIB_DEVICE_MEMORY_MANAGER_H
#define GPUMLIB_DEVICE_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include "MemoryManager.h"

namespace GPUMLib {

//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
//! @{

//! Device (GPU) memory manager class
template <class Type> class DeviceMemoryManager : public MemoryManager<Type> {
	public:
		virtual void Alloc(size_t size) {
			if (size > 0 && cudaMalloc((void **) &(this->data), size * sizeof(Type)) == cudaSuccess) {
				this->size = size;
			} else {
				this->Reset();
			}
		}

		virtual void Dispose() {
			if (this->size > 0) cudaFree(this->data);
			this->Reset();
		}

		virtual void CopyDataFromDevice(Type * data, size_t size) {
			this->ResizeWithoutPreservingData(size);

			if (this->size > 0) {
				cudaMemcpy(this->data, data, this->SizeInBytes(), cudaMemcpyDeviceToDevice);
			}
		}

		virtual void CopyDataFromHost(Type * data, size_t size) {
			this->ResizeWithoutPreservingData(size);

			if (this->size > 0) {
				cudaMemcpy(this->data, data, this->SizeInBytes(), cudaMemcpyHostToDevice);
			}
		}

		~DeviceMemoryManager() {
			Dispose();
		}
};

//! @}

}

#endif //GPUMLIB_DEVICE_MEMORY_MANAGER_H
