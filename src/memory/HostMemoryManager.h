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

#ifndef GPUMLIB_HOST_MEMORY_MANAGER_H
#define GPUMLIB_HOST_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include <cstring>
#include <new>

#include "MemoryManager.h"

namespace GPUMLib {

//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
//! @{

//! Host (CPU) memory manager class
template <class Type> class HostMemoryManager : public MemoryManager<Type> {
	public:
		virtual void Alloc(size_t size) {
			if (size > 0) {
				this->data = new (std::nothrow) Type[size];
				this->size = (this->data != nullptr) ? size : 0;
			} else {
				this->Reset();
			}
		}

		virtual void Dispose() {
			if (this->size > 0) delete [] this->data;
			this->Reset();
		}

		virtual void CopyDataFromDevice(Type * data, size_t size) {
			this->ResizeWithoutPreservingData(size);

			if (this->size > 0) {
				cudaMemcpy(this->data, data, this->SizeInBytes(), cudaMemcpyDeviceToHost);
			}
		}

		virtual void CopyDataFromHost(Type * data, size_t size) {
			this->ResizeWithoutPreservingData(size);

			if (this->size > 0) {
				memcpy(this->data, data, this->SizeInBytes());
			}
		}

		~HostMemoryManager() {
			Dispose();
		}
};

//! @}

}

#endif //GPUMLIB_HOST_MEMORY_MANAGER_H
