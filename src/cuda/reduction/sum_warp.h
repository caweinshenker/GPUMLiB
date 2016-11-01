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

#ifndef GPUMLib_SumWarp_h
#define GPUMLib_SumWarp_h

#include "../definitions.h"

namespace GPUMLib {

	template <int blockSize> __device__ __forceinline__ void SumBeforeWarp(cudafloat * s) {
		if (blockSize >= 1024) {
			if (threadIdx.x < 512) s[threadIdx.x] += s[threadIdx.x + 512];
			__syncthreads();
		}

		if (blockSize >= 512) {
			if (threadIdx.x < 256) s[threadIdx.x] += s[threadIdx.x + 256];
			__syncthreads();
		}

		if (blockSize >= 256) {
			if (threadIdx.x < 128) s[threadIdx.x] += s[threadIdx.x + 128];
			__syncthreads();
		}

		if (blockSize >= 128) {
			if (threadIdx.x < 64) s[threadIdx.x] += s[threadIdx.x + 64];
			__syncthreads();
		}
	}

	template <int blockSize> __device__ __forceinline__ void SumWarp(volatile cudafloat * s) {
		if (blockSize >= 64) s[threadIdx.x] += s[threadIdx.x + 32];
		if (blockSize >= 32) s[threadIdx.x] += s[threadIdx.x + 16];
		if (blockSize >= 16) s[threadIdx.x] += s[threadIdx.x + 8];
		if (blockSize >= 8) s[threadIdx.x] += s[threadIdx.x + 4];
		if (blockSize >= 4) s[threadIdx.x] += s[threadIdx.x + 2];
		if (blockSize >= 2) s[threadIdx.x] += s[threadIdx.x + 1];
	}

}

#endif
