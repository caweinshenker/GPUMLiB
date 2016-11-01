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

#ifndef GPUMLIB_UTILITIES_H
#define GPUMLIB_UTILITIES_H

#include "../cuda/definitions.h"
#include "../memory/HostArray.h"

namespace GPUMLib {

//! \addtogroup commonframework Common framework
//! @{

//! Finds the number of threads (multiple of 2) per block that either is greater that the number of threads needed or identical to the maximum number of threads per block.
//! \param threads Number of threads.
//! \param maxThreadsPerBlock Maximum number of threads.
//! \return The number of threads (multiple of 2) per block that either is greater that the number of threads needed or identical to the maximum number of threads per block.
//! \sa CUDA_MAX_THREADS_PER_BLOCK, NumberBlocks
inline int NumberThreadsPerBlockThatBestFit(int threads, int maxThreadsPerBlock = CUDA_MAX_THREADS_PER_BLOCK) {
	int nt = 1;
	while(nt < threads && nt < maxThreadsPerBlock) nt <<= 1;

	return nt;
}

//! Finds the number of blocks needed to execute the number of threads specified, given a block size.
//! \param threads Number of threads.
//! \param blockSize Block size.
//! \return The number of blocks needed to execute the number of threads specified.
//! \sa NumberThreadsPerBlockThatBestFit, CUDA_MAX_THREADS_PER_BLOCK
inline int NumberBlocks(int threads, int blockSize) {
	int nb = threads / blockSize;

	if (threads % blockSize != 0) nb++;

	return nb;
}

//! Makes sure that the block does not have more than the maximum number of threads supported by CUDA, reducing the number of threads in each dimension if necessary.
//! \param block block.
//! \sa MAX_THREADS_PER_BLOCK
inline void MakeSureBlockDoesNotHaveTooMuchThreads(dim3 & block) {
	unsigned x = NumberThreadsPerBlockThatBestFit(block.x);
	unsigned y = NumberThreadsPerBlockThatBestFit(block.y);
	unsigned z = NumberThreadsPerBlockThatBestFit(block.z);

	while (x * y * z > CUDA_MAX_THREADS_PER_BLOCK) {
		if (z > 1 && z >= y) {
			z >>= 1;
		} else if (y >= x) {
			y >>= 1;
		} else {
			x >>= 1;
		}
	}

	// fix the value of z

	if (z < block.z) block.z = z;

	while (2 * x * y * block.z < CUDA_MAX_THREADS_PER_BLOCK) {
		if (x < block.x) {
			if (y < x && y < block.y) {
				y <<= 1;
			} else {
				x <<= 1;
			}
		} else if (y < block.y) {
			y <<= 1;
		} else {
			break;
		}
	}

	// fix the value of y

	if (y < block.y) block.y = y;

	while (x < block.x && 2 * x * y * z < CUDA_MAX_THREADS_PER_BLOCK) {
		x <<= 1;
	}

	// fix the value of x

	if (x < block.x) block.x = x;
}

//! @}

}

#endif //GPUMLIB_UTILITIES_H
