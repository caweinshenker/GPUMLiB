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

//! \addtogroup commonframework Common framework
//! @{

#ifndef GPUMLIB_CUDA_DEFINITIONS_H
#define GPUMLIB_CUDA_DEFINITIONS_H

#include <float.h>
#include <math.h>


//! Tells cuda to use single precision variables.
//! To use double precision variables and values comment this define. You also need to compile with -arch sm_13
#define USE_SINGLE_PRECISION_VARIABLES

//! Tells cuda to use single precision functions.
//! To use double precision functions (on supported devices) comment this define
#define USE_SINGLE_PRECISION_FUNCTIONS

#ifndef CUDA_MAX_THREADS_PER_BLOCK
//! Defines the maximum number of CUDA threads per block
#define CUDA_MAX_THREADS_PER_BLOCK 512
#endif

//! Defines the size of a small vector (for reduction purposes).
//! This value defines the size of a vector for which computing a reduction (sum, average, max, min, ...) using a single block is faster than using several blocks. This value is optimized for a GTX 280.
//! \attention If you optimize the value for other boards, please send me the information (noel@ipg.pt) so that I can include it here (thank you).
#define SIZE_SMALL_CUDA_VECTOR (3 * CUDA_MAX_THREADS_PER_BLOCK)

//! Defines the optimal block size for reduction operations. This value is optimized for a GTX 280.
//! \attention If you optimize the value for other boards, please send me the information (noel@ipg.pt) so that I can include it here (thank you).
#define OPTIMAL_BLOCK_SIZE_REDUCTION 128

#ifndef CUDA_MAX_GRID_X_DIM
//! Defines the maximum x-dimension of a CUDA grid of thread blocks
#define CUDA_MAX_GRID_X_DIM 65535
#endif

//! Defines the maximum y-dimension of a CUDA grid of thread blocks
#define CUDA_MAX_GRID_Y_DIM 65535

//! Defines the maximum z-dimension of a CUDA grid of thread blocks
#define CUDA_MAX_GRID_Z_DIM 65535

#ifdef USE_SINGLE_PRECISION_VARIABLES
//! Type used by CUDA to represent floating point numbers. If USE_SINGLE_PRECISION_VARIABLES is defined, cudafloat represents a float. Otherwise it represents a double
//! \sa USE_SINGLE_PRECISION_VARIABLES
typedef float cudafloat;

//! Represents a floating point number, accordingly to the type used by CUDA to represent floating point values.
//! \sa cudafloat
//! \sa USE_SINGLE_PRECISION_VARIABLES
#define CUDA_VALUE(X) (X##f)

//! Maximum value that a cudafloat can hold.
//! \sa cudafloat
//! \sa USE_SINGLE_PRECISION_VARIABLES
#define MAX_CUDAFLOAT (FLT_MAX)

//! Minimum positive value that a cudafloat can hold.
//! \sa MIN_CUDAFLOAT
//! \sa cudafloat
//! \sa USE_SINGLE_PRECISION_VARIABLES
#define MIN_POSITIVE_CUDAFLOAT (FLT_MIN)

//! Minimum (negative) value that a cudafloat can hold.
//! \sa MIN_POSITIVE_CUDAFLOAT
//! \sa cudafloat
//! \sa USE_SINGLE_PRECISION_VARIABLES
//! \attention Check also MIN_POSITIVE_CUDAFLOAT to see which one is appropriate for your purposes.
#define MIN_CUDAFLOAT (-FLT_MAX)
#else // double precision variables and values
//! Type used by CUDA to represent floating point numbers. If USE_SINGLE_PRECISION_VARIABLES is defined, cudafloat represents a float. Otherwise it represents a double
//! \sa USE_SINGLE_PRECISION_VARIABLES
typedef double cudafloat;

//! Represents a floating point number, accordingly to the type used by CUDA to represent floating point values.
//! \sa cudafloat
//! \sa USE_SINGLE_PRECISION_VARIABLES
#define CUDA_VALUE(X) (X)

//! Maximum value that a cudafloat can hold.
//! \sa cudafloat
//! \sa USE_SINGLE_PRECISION_VARIABLES
#define MAX_CUDAFLOAT (DBL_MAX)

//! Minimum positive value that a cudafloat can hold.
//! \sa MIN_CUDAFLOAT
//! \sa cudafloat
//! \sa USE_SINGLE_PRECISION_VARIABLES
#define MIN_POSITIVE_CUDAFLOAT (DBL_MIN)

//! Minimum (negative) value that a cudafloat can hold.
//! \sa cudafloat
//! \sa USE_SINGLE_PRECISION_VARIABLES
#define MIN_CUDAFLOAT (-DBL_MAX)
#endif

#ifdef USE_SINGLE_PRECISION_FUNCTIONS
//! Defines the exponential function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_EXP  expf

//! Defines the square root function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_SQRT sqrtf

//! Defines the hyperbolic tangent function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_TANH tanhf

//! Defines the hyperbolic cosine function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_COSH coshf

//! Defines the power function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_POW powf

//! Defines the square root function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_RSQRT rsqrtf
#else // double precision functions
//! Defines the exponential function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_EXP  exp

//! Defines the square root function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_SQRT sqrt

//! Defines the hyperbolic tangent function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_TANH tanh

//! Defines the hyperbolic cosine function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_COSH cosh

//! Defines the power function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_POW pow

//! Defines the square root function to be used by CUDA. Depends on USE_SINGLE_PRECISION_FUNCTIONS being defined.
//! \sa USE_SINGLE_PRECISION_FUNCTIONS
#define CUDA_RSQRT rsqrtf
#endif

//! Defines the logistic function to be used by CUDA.
//! \sa CUDA_EXP
#define CUDA_SIGMOID(X) (CUDA_VALUE(1.0) / (CUDA_VALUE(1.0) + CUDA_EXP(-(X))))

//! Defines the logistic function derivate to be used by CUDA.
//! \sa CUDA_SIGMOID
#define CUDA_SIGMOID_DERIVATE(OUTPUT) ((OUTPUT) * (CUDA_VALUE(1.0) - (OUTPUT)))

//! Verifies if \b X and \b Y have the same signal.
#define SAME_DIRECTION(X, Y) (((X) > CUDA_VALUE(0.0) && (Y) > CUDA_VALUE(0.0)) || ((X) < CUDA_VALUE(0.0) && (Y) < CUDA_VALUE(0.0)))

namespace GPUMLib {
#if defined(__CUDA_ARCH__)

	__device__ __forceinline__ bool IsInfOrNaN(cudafloat x) {
		return (isnan(x) || isinf(x));
	}

	__device__ __forceinline__ cudafloat Log(cudafloat x) {
		if (x != 0) {
			cudafloat y = log(x);
			if (!IsInfOrNaN(y)) return y;
		}

		return CUDA_VALUE(-7.0); //log(0.0000001)
	}

#else

	inline bool IsInfOrNaN(cudafloat x) {
#if (defined(_MSC_VER))
		return (!_finite(x));
#else
		return (isnan(x) || isinf(x));
#endif
	}

	inline cudafloat Log(cudafloat x) {
		if (x != 0) {
			cudafloat y = log(x);
			if (!IsInfOrNaN(y)) return y;
		}

		return CUDA_VALUE(-7.0); //log(0.0000001)
	}

	inline cudafloat AbsDiff(cudafloat a, cudafloat b) {
		cudafloat d = a - b;
		if (d < cudafloat(0.0)) return -d;
		return d;
	}

#endif
}

#endif // GPUMLIB_CUDA_DEFINITIONS_H

//! @}
