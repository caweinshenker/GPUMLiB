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

#include "nmf_additive_euclidian.h"
#include "../../common/Utilities.h"

namespace GPUMLib {

	//! \addtogroup nmf Non-negative Matrix Factorization classes
	//! @{

	// V (n x m) | W (n x r) | H (r x m)
	void NMF_AdditiveEuclidian::DoIteration(bool updateW) {
		DetermineQualityImprovement(true);

		// Update H
		W.ReplaceByTranspose();
		DeviceMatrix<cudafloat>::Multiply(W, V, deltaH);
		W.MultiplyBySelfTranspose(aux);
		//DeviceMatrix<cudafloat>::Multiply(aux, H, deltaH, CUDA_VALUE(-1.0), CUDA_VALUE(1.0));
		DeviceMatrix<cudafloat>::Multiply(aux, H, deltaH2);
		W.ReplaceByTranspose();
		//UpdateMatrixNMFadditive<<<NumberBlocks(H.Elements(), SIZE_BLOCKS_NMF), SIZE_BLOCKS_NMF>>>(H.Pointer(), deltaH.Pointer(), CUDA_VALUE(0.001), H.Elements());
		gpumlib_cuda_nmf_multiplicative_euclidean_update_matrix(NumberBlocks((int)H.Elements(), SIZE_BLOCKS_NMF), SIZE_BLOCKS_NMF, H.Pointer(), deltaH.Pointer(), deltaH2.Pointer(), (int)H.Elements());

		if (!updateW) return;

		// Update W
		H.ReplaceByTranspose();
		DeviceMatrix<cudafloat>::Multiply(V, H, deltaW);
		H.ReplaceByTranspose();
		H.MultiplyBySelfTranspose(aux);
		//DeviceMatrix<cudafloat>::Multiply(W, aux, deltaW, CUDA_VALUE(-1.0), CUDA_VALUE(1.0));
		DeviceMatrix<cudafloat>::Multiply(W, aux, deltaW2);
		//UpdateMatrixNMFadditive<<<NumberBlocks(W.Elements(), SIZE_BLOCKS_NMF), SIZE_BLOCKS_NMF>>>(W.Pointer(), deltaW.Pointer(), CUDA_VALUE(0.001), W.Elements());
		
		gpumlib_cuda_nmf_additive_euclidean_update_matrix(NumberBlocks((int)W.Elements(), SIZE_BLOCKS_NMF), SIZE_BLOCKS_NMF, W.Pointer(), deltaW.Pointer(), deltaW2.Pointer(), (int)W.Elements());
	}

	//! @}

}
