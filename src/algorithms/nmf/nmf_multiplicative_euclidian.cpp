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

#include "nmf_multiplicative_euclidian.h"

namespace GPUMLib {

	//! \addtogroup nmf Non-negative Matrix Factorization classes
	//! @{

	void NMF_MultiplicativeEuclidianDistance::DoIteration(bool updateW) {
		DetermineQualityImprovement(true);

		// Calculate Wt
		W.ReplaceByTranspose();
		DeviceMatrix<cudafloat> & Wt = W;

		// Calculate WtV
		DeviceMatrix<cudafloat>::Multiply(Wt, V, WtV);

		// Calculate WtW
		Wt.MultiplyBySelfTranspose(WtW);

		// Calculate WtWH
		DeviceMatrix<cudafloat>::Multiply(WtW, H, WtWH);

		gpumlib_cuda_nmf_multiplicative_euclidean_update_matrix(blocksH, SIZE_BLOCKS_NMF, WtV.Pointer(), WtWH.Pointer(), H.Pointer(), (int)H.Elements());

		Wt.ReplaceByTranspose();

		if (!updateW) return;

		// Calculate Ht
		H.ReplaceByTranspose();
		DeviceMatrix<cudafloat> & Ht = H;

		// Calculate VHt
		DeviceMatrix<cudafloat>::Multiply(V, Ht, VHt);

		// Calculate HHt
		DeviceMatrix<cudafloat> & HHt = WtW;
		Ht.ReplaceByTranspose();
		H.MultiplyBySelfTranspose(HHt);

		// Calculate WHHt
		DeviceMatrix<cudafloat>::Multiply(W, HHt, WHHt);

		gpumlib_cuda_nmf_multiplicative_euclidean_update_matrix(blocksW, SIZE_BLOCKS_NMF, VHt.Pointer(), WHHt.Pointer(), W.Pointer(), (int)W.Elements());
	}

	//! @}

}
