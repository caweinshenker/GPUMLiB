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

#include "nmf_multiplicative_divergence.h"

namespace GPUMLib {

	//! \addtogroup nmf Non-negative Matrix Factorization classes
	//! @{

	void NMF_MultiplicativeDivergence::DoIteration(bool updateW) {
		int n = (int)V.Rows();
		int m = (int)V.Columns();
		int r = (int)W.Columns();

		// Update H
		DeviceMatrix<cudafloat>::Multiply(W, H, WH);

		DetermineQualityImprovement(false);

		KernelSumW(NumberThreadsPerBlockThatBestFit(n), W.Pointer(), n, r, sum.Pointer());

		gpumlib_cuda_nmf_multiplicative_divergence_update_matrix_h(gh, bh, H.Pointer(), W.Pointer(), V.Pointer(), WH.Pointer(), sum.Pointer(), n, m, r);

		if (!updateW) return;

		// Update W
		DeviceMatrix<cudafloat>::Multiply(W, H, WH);
		KernelSumH(NumberThreadsPerBlockThatBestFit(m), H.Pointer(), r, m, sum.Pointer());

		gpumlib_cuda_nmf_multiplicative_divergence_update_matrix_w(gw, bw, W.Pointer(), H.Pointer(), V.Pointer(), WH.Pointer(), sum.Pointer(), n, m, r);
	}

	//! @}

}
