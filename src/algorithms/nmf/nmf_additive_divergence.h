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

#ifndef GPUMLIB_NMF_ADDITIVE_DIVERGENCE_H
#define GPUMLIB_NMF_ADDITIVE_DIVERGENCE_H

#include "../../memory/DeviceArray.h"
#include "../../common/Utilities.h"
#include "nmf_base.h"

namespace GPUMLib {

	//! \addtogroup nmf Non-negative Matrix Factorization classes
	//! @{

	//! Represents a Non-Negative Matrix Factorization (NMF) algorithm that uses additive update rules and the (Kullback-Leibler) divergence metric.
	class NMF_AdditiveDivergence : public NMF {
	private:
		DeviceArray<cudafloat> sum;
		dim3 bh, gh;
		dim3 bw, gw;

	public:
		//! Constructs a Non-negative Matrix Factorization (NMF) algorithm object that uses additive update rules and the (Kullback-Leibler) divergence metric.
		//! Given a non-negative matrix V (n x m), the NMF algorithm will find non-negative matrix factors W (n x r) and H (r x m) such that V is approximately equal to WH.
		//! \param v n x m matrix (V) containing a set of multivariate n-dimensional data vectors. m is the number of examples in the dataset.
		//! \param r Determines the size of matrices W (n x r) and H (r x m).
		//! \attention Matrix V must be in column-major order
		NMF_AdditiveDivergence(HostMatrix<cudafloat> & v, int r) : NMF(v, r), sum(r), bh(32, 16), gh(NumberBlocks((int)v.Columns(), 32), NumberBlocks(r, 32)), bw(16, 32), gw(NumberBlocks(r, 32), NumberBlocks((int)v.Rows(), 32)) {
		}

		//! Constructs a Non-negative Matrix Factorization (NMF) algorithm object that uses additive update rules and the (Kullback-Leibler) divergence metric.
		//! Given a non-negative matrix V (n x m), the NMF algorithm will find non-negative matrix factors W (n x r) and H (r x m) such that V is approximately equal to WH.
		//! \param v n x m matrix (V) containing a set of multivariate n-dimensional data vectors. m is the number of examples in the dataset.
		//! \param w n x r matrix.
		//! \param h r x m matrix.
		//! \attention Matrix V must be in column-major order
		NMF_AdditiveDivergence(HostMatrix<cudafloat> & v, HostMatrix<cudafloat> & w, HostMatrix<cudafloat> & h) : NMF(v, w, h), sum(w.Columns()), bh(32, 16), gh(NumberBlocks((int)v.Columns(), 32), NumberBlocks((int)h.Rows(), 32)), bw(16, 32), gw(NumberBlocks((int)w.Columns(), 32), NumberBlocks((int)v.Rows(), 32)) {
		}

		//! Do an algorithm iteration. Adjusts W and H matrices
		//! \param updateW Specifies if the matrix W is to be updated.
		void DoIteration(bool updateW = true);
	};

	//! @}

}

#endif // GPUMLIB_NMF_ADDITIVE_DIVERGENCE_H
