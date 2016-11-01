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

#ifndef GPUMLIB_DBN_HOST_H
#define GPUMLIB_DBN_HOST_H

#ifndef _CONSOLE
#include "../apps/common/progress/ProgressInfo.h"
#endif
#include "../RBM/HostRBM.h"

namespace GPUMLib {

	//! \addtogroup dbnh Deep Belief Networks Host (CPU) class
	//! @{

	//! Represents a Deep Belief Network (Host - CPU).
	class DBNhost {
	private:
		HostArray<RBMhost *> rbms;

	public:
		//! Constructs a Deep Belief Network that can be trained using the CPU (Host).
		//! \param layers Number of units in each layer.
		//! \param inputs Inputs of the training dataset. Each row of the matrix should contain a pattern (sample) and each column an input.
		//! \param initialLearningRate Initial learning rate.
		//! \param momentum Momentum
		//! \param useBinaryValuesVisibleReconstruction Use binary values for the visibible layer reconstruction
		//! \param stdWeights Defines the maximum and minimum value for the weights. The weights will be initialized with a random number between -stdWeights and stdWeights.
		DBNhost(HostArray<int> & layers, HostMatrix<cudafloat> & inputs, cudafloat initialLearningRate, cudafloat momentum = DEFAULT_MOMENTUM, bool useBinaryValuesVisibleReconstruction = false, cudafloat stdWeights = STD_WEIGHTS) {
			int nlayers = (int)layers.Length();
			assert(nlayers >= 2);

			rbms.ResizeWithoutPreservingData(nlayers - 1);

			rbms[0] = new RBMhost(layers[0], layers[1], inputs, initialLearningRate, momentum, useBinaryValuesVisibleReconstruction, stdWeights);

			for (int r = 1; r < nlayers - 1; r++) {
				rbms[r] = new RBMhost(rbms[r - 1], layers[r + 1], initialLearningRate, momentum, useBinaryValuesVisibleReconstruction, stdWeights);
			}
		}

		~DBNhost() {
			int nlayers = (int)rbms.Length();

			for (int l = 0; l < nlayers; l++) {
				delete rbms[l];
				rbms[l] = nullptr;
			}
		}

		//! Get an RBM that is part of the DBN.
		//! \param layer Layer to obtain.
		//! \return The RBM corresponding to layer specified.
		RBMhost * GetRBM(int layer) {
			assert(layer < (int)rbms.Length());
			return rbms[layer];
		}

		//! Gets the number of RBMs.
		//! \return The number of RBMs that compose the DBN.
		int GetNumberRBMs() const {
			return (int)rbms.Length();
		}

		//! Train the DBN (train each RBM).
		//! \param epochs Maximum number of epochs that each RBM should be trained.
		//! \param cd Define the value of k in CD-k
		void Train(int epochs, int cd) {
			int nlayers = (int)rbms.Length();
			for (int l = 0; l < nlayers; l++) {
				RBMhost * layer = rbms[l];
				if (l > 0) layer->RandomizeWeights();

				for (int e = 0; e < epochs; e++) layer->ContrastiveDivergence(cd);
				layer->ComputeStatusHiddenUnits(*(layer->v), layer->h_data);
			}
		}

		//! Train the DBN (train each RBM).
		//! \param epochs Maximum number of epochs that each RBM should be trained.
		//! \param cd Define the value of k in CD-k
		//! \param errorStop Stop the training in each RBM when the Mean Square Error (MSE) is inferior to this value.
#ifdef _CONSOLE
		void Train(int epochs, int cd, cudafloat errorStop) {
#else
		void Train(int epochs, int cd, cudafloat errorStop, ProgressInfo * progress = nullptr, QString * progressLog = nullptr) {
#endif
			int nlayers = (int)rbms.Length();

			for (int l = 0; l < nlayers; l++) {
#ifndef _CONSOLE
				if (progressLog != nullptr && progress != nullptr) {
					QString info = progressLog->arg(l + 1).arg(nlayers).arg(0).arg("-");
					progress->SetValue(epochs == 0 ? l : l * epochs, info);
				}
#endif

				RBMhost * layer = rbms[l];
				if (l > 0) layer->RandomizeWeights();

				for (int e = 0; epochs == 0 || e < epochs; e++) {
					layer->ContrastiveDivergence(cd);
					cudafloat  error = layer->MeanSquareError();
					if (error < errorStop) break;

#ifndef _CONSOLE
					if (progress != nullptr && progress->NeedsUpdating()) {
						if (progressLog != nullptr) {
							QString info = progressLog->arg(l + 1).arg(nlayers).arg(e).arg(error);
							progress->SetValue(epochs == 0 ? l : l * epochs + e, info);
						}
					}
#endif
				}
				layer->ComputeStatusHiddenUnits(*(layer->v), layer->h_data);
			}
		}
		};

	//! \example DBNapp.cpp
	//! Example of the DBN and RBM algorithms usage.

	//! @}

	}

#endif //GPUMLIB_DBN_HOST_H
