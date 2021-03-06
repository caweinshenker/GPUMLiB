/*
	Noel Lopes is a Professor at the Polytechnic of Guarda, Portugal
	and a researcher at the CISUC - University of Coimbra, Portugal
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

#include "multiple_back_propagation.h"

namespace GPUMLib {

	MultipleBackPropagation::MultipleBackPropagation(HostArray<int> & sizeLayers, HostArray<bool> & selectiveNeurons, HostArray<int> & sizeAdditionalSpaceLayers, HostMatrix<cudafloat> & trainInputPatterns, HostMatrix<cudafloat> & trainDesiredOutputPatterns, cudafloat initialLearningRate) {
		int processingLayers = (int)sizeLayers.Length() - 1;

		assert((int)selectiveNeurons.Length() == processingLayers);

		int outputsSpaceLayer = 0;
		for (int l = 0; l < processingLayers; l++) {
			if (selectiveNeurons[l]) outputsSpaceLayer += sizeLayers[l + 1];
		}

		assert(outputsSpaceLayer > 0);

		int additionalLayers = (int)sizeAdditionalSpaceLayers.Length();
		HostArray<int> sizeSpaceLayers(additionalLayers + 1);

		for (int l = 0; l < additionalLayers; l++) {
			assert(sizeAdditionalSpaceLayers[l] > 0);
			sizeSpaceLayers[l] = sizeAdditionalSpaceLayers[l];
		}
		sizeSpaceLayers[additionalLayers] = outputsSpaceLayer;

		CreateNetwork(sizeLayers, &sizeSpaceLayers, &selectiveNeurons, trainInputPatterns, trainDesiredOutputPatterns, initialLearningRate);

		layerHasSelectiveNeurons = selectiveNeurons;
	}

	bool MultipleBackPropagation::HasSelectiveNeurons(int layer) const {
		assert(layer >= 0 && layer < (int)layerHasSelectiveNeurons.Length());
		return layerHasSelectiveNeurons[layer];
	}

	int MultipleBackPropagation::GetNumberLayersSpaceNetwork() const {
		return (int)spaceLayers.Length();
	}

	int MultipleBackPropagation::GetNumberNeuronsSpaceNetwork(int layer) const {
		assert(layer >= 0 && layer < (int)spaceLayers.Length());
		return spaceLayers[layer].neurons;
	}

	HostArray<cudafloat> MultipleBackPropagation::GetLayerWeightsSpaceNetwork(int layer) {
		assert(layer >= 0 && layer < (int)spaceLayers.Length());
		return HostArray<cudafloat>(spaceLayers[layer].d_weights);
	}

	void MultipleBackPropagation::SetLayerWeightsSpaceNetwork(int layer, HostArray<cudafloat> & weights) {
		assert(layer >= 0 && layer < (int)spaceLayers.Length());
		spaceLayers[layer].d_weights = weights;
	}

	HostArray<cudafloat> MultipleBackPropagation::GetSelectiveInputWeightsSpaceNetwork() {
		return HostArray<cudafloat>(selectiveInputLayerSpaceNetwork->d_weights);
	}

	void MultipleBackPropagation::SetSelectiveInputWeightsSpaceNetwork(HostArray<cudafloat> & weights) {
		selectiveInputLayerSpaceNetwork->d_weights = weights;
	}

	HostArray<cudafloat> MultipleBackPropagation::GetSelectiveInputBiasSpaceNetwork() {
		return HostArray<cudafloat>(selectiveInputLayerSpaceNetwork->d_bias);
	}

	void MultipleBackPropagation::SetSelectiveInputBiasSpaceNetwork(HostArray<cudafloat> & bias) {
		selectiveInputLayerSpaceNetwork->d_bias = bias;
	}

}
