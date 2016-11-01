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

#include <cstdlib>
#include <stdexcept>

#include "../cuda/definitions.h"
#include "../common/Utilities.h"
#include "../cuda/random/random.h"
#include "../cuda/rbm/rbm_kernels.h"
#include "RBM.h"

namespace GPUMLib {

	void RBM::RandomizeWeights() {
		int nWeights = (int)w.Elements();

		cudafloat * weights = w.HostPointer();

		for (int i = 0; i < nWeights; i++) weights[i] = CUDA_VALUE(2.0) * stdWeights * ((cudafloat)rand() / RAND_MAX) - stdWeights;
		w.UpdateDevice();

		int blockSize = NumberThreadsPerBlockThatBestFit(nWeights);
		int blocks = NumberBlocks(nWeights, blockSize);

		cudaError_t error = gpumlib_cuda_rbm_init_bias_and_deltas(blocks, blockSize, b.DevicePointer(), INITIAL_BIAS_HIDDEN_UNITS, lastDelta.w.Pointer(), lastDelta.b.Pointer(), lastDeltaWithoutLearningMomentum.w.Pointer(), lastDeltaWithoutLearningMomentum.b.Pointer(), learningRate.w.Pointer(), learningRate.b.Pointer(), initialLearningRate, nWeights, J);
		if (error != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(error));
		}

		blocks = NumberBlocks(I, inputsBlockSize);

		error = gpumlib_cuda_rbm_init_input_bias_and_deltas(blocks, inputsBlockSize, v.Pointer(), a.DevicePointer(), lastDelta.a.Pointer(), lastDeltaWithoutLearningMomentum.a.Pointer(), learningRate.a.Pointer(), initialLearningRate, I, samples);
		if (error != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(error));
		}

		epoch = 0;
	}

	void RBM::ComputeStatusUnits(cudafloat * v, cudafloat * h, cudafloat * v_reconstructed, int samples, float * rnd) {
		size_t connections = w.Elements();

		dim3 dimSamplesJ;
		dimSamplesJ.y = J;

		if (connections > CUDA_MAX_THREADS_PER_BLOCK) {
			int processed = 0;
			do {
				int samplesToProcess = (samples > CUDA_MAX_GRID_X_DIM) ? CUDA_MAX_GRID_X_DIM : samples;
				dimSamplesJ.x = samplesToProcess;
				cudaError_t error = gpumlib_cuda_rbm_compute_status_hidden_units(dimSamplesJ, inputsBlockSize, v + (processed * I), w.DevicePointer(), b.DevicePointer(), h + (processed * J), rnd + (J * cd * processed), I);

				if (error != cudaSuccess) {
					throw std::runtime_error(cudaGetErrorString(error));
				}

				processed += samplesToProcess;
			} while (processed < samples);
		} else {
			cudaError_t error = gpumlib_cuda_rbm_compute_status_hidden_units_small(samples, dimIJ, (unsigned)(connections * sizeof(cudafloat)), v, w.DevicePointer(), b.DevicePointer(), h, rnd);

			if (error != cudaSuccess) {
				throw std::runtime_error(cudaGetErrorString(error));
			}
		}

		dim3 dimSamplesI;
		dimSamplesI.y = I;

		if (v_reconstructed != nullptr) {
			rnd = (useBinaryValuesVisibleReconstruction) ? (rnd + (J  * cd * samples)) : nullptr;

			if (connections > CUDA_MAX_THREADS_PER_BLOCK) {
				int processed = 0;
				do {
					int samplesToProcess = (samples > CUDA_MAX_GRID_X_DIM) ? CUDA_MAX_GRID_X_DIM : samples;
					dimSamplesI.x = samplesToProcess;
					cudaError_t error = gpumlib_cuda_rbm_compute_status_visible_units(dimSamplesI, hiddenUnitsBlockSize, h + (processed * J), w.DevicePointer(), a.DevicePointer(), v_reconstructed + (processed * I), rnd + ((J + I) * cd * processed), J);

					if (error != cudaSuccess) {
						throw std::runtime_error(cudaGetErrorString(error));
					}

					processed += samplesToProcess;
				} while (processed < samples);
			} else {
				cudaError_t error = gpumlib_cuda_rbm_compute_status_visible_units_small(samples, dimJI, (unsigned)(connections * sizeof(cudafloat)), h, w.DevicePointer(), a.DevicePointer(), v_reconstructed, rnd);

				if (error != cudaSuccess) {
					throw std::runtime_error(cudaGetErrorString(error));
				}
			}
		}
	}

	void RBM::ContrastiveDivergence(int n) {
		int sizeLastBatch = samples;
		int batches = 1;

		if (miniBatchSize > 0) {
			batches = samples / miniBatchSize;
			sizeLastBatch = samples % miniBatchSize;
			if (sizeLastBatch > 0) {
				batches++;
			} else {
				sizeLastBatch = miniBatchSize;
			}
		}

		dim3 block;
		block.x = 16;
		block.y = 16;

		dim3 grid;
		grid.x = NumberBlocks(I, block.x);
		grid.y = NumberBlocks(J, block.y);

		cudafloat * vd = v.Pointer();
		cudafloat * hd = h_data.Pointer();
		cudafloat * vr = v_recon.Pointer();
		cudafloat * hr = h_recon.Pointer();
		cudafloat * cerrors = errors.Pointer();

		Random::Fill(randomValues);
		float * rnd = randomValues.Pointer();

		int lastBatch = batches - 1;
		for (int batch = 0; batch < batches; batch++) {
			int samples = (batch == lastBatch) ? sizeLastBatch : miniBatchSize;

			ComputeStatusUnits(vd, hd, vr, samples, rnd);
			rnd += samples * (useBinaryValuesVisibleReconstruction ? (I + J) : J);

			for (int k = 1; k < n; k++) {
				ComputeStatusUnits(vr, hr, vr, samples, rnd);
				rnd += samples * (useBinaryValuesVisibleReconstruction ? (I + J) : J);
			}

			ComputeStatusUnits(vr, hr, nullptr, samples, nullptr);

			cudaError_t error = gpumlib_cuda_rbm_correct_weights(grid, block, vd, hd, vr, hr, samples, learningRate.w.Pointer(), lastDeltaWithoutLearningMomentum.w.Pointer(), lastDelta.w.Pointer(), learningRate.b.Pointer(), lastDeltaWithoutLearningMomentum.b.Pointer(), lastDelta.b.Pointer(), learningRate.a.Pointer(), lastDeltaWithoutLearningMomentum.a.Pointer(), lastDelta.a.Pointer(), U_FACTOR, D_FACTOR, momentum, w.DevicePointer(), b.DevicePointer(), a.DevicePointer(), cerrors, I, J);

			if (error != cudaSuccess) {
				throw std::runtime_error(cudaGetErrorString(error));
			}

			vd += miniBatchSize;
			hd += miniBatchSize;
			vr += miniBatchSize;
			hr += miniBatchSize;
			cerrors += miniBatchSize;
		}

		epoch++;
	}

}
