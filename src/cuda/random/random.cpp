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
#include "random.h"

namespace GPUMLib {

	curandGenerator_t Random::randomGenerator = nullptr;
	curandRngType_t Random::randomGeneratorType = CURAND_RNG_PSEUDO_DEFAULT;

	curandGenerator_t Random::RandomGenerator() {
		if (randomGenerator == nullptr) {
			curandCreateGenerator(&randomGenerator, randomGeneratorType);
			atexit(&CleanUp);
		}

		return randomGenerator;
	}

	void Random::CleanUp() {
		if (randomGenerator != nullptr) {
			curandDestroyGenerator(randomGenerator);
			randomGenerator = nullptr;
		}
	}

	void Random::SetSeed(unsigned long long seed, curandRngType_t generatorType) {
		if (generatorType != randomGeneratorType) {
			randomGeneratorType = generatorType;
			CleanUp();
		}

		curandSetPseudoRandomGeneratorSeed(RandomGenerator(), seed);
	}

	void Random::Fill(DeviceArray<float> & a) {
		curandGenerateUniform(RandomGenerator(), a.Pointer(), a.Length());
	}

}
