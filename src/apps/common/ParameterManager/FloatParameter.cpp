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

#include <cmath>
#include <cfloat>

#include "FloatParameter.h"
#include "qteditorfactory.h"

#include "assert.h"

namespace GPUMLib {

	QtDoublePropertyManager * FloatParameter::propManager = nullptr;

	FloatParameter::FloatParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, const QString & argumentName, bool optional, const QString & min, const QString & max, const QString & decimals, const QString & step)
		: NumericParameter<float>(fullname, name, summaryDescrition, argumentName, optional) {
		this->minimum = (min.isEmpty()) ? -FLT_MAX : min.toFloat();
		this->maximum = (max.isEmpty()) ? FLT_MAX : max.toFloat();
		assert(this->minimum < this->maximum);

		this->decimals = (decimals.isEmpty()) ? 2 : decimals.toInt();

		this->step = (step.isEmpty()) ? 1.0f / powf(10.0f, this->decimals) : step.toFloat();
	}

	void FloatParameter::SetValue(const QString & newValue) {
		NumericParameter<float>::SetValue(newValue.toFloat());
		if (property != nullptr) propManager->setValue(property, value);
	}

	void FloatParameter::InitializeGUIproperty() {
		propManager->setRange(property, minimum, maximum);
		propManager->setDecimals(property, decimals);
		propManager->setSingleStep(property, step);
		if (hasValue) propManager->setValue(property, value);
	}

	QtAbstractPropertyManager * FloatParameter::GetGUIpropertyManager(QtTreePropertyBrowser * browser) {
		if (propManager == nullptr) {
			propManager = new QtDoublePropertyManager(browser);
			browser->setFactoryForManager(propManager, new QtDoubleSpinBoxFactory(browser));
		}

		return propManager;
	}

} // namespace GPUMLib