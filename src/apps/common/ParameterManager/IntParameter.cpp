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

#include "IntParameter.h"
#include "qteditorfactory.h"

#include "assert.h"

namespace GPUMLib {

	QtIntPropertyManager * IntParameter::propManager = nullptr;

	IntParameter::IntParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, const QString & argumentName, bool optional, const QString & min, const QString & max)
		: NumericParameter<int>(fullname, name, summaryDescrition, argumentName, optional) {
		this->minimum = (min.isEmpty()) ? INT_MIN : min.toInt();
		this->maximum = (max.isEmpty()) ? INT_MAX : max.toInt();
		assert(this->minimum < this->maximum);
	}

	void IntParameter::SetValue(const QString & newValue) {
		NumericParameter<int>::SetValue(newValue.toInt());
		if (property != nullptr) propManager->setValue(property, value);
	}

	void IntParameter::InitializeGUIproperty() {
		propManager->setRange(property, minimum, maximum);
		if (hasValue) propManager->setValue(property, value);
	}

	QtAbstractPropertyManager * IntParameter::GetGUIpropertyManager(QtTreePropertyBrowser * browser) {
		if (propManager == nullptr) {
			propManager = new QtIntPropertyManager(browser);
			browser->setFactoryForManager(propManager, new QtSpinBoxFactory(browser));
		}

		return propManager;
	}

} // namespace GPUMLib