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

#include "BoolParameter.h"
#include "qteditorfactory.h"

namespace GPUMLib {

	QtBoolPropertyManager * BoolParameter::propManager = nullptr;

	BoolParameter::BoolParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, const QString & argumentName, bool optional)
		: Parameter(fullname, name, summaryDescrition, argumentName, optional) {
	}

	void BoolParameter::SetValue(const QString & newValue) {
		value = (newValue.compare("true", Qt::CaseInsensitive) == 0) ? true : false;
		if (property != nullptr) propManager->setValue(property, value);
		hasValue = true;
	}

	void BoolParameter::GetParametersValues(QMap<QString, QString> & parametersValues) const {
		if (property != nullptr) {
			if (property->hasValue()) {
				parametersValues.insert(argument, property->valueText());
			}
		}
		else if (hasValue) {
			parametersValues.insert(argument, (value) ? "True" : "False");
		}
	}

	void BoolParameter::InitializeGUIproperty() {
		if (hasValue) propManager->setValue(property, value);
	}

	QtAbstractPropertyManager * BoolParameter::GetGUIpropertyManager(QtTreePropertyBrowser * browser) {
		if (propManager == nullptr) {
			propManager = new QtBoolPropertyManager(browser);
			browser->setFactoryForManager(propManager, new QtCheckBoxFactory(browser));
		}

		return propManager;
	}

} // namespace GPUMLib