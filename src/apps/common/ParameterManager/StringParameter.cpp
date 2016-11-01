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

#include "StringParameter.h"
#include "qtpropertymanager.h"
#include "qteditorfactory.h"

namespace GPUMLib {

	QtStringPropertyManager * StringParameter::propManager = nullptr;

	StringParameter::StringParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, const QString & argumentName, bool optional, const QString & validationExpression)
		: Parameter(fullname, name, summaryDescrition, argumentName, optional) {
		this->validationExpression = validationExpression;
	}

	bool StringParameter::IsValid(LogHTML * log) {
		if (property != nullptr) {
			this->hasValue = property->hasValue();
			if (this->hasValue) value = property->valueText();
		}

		if (!hasValue || value.isEmpty()) {
			LogMustSpecifyParameter(log);
			return false;
		}

		return true;
	}

	void StringParameter::SetValue(const QString & newValue) {
		value = newValue;
		if (property != nullptr) propManager->setValue(property, value);
		hasValue = true;
	}

	void StringParameter::GetParametersValues(QMap<QString, QString> & parametersValues) const {
		if (property != nullptr) {
			if (property->hasValue()) {
				parametersValues.insert(argument, property->valueText());
			}
		}
		else if (hasValue) {
			parametersValues.insert(argument, value);
		}
	}

	void StringParameter::InitializeGUIproperty() {
		if (!validationExpression.isEmpty()) propManager->setRegExp(property, QRegExp(validationExpression));
		if (hasValue) propManager->setValue(property, value);
	}

	QtAbstractPropertyManager * StringParameter::GetGUIpropertyManager(QtTreePropertyBrowser * browser) {
		if (propManager == nullptr) {
			propManager = new QtStringPropertyManager(browser);
			browser->setFactoryForManager(propManager, new QtLineEditFactory(browser));
		}

		return propManager;
	}

} // namespace GPUMLib