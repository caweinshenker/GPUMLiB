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

#include "ListParameter.h"
#include "qteditorfactory.h"

namespace GPUMLib {

	QtEnumPropertyManager * ListParameter::propManager = nullptr;

	ListParameter::ListParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, const QString & argumentName, bool optional, const QStringList & options, const QStringList & valuesOptions)
		: Parameter(fullname, name, summaryDescrition, argumentName, optional) {
		this->options = options;
		this->valuesOptions = valuesOptions;
	}

	void ListParameter::SetValue(const QString & newValue) {
		value = valuesOptions.indexOf(newValue);
		assert(value >= 0 && value < valuesOptions.size());
		if (property != nullptr) propManager->setValue(property, value);
		hasValue = true;
	}

	void ListParameter::GetParametersValues(QMap<QString, QString> & parametersValues) const {
		if (property != nullptr) {
			if (property->hasValue()) {
				QString option = property->valueText();

				for (int i = 0; i < options.length(); ++i) {
					if (options[i] == option) {
						parametersValues.insert(argument, valuesOptions[i]);
						break;
					}
				}
			}
		}
		else if (hasValue) {
			parametersValues.insert(argument, valuesOptions[value]);
		}
	}

	void ListParameter::InitializeGUIproperty() {
		propManager->setEnumNames(property, options);
		if (hasValue) propManager->setValue(property, value);
	}

	QtAbstractPropertyManager * ListParameter::GetGUIpropertyManager(QtTreePropertyBrowser * browser) {
		if (propManager == nullptr) {
			propManager = new QtEnumPropertyManager(browser);
			browser->setFactoryForManager(propManager, new QtEnumEditorFactory(browser));
		}

		return propManager;
	}

} // namespace GPUMLib