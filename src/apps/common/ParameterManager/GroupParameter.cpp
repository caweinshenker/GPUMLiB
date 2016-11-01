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

#include "GroupParameter.h"

namespace GPUMLib {

	QtGroupPropertyManager * GroupParameter::propManager = nullptr;

	GroupParameter::GroupParameter() {
		optional = false;
	}

	GroupParameter::GroupParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, bool optional)
		: BaseParameter(fullname, name, summaryDescrition, optional) {
	}

	void GroupParameter::AddParameter(BaseParameter * p) {
		assert(p != nullptr);
		parameters.push_back(p);
	}


	GroupParameter::~GroupParameter() {
		for (auto p = parameters.begin(); p != parameters.end(); ++p) {
			delete *p;
		}
	}

	bool GroupParameter::IsValid(LogHTML * log) {
		bool isValid = true;

		bool optionalParamSpecified = false;
		bool optionalParametersValid = true;

		LogHTML optionalLog;

		for (auto p = parameters.begin(); p != parameters.end(); ++p) {
			BaseParameter * param = *p;

			if (param->IsOptional()) {
				if (param->IsValid(&optionalLog)) {
				/*
					if (!optionalParametersValid) {
						isValid = false;
						if (log == nullptr) break;
					}
					optionalParamSpecified = true;
				} else {
					if (optionalParamSpecified) {
						isValid = false;
						if (log == nullptr) break;
					}
					optionalParametersValid = false;*/
				}
			} else {
				if (!param->IsValid(log)) {
					isValid = false;
					if (log == nullptr) break;
				}
			}
		}

		if (log != nullptr) log->Append(optionalLog.ToString());

		return isValid;
	}

	void GroupParameter::AddToGUIpropertyBrowser(QtTreePropertyBrowser * browser) {
		assert(browser != nullptr);

		if (!name.isEmpty()) {
			CreateGUIproperty(browser);
			browser->addProperty(property);

			for (auto p = parameters.begin(); p != parameters.end(); ++p) {
				AddSubproperty(browser, *p);
			}
		} else {
			for (auto p = parameters.begin(); p != parameters.end(); ++p) {
				BaseParameter * param = *p;
				param->AddToGUIpropertyBrowser(browser);
			}
		}
	}

	QtAbstractPropertyManager * GroupParameter::GetGUIpropertyManager(QtTreePropertyBrowser * browser) {
		if (propManager == nullptr) {
			propManager = new QtGroupPropertyManager(browser);
		}

		return propManager;
	}

	void GroupParameter::AddSubproperty(QtTreePropertyBrowser * browser, BaseParameter * subPropParam) {
		subPropParam->CreateGUIproperty(browser);

		GroupParameter * subgroup = dynamic_cast<GroupParameter *>(subPropParam);

		if (subgroup != nullptr) {
			for (auto p = subgroup->parameters.begin(); p != subgroup->parameters.end(); ++p) {
				subgroup->AddSubproperty(browser, *p);
			}
		}

		property->addSubProperty(subPropParam->property);
	}

	void GroupParameter::GetParametersValues(QMap<QString, QString> & parametersValues) const {
		for (auto p = parameters.begin(); p != parameters.end(); ++p) {
			BaseParameter * param = *p;
			param->GetParametersValues(parametersValues);
		}
	}

} //namespace GPUMLib