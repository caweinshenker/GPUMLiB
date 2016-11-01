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

#include "Parameter.h"

namespace GPUMLib {

	bool Parameter::IsValid(LogHTML * log) {
		if (property != nullptr) hasValue = property->hasValue();

		if (!hasValue) LogMustSpecifyParameter(log);

		return hasValue;
	}

	void Parameter::AddToGUIpropertyBrowser(QtTreePropertyBrowser * browser) {
		assert(browser != nullptr);
		CreateGUIproperty(browser);
		browser->addProperty(property);
	}

	Parameter::Parameter(const QString & fullname, const QString & name, const QString & summaryDescrition, const QString & argumentName, bool optional)
		: BaseParameter(fullname, name, summaryDescrition, optional) {
		this->argument = argumentName;
		this->hasValue = false;
	}

	void Parameter::LogMustSpecifyParameter(LogHTML * log) {
		if (log != nullptr) log->AppendLine(QString(QObject::tr("Please specify the %1").arg(fullname)));
	}

} //namespace GPUMLib