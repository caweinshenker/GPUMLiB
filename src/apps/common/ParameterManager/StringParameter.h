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

#ifndef GPUMLIB_STRING_PARAMETER_H
#define GPUMLIB_STRING_PARAMETER_H

#include "Parameter.h"
#include "qtpropertymanager.h"

namespace GPUMLib {

	class StringParameter : public Parameter {
	public:
		StringParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, const QString & argumentName, bool optional, const QString & validationExpression);

		virtual bool IsValid(LogHTML * log) override;

		virtual void SetValue(const QString & newValue) override;

		virtual void GetParametersValues(QMap<QString, QString> & parametersValues) const override;

	private:
		virtual void InitializeGUIproperty();

		virtual QtAbstractPropertyManager * GetGUIpropertyManager(QtTreePropertyBrowser * browser) override;

		static QtStringPropertyManager * propManager;

		QString validationExpression;
		QString value;
	};

} // namespace GPUMLib

#endif // GPUMLIB_STRING_PARAMETER_H
