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

#ifndef GPUMLIB_BASE_PARAMETER_H
#define GPUMLIB_BASE_PARAMETER_H

#include "../log/LogHTML.h"

#include <QtTreePropertyBrowser>
#include <QString>
#include <QMap>

namespace GPUMLib {

	class BaseParameter {
	public:
		friend class GroupParameter;

		bool IsOptional() const {
			return optional;
		}

		QString GetFullName() const {
			return fullname;
		}

		virtual bool IsValid(LogHTML * log) = 0;

		virtual void AddToGUIpropertyBrowser(QtTreePropertyBrowser * browser) = 0;

		virtual void GetParametersValues(QMap<QString, QString> & parametersValues) const = 0;

		virtual ~BaseParameter() {}

	protected:
		BaseParameter() {}

		BaseParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, bool optional);

		void CreateGUIproperty(QtTreePropertyBrowser * browser);

		virtual QtAbstractPropertyManager * GetGUIpropertyManager(QtTreePropertyBrowser * browser) = 0;
		virtual void InitializeGUIproperty() = 0;

		QtProperty * property;
		QString fullname;
		QString name;
		QString summary;
		bool optional;
	};

} // namespace GPUMLib

#endif // GPUMLIB_BASE_PARAMETER_H
