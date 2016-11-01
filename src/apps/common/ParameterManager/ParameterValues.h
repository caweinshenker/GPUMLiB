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

#ifndef GPUMLIB_PARAMETER_VALUES_H
#define GPUMLIB_PARAMETER_VALUES_H

#include <QMap>

namespace GPUMLib {

	class ParameterValues : public QMap < QString, QString > {
	public:
		QString GetStringParameter(const QString & name) const {
			return this->operator [](name);
		}

		bool GetBoolParameter(const QString & name) const {
			return GetStringParameter(name).compare("true", Qt::CaseInsensitive) == 0;
		}

		int GetIntParameter(const QString & name) const {
			return GetStringParameter(name).toInt();
		}

		double GetDoubleParameter(const QString & name) const {
			return GetStringParameter(name).toDouble();
		}

		bool Contains(const char * parameter) {
			return this->contains(QString(parameter));
		}
	};

} // namespace GPUMLib

#endif // GPUMLIB_PARAMETER_VALUES_H
