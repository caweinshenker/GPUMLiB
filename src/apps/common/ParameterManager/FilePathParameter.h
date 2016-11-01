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

#ifndef GPUMLIB_FILE_PATH_PARAMETER_H
#define GPUMLIB_FILE_PATH_PARAMETER_H

#include "Parameter.h"
#include "../PropertyBrowserExtension/filepathmanager.h"

namespace GPUMLib {

	class FilePathParameter : public Parameter {
	public:
		FilePathParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, const QString & argumentName, bool optional, const QString & filter);

		virtual bool IsValid(LogHTML * log) override;

		virtual void SetValue(const QString & newValue) override;

		virtual void GetParametersValues(QMap<QString, QString> & parametersValues) const override;

	private:
		virtual void InitializeGUIproperty() override;

		virtual QtAbstractPropertyManager * GetGUIpropertyManager(QtTreePropertyBrowser * browser) override;

		static FilePathManager * propManager;

		QString value;
		QString filter;
	};

} // namespace GPUMLib

#endif // GPUMLIB_FILE_PATH_PARAMETER_H
