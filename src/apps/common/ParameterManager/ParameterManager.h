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

#ifndef GPUMLIB_PARAMETER_MANAGER_H
#define GPUMLIB_PARAMETER_MANAGER_H

#include "GroupParameter.h"
#include "ParameterValues.h"
#include "../log/LogHTML.h"

#include <memory>
#include <QtTreePropertyBrowser>
#include <QtXml/QDomDocument>
#include <QScrollArea>
#include <QSplitter>
#include <QLabel>

namespace GPUMLib {

	class ParameterManager {
	public:
		ParameterManager(const char * parameterFile, int argc, char ** argv);

		~ParameterManager();

		QWidget * GetGUI();

		bool ParametersAreValid(LogHTML * log = nullptr) const;
		bool AutoRun() const;

		void GetParameterValues(ParameterValues & parametersValues);

		QString GetApplicationName() const;
		QString GetApplicationAcronym() const;

	private:
		void ProcessParameters(GroupParameter * group, QDomNode node, QMap<QString, QString> & arguments);

		void CreatePropertyBrowser();

		GroupParameter * mainGroupParameters;

		QtTreePropertyBrowser * propBrowser;

		QScrollArea * scrollLabel;
		QSplitter * splitter;
		QLabel * labelInfo;

		std::unique_ptr<LogHTML> info;

		QString appName;
		QString appAcronym;

		bool appSupportsCPU;
		bool appSupportsGPU;

		bool autorun;
	};

} //namespace GPUMLib

#endif // GPUMLIB_PARAMETER_MANAGER_H
