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

#ifndef GPUMLIB_ALGORITHM_WIDGET_H
#define GPUMLIB_ALGORITHM_WIDGET_H

#include "../ParameterManager/ParameterManager.h"

#include <QWidget>

namespace GPUMLib {
	class AlgorithmWidget : public QWidget {
		Q_OBJECT

	public:
		bool AutoRun() const {
			return parameterManager.AutoRun();
		}

	public slots :
		void StartRunning();

	protected:
		explicit AlgorithmWidget(const char * parameterFile, int argc, char ** argv, QWidget *parent = 0, const char * runText = "&Train");

		virtual void Run(ParameterValues & parameterValues, LogHTML & summaryLog, LogHTML & log) = 0;

		virtual void LogConfiguration(LogHTML & log, ParameterValues & parameterValues) = 0;

		void LogSystemInfo(LogHTML & log);

		bool DeviceIsCPU() const {
			return device < 0;
		}

		bool DeviceIsGPU() const {
			return device >= 0;
		}

		bool DeviceIsValid() const {
			return deviceIsValid;
		}

		time_t randomSeed;

	private:
		ParameterManager parameterManager;
		int device;
		bool deviceIsValid;		
	};
} // namespace GPUMLib

#endif // GPUMLIB_ALGORITHM_WIDGET_H
