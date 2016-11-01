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

#ifndef GPUMLIB_ATS_WIDGET_H
#define GPUMLIB_ATS_WIDGET_H

#include "../common/widget/AlgorithmWidget.h"
#include "../../algorithms/mbp/multiple_back_propagation.h"

namespace GPUMLib {

	class ATSwidget : public AlgorithmWidget {
		Q_OBJECT

	public:
		explicit ATSwidget(const char * parameterFile, int argc, char ** argv, QWidget *parent = 0) : AlgorithmWidget(parameterFile, argc, argv, parent) {}

	private:
		virtual void Run(ParameterValues & parameterValues, LogHTML & summaryLog, LogHTML & log) override;
		virtual void LogConfiguration(LogHTML & log, ParameterValues & parameterValues) override;

		void SaveMBPNetwork(const QString & filename, GPUMLib::BackPropagation * network, const QString & trainfile, const QString & validationfile, double rmsStop, int maxEpochs, bool mbp);
		void SaveSelectiveInputWeights(QTextStream & fs, GPUMLib::HostArray<cudafloat> & weights, GPUMLib::HostArray<cudafloat> & bias);
		void SaveWeights(QTextStream & f, GPUMLib::HostArray<cudafloat> & weights);

	};

} // namespace GPUMLib

#endif // GPUMLIB_ATS_WIDGET_H
