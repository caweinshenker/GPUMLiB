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

#include "../../../cuda/definitions.h"
#include "AlgorithmWidget.h"

#include <QMessageBox>
#include <QVBoxLayout>
#include <QPushButton>

#include <cuda_runtime.h>
#include <ctime>

namespace GPUMLib {

	AlgorithmWidget::AlgorithmWidget(const char * parameterFile, int argc, char ** argv, QWidget *parent, const char * runText) : QWidget(parent),
		parameterManager(parameterFile, argc, argv) {

		QPushButton * buttonRun = new QPushButton();
		buttonRun->setText(QObject::tr(runText));
		connect(buttonRun, SIGNAL(clicked()), SLOT(StartRunning()));

		QHBoxLayout * layoutButtons = new QHBoxLayout();
		layoutButtons->addWidget(buttonRun);
		layoutButtons->addStretch();

		QVBoxLayout * layout = new QVBoxLayout(this);
		layout->addWidget(parameterManager.GetGUI(), 1);
		layout->addLayout(layoutButtons);

		setWindowTitle(QString("GPUMLib %1 - %2").arg(GPUMLIB_VERSION).arg(parameterManager.GetApplicationName()));
		setWindowState(Qt::WindowMaximized);
		show();

		deviceIsValid = false;
	}

	void AlgorithmWidget::LogSystemInfo(LogHTML & log) {
		log.AppendSection("System configuration");

		log.BeginTable(0, 1);

		if (device < 0) {
			log.BeginRow();
			log.AddColumn("Device");
			log.AddColumn("CPU");
			log.EndRow();
		} else {
			cudaDeviceProp deviceProperties;
			cudaGetDeviceProperties(&deviceProperties, device);

			log.BeginRow();
			log.AddColumn("Device");
			log.AddColumn(QString("%1 (%2 Mhz) - supports CUDA %3.%4").arg(deviceProperties.name).arg(deviceProperties.clockRate / 1000).arg(deviceProperties.major).arg(deviceProperties.minor));
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Multi-Processors");
			log.AddColumn(deviceProperties.multiProcessorCount);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Global Memory");
			log.AddColumn(deviceProperties.totalGlobalMem);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Const Memory");
			log.AddColumn(deviceProperties.totalConstMem);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Shared Memory per Block");
			log.AddColumn(deviceProperties.sharedMemPerBlock);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Registers per Block");
			log.AddColumn(deviceProperties.regsPerBlock);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Maximum Threads per Block");
			log.AddColumn(deviceProperties.maxThreadsPerBlock);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Maximum Threads Dim");
			log.AddColumn(QString("(%1, %2, %3)").arg(deviceProperties.maxThreadsDim[0]).arg(deviceProperties.maxThreadsDim[1]).arg(deviceProperties.maxThreadsDim[2]));
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Maximum Grid Size");
			log.AddColumn(QString("(%1, %2, %3)").arg(deviceProperties.maxGridSize[0]).arg(deviceProperties.maxGridSize[1]).arg(deviceProperties.maxGridSize[2]));
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Warp Size");
			log.AddColumn(deviceProperties.warpSize);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Mem Pitch");
			log.AddColumn(deviceProperties.memPitch);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Texture Alignment");
			log.AddColumn(deviceProperties.textureAlignment);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Device Overlap");
			log.AddColumn(deviceProperties.deviceOverlap);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Kernel Timeout Enabled");
			log.AddColumn(deviceProperties.kernelExecTimeoutEnabled);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Device integrated");
			log.AddColumn(deviceProperties.integrated);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Can Map Host Memory");
			log.AddColumn(deviceProperties.canMapHostMemory);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Compute Mode");
			log.AddColumn(deviceProperties.computeMode);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Size of Floating Type");
			log.AddColumn(sizeof(cudafloat));
			log.EndRow();
		}

		log.BeginRow();
		log.AddColumn("Initial Random Seed");
		log.AddColumn(randomSeed);
		log.EndRow();

		log.EndTable();
	}


	void AlgorithmWidget::StartRunning() {
		LogHTML summaryLog;

		if (!parameterManager.ParametersAreValid(&summaryLog)) {
			QMessageBox(QMessageBox::Warning, "Invalid parameters", summaryLog.ToString()).exec();
			return;
		}

		ParameterValues parameterValues;
		parameterManager.GetParameterValues(parameterValues);

		randomSeed = (time_t)(parameterValues.Contains("random") ? parameterValues.GetIntParameter("random") : 0);
		if (randomSeed == 0) randomSeed = time(0);
		srand(randomSeed);

		LogHTML log(parameterManager.GetApplicationName(), QString("log_%1_%2.html").arg(parameterManager.GetApplicationAcronym().toLower()).arg(randomSeed));

		try {
			if (parameterValues.Contains("device")) {
				device = parameterValues.GetIntParameter("device");

				deviceIsValid = (device < 0 || cudaSetDevice(device) == cudaSuccess);
				if (!deviceIsValid) {
					QMessageBox(QMessageBox::Warning, QObject::tr("Problem initializing device"), cudaGetErrorString(cudaGetLastError())).exec();
					return;
				}
			} else {
				device = -1;
				deviceIsValid = true;
			}

			summaryLog.AppendGPUMLibHeader(parameterManager.GetApplicationName());

			this->LogSystemInfo(log);
			this->LogConfiguration(log, parameterValues);

			log.AppendSection("Results");

			this->Run(parameterValues, summaryLog, log);
			QMessageBox(QMessageBox::Information, QObject::tr("Results"), summaryLog.ToString()).exec();
		} catch (...) {
			QMessageBox(QMessageBox::Warning, QObject::tr("Error"), QString("<span style='color:red'><b>An error has occurred. %1</b></span>").arg(log.ToString())).exec();
		}
	}

} // namespace GPUMLib
