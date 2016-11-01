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

#include "dbn_widget.h"

#include "../common/progress/ProgressInfo.h"
#include "../common/ConfusionMatrix.h"
#include "../common/dataset.h"

#include "../../DBN/HostDBN.h"
#include "../../DBN/DBN.h"

#include <QMessageBox>
#include <ctime>
#include <memory>

namespace GPUMLib {

	void DBNwidget::Save(OutputFile & f, float v) {
		f.Write("<float>");
		f.Write(((double)v));
		f.WriteLine("</float>");
	}

	void DBNwidget::SaveDBNheader(OutputFile & f) {
		f.WriteLine("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
		f.WriteLine("<DBNmodel xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">");
		f.WriteLine("<layers>");
	}

	void DBNwidget::SaveDBNfooter(OutputFile & f, const QString & trainFilename, const QString & testFilename) {
		f.WriteLine("</layers>");

		f.Write("<TrainFilename>");
		f.Write(trainFilename.toLatin1().data());
		f.Write("</TrainFilename>");

		f.Write("<TestFilename>");
		f.Write(testFilename.toLatin1().data());
		f.Write("</TestFilename>");

		f.WriteLine("</DBNmodel>");
	}

	void DBNwidget::SaveDBNlayer(OutputFile & f, HostMatrix<cudafloat> & weights, HostArray<cudafloat> & a, HostArray<cudafloat> & b) {
		int I = (int)weights.Columns();
		int J = (int)weights.Rows();

		f.WriteLine("<RBMlayer>");

		f.WriteLine("<weights>");
		for (int j = 0; j < J; j++) for (int i = 0; i < I; i++) Save(f, weights(j, i));
		f.WriteLine("</weights>");

		f.WriteLine("<biasVisibleLayer>");
		for (int i = 0; i < I; i++) Save(f, a[i]);
		f.WriteLine("</biasVisibleLayer>");

		f.WriteLine("<biasHiddenLayer>");
		for (int j = 0; j < J; j++) Save(f, b[j]);
		f.WriteLine("</biasHiddenLayer>");

		f.WriteLine("</RBMlayer>");
	}

	void DBNwidget::SaveDBN(DBNhost & network, time_t randomSeed, const QString & trainFilename, const QString & testFilename) {
		ostringstream sstream;
		sstream << randomSeed << ".dbn";

		OutputFile f(sstream.str().c_str());
		string s;

		SaveDBNheader(f);

		int nLayers = network.GetNumberRBMs();
		for (int l = 0; l < nLayers; l++) {
			RBMhost * layer = network.GetRBM(l);

			HostMatrix<cudafloat> w = layer->GetWeights();
			HostArray<cudafloat> a = layer->GetVisibleBias();
			HostArray<cudafloat> b = layer->GetHiddenBias();

			SaveDBNlayer(f, w, a, b);
		}

		SaveDBNfooter(f, trainFilename, testFilename);
	}

	void DBNwidget::SaveDBN(DBN & network, time_t randomSeed, const QString & trainFilename, const QString & testFilename) {
		ostringstream sstream;
		sstream << randomSeed << ".dbn";

		OutputFile f(sstream.str().c_str());
		string s;

		SaveDBNheader(f);

		int nLayers = network.GetNumberRBMs();
		for (int l = 0; l < nLayers; l++) {
			RBM * layer = network.GetRBM(l);

			HostMatrix<cudafloat> w = layer->GetWeights();
			HostArray<cudafloat> a = layer->GetVisibleBias();
			HostArray<cudafloat> b = layer->GetHiddenBias();

			SaveDBNlayer(f, w, a, b);
		}

		SaveDBNfooter(f, trainFilename, testFilename);
	}

	void DBNwidget::LogConfiguration(LogHTML & log, ParameterValues & parameterValues) {
		log.AppendSection("DBN configuration");
		log.BeginTable(0, 1);

		log.BeginRow();
		log.AddColumn("Topology");
		log.AddColumn(parameterValues["topology"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Classification");
		log.AddColumn(parameterValues["classification"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Algorithm");
		log.AddColumn(QString("CD-%1").arg(parameterValues["cd"]));
		log.EndRow();

		if (parameterValues.GetIntParameter("minibatch") > 0) {
			log.BeginRow();
			log.AddColumn("Mini-batch size");
			log.AddColumn(parameterValues["minibatch"]);
			log.EndRow();
		}

		log.BeginRow();
		log.AddColumn("Initial Learning Rate");
		log.AddColumn(parameterValues["learning_rate"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Initial Momentum");
		log.AddColumn(parameterValues["momentum"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Maximum number of epochs");
		log.AddColumn(parameterValues["epochs"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Stop when the Mean Square Error (MSE) is lower than");
		log.AddColumn(parameterValues["mse"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("The datasets contain an header line");
		log.AddColumn(parameterValues["header"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Rescale data");
		log.AddColumn(parameterValues["rescale"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Training filename");
		log.AddColumn(parameterValues["trainfile"]);
		log.EndRow();

		if (parameterValues.GetIntParameter("trainsamples") > 0) {
			log.BeginRow();
			log.AddColumn("Training samples");
			log.AddColumn(parameterValues["trainsamples"]);
			log.EndRow();
		}

		if (parameterValues["validationfile"] != "") {
			log.BeginRow();
			log.AddColumn("Validation filename");
			log.AddColumn(parameterValues["validationfile"]);
			log.EndRow();
		}

		if (parameterValues.GetIntParameter("validationsamples") > 0) {
			log.BeginRow();
			log.AddColumn("Validation samples");
			log.AddColumn(parameterValues["validationsamples"]);
			log.EndRow();
		}

		log.EndTable();
	}

	void DBNwidget::Run(ParameterValues & parameterValues, LogHTML & summaryLog, LogHTML & log) {
		QString topology = parameterValues["topology"];
		QStringList slayers = topology.split('-', QString::SkipEmptyParts);

		int numberLayers = slayers.length();
		int lastLayer = numberLayers - 1;

		HostArray<int> layers(numberLayers);

		for (int l = 0; l < numberLayers; l++) {
			layers[l] = slayers[l].toInt();
		}

		int nInputs = layers[0];
		int nOutputs = layers[lastLayer];

		bool classification = parameterValues.GetBoolParameter("classification");
		if (classification) {
			numberLayers--;
		} else {
			nOutputs = 0;
		}

		if (numberLayers < 2) {
			QMessageBox(QMessageBox::Warning, "Warning", QString("Invalid topology. The network must have at least 2 layers (3 if classification is used).")).exec();
			return;
		}

		bool hasHeader = parameterValues.GetBoolParameter("header");
		bool rescale = parameterValues.GetBoolParameter("rescale");

		int trainSamples = parameterValues.GetIntParameter("trainsamples");
		QString trainfile = parameterValues["trainfile"];

		int maxEpochs = parameterValues.GetIntParameter("epochs");
		cudafloat mseStop = parameterValues.GetDoubleParameter("mse");

		HostArray<int> dbnLayers(numberLayers);
		for (int l = 0; l < numberLayers; l++) dbnLayers[l] = layers[l];

		int numberRBMs = numberLayers - 1;

		int progressSize = numberRBMs;
		if (maxEpochs > 0) progressSize *= maxEpochs;

		ProgressInfo progress(this, "DBN - Training network", 0, progressSize);
		progress.Update("Loading datasets");

		std::unique_ptr<Dataset> dsTrain;

		try {
			if (rescale) {
				dsTrain = std::move(std::unique_ptr<Dataset>(new Dataset(trainfile, hasHeader, 0, 1, nInputs, nOutputs, trainSamples, log)));
			} else {
				dsTrain = std::move(std::unique_ptr<Dataset>(new Dataset(trainfile, hasHeader, rescale, nInputs, nOutputs, trainSamples, log)));
			}
		} catch (QString & error) {
			QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the training dataset. ") + error).exec();
			return;
		} catch (...) {
			QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the training dataset: <i>%1</i>.").arg(trainfile)).exec();
			return;
		}

		int validationSamples = parameterValues.GetIntParameter("validationsamples");
		QString validationfile = parameterValues["validationfile"];		

		std::unique_ptr<Dataset> dsValidation;

		if (!validationfile.isEmpty()) {
			try {
				if (rescale) {
					dsValidation = std::move(std::unique_ptr<Dataset>(new Dataset(validationfile, hasHeader, 0, 1, nInputs, nOutputs, validationSamples, log, dsTrain->GetProperties())));
				} else {
					dsValidation = std::move(std::unique_ptr<Dataset>(new Dataset(validationfile, hasHeader, rescale, nInputs, nOutputs, validationSamples, log, dsTrain->GetProperties())));
				}
			} catch (QString & error) {
				QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the validation dataset. ") + error).exec();
				return;
			} catch (...) {
				QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the validation dataset: <i>%1</i>.").arg(validationfile)).exec();
				return;
			}
		}

		double learningRate = parameterValues.GetDoubleParameter("learning_rate");
		double momentum = parameterValues.GetDoubleParameter("momentum");
		int cd = parameterValues.GetIntParameter("cd");
		int minibatch = parameterValues.GetIntParameter("minibatch");

		if (minibatch > 0 && DeviceIsCPU()) {
			log.AppendParagraph("Currently mini-batch is only supported by the GPU version. Training will proceed ignoring this option.");
		}

		progress.Update("Training network ...");

		log.BeginTable();

		log.BeginRow();
		log.AddColumn("Layer");
		log.AddColumn("MSE");
		log.AddColumn("Epoch");
		log.AddColumn("Time (s)");
		log.EndRow();

		LogHTML progressInfo;

		progressInfo.AppendParagraph("Training network");

		progressInfo.BeginTable(0, 1);

		progressInfo.BeginRow();
		progressInfo.AddColumn("Layer");
		progressInfo.AddColumn("%1 of %2");
		progressInfo.EndRow();

		progressInfo.BeginRow();
		progressInfo.AddColumn("Epochs (layer)");
		progressInfo.AddColumn("%3");
		progressInfo.EndRow();

		progressInfo.BeginRow();
		progressInfo.AddColumn("Layer MSE (%)");
		progressInfo.AddColumn("%4");
		progressInfo.EndRow();

		progressInfo.EndTable();

		QString progressLog = progressInfo.ToString();

		clock_t initialTime;
		unsigned time;

		if (DeviceIsGPU()) {
			DBN dbn(dbnLayers, dsTrain->GetInputs(), learningRate, momentum);

			initialTime = clock();

			if (dbn.Train(maxEpochs, cd, minibatch, mseStop, &progress, &progressLog)) {
				cudaThreadSynchronize();
				time = (clock() - initialTime);

				for (int l = 0; l < numberRBMs; l++) {
					RBM * layer = dbn.GetRBM(l);

					log.BeginRow();
					log.AddColumn(l + 1);
					log.AddColumn(layer->GetMSE());
					log.AddColumn(layer->Epoch());
					log.AddColumn((double)time / CLOCKS_PER_SEC);
					log.EndRow();
				}

				SaveDBN(dbn, randomSeed, trainfile, validationfile);
			} else {
				log.AppendParagraph("Could not train the network - Insuficient device (GPU) memory");
				summaryLog.AppendParagraph("Could not train the network. Insuficient device (GPU) memory.");
			}
		} else {
			DBNhost dbn(dbnLayers, dsTrain->GetInputs(), learningRate, momentum);

			initialTime = clock();
			dbn.Train(maxEpochs, cd, mseStop, &progress, &progressLog);
			time = (clock() - initialTime);

			for (int l = 0; l < numberRBMs; l++) {
				RBMhost * layer = dbn.GetRBM(l);

				log.BeginRow();
				log.AddColumn(l + 1);
				log.AddColumn(layer->MeanSquareError());
				log.AddColumn(layer->Epoch());
				log.AddColumn((double)time / CLOCKS_PER_SEC);
				log.EndRow();
			}

			SaveDBN(dbn, randomSeed, trainfile, validationfile);

			//if (cfg.Classification()) {
			//    cout << "Currently classfication is only supported by the GPU version." << endl;
			//}
		}

		progress.End();

		log.EndTable();

		summaryLog.Append(log.ToString());

		//cfg.maxEpochsClassification = 0;
		//cfg.rmsStopClassification = 0.01f;
		//cfg.learningRateClassification = 0.01f;
	}
} // namespace GPUMLib
