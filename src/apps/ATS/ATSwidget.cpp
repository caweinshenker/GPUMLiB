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

#include "ATSwidget.h"

#include "../common/progress/ProgressInfo.h"
#include "../common/ConfusionMatrix.h"
#include "../common/dataset.h"

#include <QMessageBox>
#include <ctime>
#include <memory>

namespace GPUMLib {

	void ATSwidget::LogConfiguration(LogHTML & log, ParameterValues & parameterValues) {
		log.AppendSection("ATS configuration");

		log.BeginTable(0, 1);

		log.BeginRow();
		log.AddColumn("Algorithm");
		QString algorithm{ "Back-Propagation" };
		if (parameterValues["algorithm"] == "mbp") algorithm = "Multiple " + algorithm;
		log.AddColumn(algorithm);
		log.EndRow();

		log.BeginRow();
		log.AddColumn(parameterValues.GetBoolParameter("fixed") ? "Topology" : "Initial topology");
		log.AddColumn(parameterValues["topology"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Number of networks to train");
		log.AddColumn(parameterValues["networks"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Robust learning");
		log.AddColumn(parameterValues["robust"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Maximum number of epochs");
		log.AddColumn(parameterValues["epochs"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Stop when the Root Mean Square (RMS) error is lower than");
		log.AddColumn(parameterValues["rms"]);
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

		log.BeginRow();
		log.AddColumn("Validation filename");
		log.AddColumn(parameterValues["validationfile"]);
		log.EndRow();

		if (parameterValues.GetIntParameter("validationsamples") > 0) {
			log.BeginRow();
			log.AddColumn("Validation samples");
			log.AddColumn(parameterValues["validationsamples"]);
			log.EndRow();
		}

		log.EndTable();
	}

	void ATSwidget::Run(ParameterValues & parameterValues, LogHTML & summaryLog, LogHTML & log) {
		if (DeviceIsCPU()) {
			log.AppendLine("Currently, the ATS does not support training with the CPU. A CUDA capable GPU is required.");
			summaryLog.Append(log.ToString());
			return;
		}

		QStringList slayers = parameterValues["topology"].split('-', QString::SkipEmptyParts);

		int nLayers = slayers.length();
		int lastLayer = nLayers - 1;

		HostArray<int> layers(nLayers);

		QString algorithm = parameterValues["algorithm"];

		QString genericTopology = algorithm + ' ';
		for (int l = 0; l < nLayers; l++) {
			layers[l] = slayers[l].toInt();
			genericTopology += (l == 1 ? QString("%1") : slayers[l]);
			if (l != lastLayer) genericTopology += '-';
		}

		int nInputs = layers[0];
		int nOutputs = layers[lastLayer];

		bool hasHeader = parameterValues.GetBoolParameter("header");
		bool rescale = parameterValues.GetBoolParameter("rescale");

		int trainSamples = parameterValues.GetIntParameter("trainsamples");
		QString trainfile = parameterValues["trainfile"];

		int numberNetsTrain = parameterValues.GetIntParameter("networks");

		ProgressInfo progress(this, "ATS - Training networks", 0, numberNetsTrain);
		progress.Update("Loading datasets");

		std::unique_ptr<Dataset> dsTrain;

		try {
			dsTrain = std::move(std::unique_ptr<Dataset>(new Dataset(trainfile, hasHeader, rescale, nInputs, nOutputs, trainSamples, log)));
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

		try {
			dsValidation = std::move(std::unique_ptr<Dataset>(new Dataset(validationfile, hasHeader, rescale, nInputs, nOutputs, validationSamples, log, dsTrain->GetProperties())));
		} catch (QString & error) {
			QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the validation dataset. ") + error).exec();
			return;
		} catch (...) {
			QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the validation dataset: <i>%1</i>.").arg(validationfile)).exec();
			return;
		}

		HostArray<bool> selectiveNeurons(nLayers - 1);

		bool mbp = (algorithm == "mbp");

		selectiveNeurons[0] = mbp;
		for (size_t l = 1; l < selectiveNeurons.Length(); l++) selectiveNeurons[l] = false;

		HostArray<int> additionalSpaceLayers;

		HostMatrix<cudafloat> & inputs = dsTrain->GetInputs();
		HostMatrix<cudafloat> & targets = dsTrain->GetTargets();

		int bestNumberHiddenNeurons = layers[1];
		int currentNumberHiddenNeurons = bestNumberHiddenNeurons;
		bool down = false;

		validationSamples = (dsValidation == nullptr) ? 0 : dsValidation->NumberOfSamples();
		bool automated = (numberNetsTrain > 1 && dsValidation != nullptr && !parameterValues.GetBoolParameter("fixed"));

		int nClasses = (nOutputs == 1) ? 2 : nOutputs;

		double bestFMeasure = 0.0;
		int increment = 0;

		log.BeginTable();

		log.BeginRow();
		log.AddColumn("Network");
		log.AddColumn("Epoch");
		log.AddColumn("Time (s)");
		log.AddColumn("Train RMS (%)");
		log.AddColumn("F-Measure (%)");
		log.AddColumn("Accuracy (%)");
		log.AddColumn("Saved (filename)");
		log.EndRow();

		cudafloat rmsStop = parameterValues.GetDoubleParameter("rms");
		int maxEpochs = parameterValues.GetIntParameter("epochs");

		std::unique_ptr<BackPropagation> network;

		double sumFmeasures = 0.0;
		double sumAccuracy = 0.0;

		bool robust = parameterValues.GetBoolParameter("robust");

		QString bestTopology = genericTopology.arg(currentNumberHiddenNeurons);

		progress.Update("Training networks ...");

		int networksTrained = 0;
		for (; networksTrained < numberNetsTrain; networksTrained++) {
			if (progress.WasCanceled()) break;

			QString currentTopology = genericTopology.arg(currentNumberHiddenNeurons);

			if (mbp) {
				network = std::move(std::unique_ptr<BackPropagation>(new MultipleBackPropagation(layers, selectiveNeurons, additionalSpaceLayers, inputs, targets)));
			} else {
				network = std::move(std::unique_ptr<BackPropagation>(new BackPropagation(layers, inputs, targets)));
			}

			network->SetRobustLearning(robust);

			log.BeginRow();
			log.AddColumn(currentTopology);

			LogHTML progressInfo;

			progressInfo.BeginTable(0, 1);

			if (validationSamples > 0) {
				if (automated) {
					progressInfo.BeginRow();
					progressInfo.AddColumn("Best topology:");
					progressInfo.AddColumn(bestTopology);
					progressInfo.EndRow();
				}

				progressInfo.BeginRow();
				progressInfo.AddColumn("Best F-Measure (%):");
				progressInfo.AddPercentageColumn(bestFMeasure);
				progressInfo.EndRow();
			}

			progressInfo.BeginRow();
			progressInfo.AddColumn("current network:");
			progressInfo.AddColumn(QString("%1 of %2").arg(networksTrained + 1).arg(numberNetsTrain));
			progressInfo.EndRow();

			if (automated) {
				progressInfo.BeginRow();
				progressInfo.AddColumn("current topology:");
				progressInfo.AddColumn(currentTopology);
				progressInfo.EndRow();
			}

			progressInfo.BeginRow();
			progressInfo.AddColumn("epochs:");
			progressInfo.AddColumn("%1");
			progressInfo.EndRow();

			progressInfo.BeginRow();
			progressInfo.AddColumn("RMS (%):");
			progressInfo.AddColumn("%2");
			progressInfo.EndRow();

			progressInfo.EndTable();

			QString progressLog = progressInfo.ToString();

			QString currentProgress = progressLog.arg("-").arg("-");
			progress.SetValue(networksTrained, currentProgress);

			clock_t initialTime = clock();

			for (int e = 0; e < maxEpochs; e++) {
				if (progress.WasCanceled()) break;

				network->TrainOneEpoch();

				cudafloat currentRMS = network->GetRMSestimate();
				if (rmsStop != cudafloat(0.0) && currentRMS <= rmsStop) {
					currentRMS = network->GetRMS();
					if (currentRMS <= rmsStop) break;
				}

				if (progress.NeedsUpdating()) {
					QString rms = QString::number(currentRMS * 100.0, 'f', 2) + "%";
					currentProgress = progressLog.arg(e).arg(rms);
					progress.Update(currentProgress);
				}
			}

			cudaThreadSynchronize();
			unsigned time = (clock() - initialTime);

			log.AddColumn<int>(network->GetEpoch());
			log.AddColumn(QString("%1").arg((double)time / CLOCKS_PER_SEC));
			log.AddPercentageColumn(network->GetRMS());

			bool saveNetwork = true;

			ConfusionMatrix cm(nClasses);

			if (validationSamples > 0) {
				HostMatrix<cudafloat> outputs = network->GetOutputs(dsValidation->GetInputs());
				HostMatrix<cudafloat> & targets = dsValidation->GetTargets();

				for (int s = 0; s < validationSamples; s++) {
					int predicted = -1;
					int correct = -1;

					double max = -1.0;

					if (nOutputs > 1) {
						for (int o = 0; o < nOutputs; o++) {
							if (targets(s, o) >= CUDA_VALUE(0.5)) correct = o; // WARNING: Assuming single class outputs

							double output = outputs(s, o);
							if (output > max) {
								predicted = o;
								max = output; // WARNING: Assuming single class outputs
							}
						}
					} else {
						correct = (targets(s, 0) >= 0.5) ? 1 : 0;
						predicted = (outputs(s, 0) >= 0.5) ? 1 : 0;
					}

					cm.Classify(correct, predicted);
				}

				double fmeasure = cm.FMeasure();
				double accuracy = cm.Accuracy();

				sumFmeasures += fmeasure;
				sumAccuracy += accuracy;

				saveNetwork = (fmeasure >= bestFMeasure);

				log.AddPercentageColumn(fmeasure);
				log.AddPercentageColumn(accuracy);

				if (saveNetwork) {
					bestFMeasure = fmeasure;
					bestNumberHiddenNeurons = currentNumberHiddenNeurons;
					bestTopology = genericTopology.arg(bestNumberHiddenNeurons);
					increment++;

					if (bestFMeasure == 1.0) automated = false; // best topology was found
				}
			}

			if (automated) {
				if (currentNumberHiddenNeurons < bestNumberHiddenNeurons) {
					if (down) {
						down = false;
						if (increment > 1) increment--;
					}
				} else if (currentNumberHiddenNeurons > bestNumberHiddenNeurons) {
					if (!down) {
						down = true;
						if (increment > 1) increment--;
					}
				}

				currentNumberHiddenNeurons += increment * ((down) ? -1 : 1);
				if (currentNumberHiddenNeurons < 1) currentNumberHiddenNeurons = 1;

				if (down && currentNumberHiddenNeurons == 1) down = false;

				layers[1] = currentNumberHiddenNeurons;
			}

			if (saveNetwork) {
				QString networkFile = QString(algorithm + "%1.bpn").arg(randomSeed);
				SaveMBPNetwork(networkFile, network.get(), trainfile, validationfile, rmsStop, maxEpochs, mbp);
				log.AddColumn(networkFile);
			} else {
				log.AddEmptyColumn();
			}

			log.EndRow();

			srand(++randomSeed);
		}

		progress.End();

		log.EndTable();

		LogHTML resultsLog;

		resultsLog.BeginTable(0, 1);

		resultsLog.BeginRow();
		resultsLog.AddColumn("Networks trained:");
		resultsLog.AddColumn(networksTrained);
		resultsLog.EndRow();

		if (validationSamples > 0) {
			resultsLog.BeginRow();
			resultsLog.AddColumn("Average F-Measure:");
			resultsLog.AddPercentageColumn(sumFmeasures / networksTrained);
			resultsLog.EndRow();

			resultsLog.BeginRow();
			resultsLog.AddColumn("Average Accuracy:");
			resultsLog.AddPercentageColumn(sumAccuracy / networksTrained);
			resultsLog.EndRow();

			resultsLog.BeginRow();
			resultsLog.AddColumn("Best F-Measure:");
			resultsLog.AddPercentageColumn(bestFMeasure);
			resultsLog.EndRow();
		}

		resultsLog.EndTable();

		summaryLog.Append(resultsLog);
		log.Append(resultsLog);

		summaryLog.Append(log.ToString());
	}

	void ATSwidget::SaveMBPNetwork(const QString & filename, GPUMLib::BackPropagation * network, const QString & trainfile, const QString & validationfile, double rmsStop, int maxEpochs, bool mbp) {
		const char * MBP_VERSION = "Multiple Back-Propagation Version 2.2.1";

		QFile file(filename);
		file.open(QIODevice::WriteOnly | QIODevice::Text);

		QTextStream fs(&file);

		fs << MBP_VERSION << endl;
		fs << "Multiple Back-Propagation can be freely obtained at http://dit.ipg.pt/MBP - This file was generated by the ATS." << endl;
		fs << trainfile << endl;
		fs << validationfile << endl;

		fs << "32" << endl; // priority normal
		fs << "0" << endl; // update screen

		fs << "1" << endl; // delta bar delta
		fs << network->GetUpStepSizeFactor() << endl;
		fs << network->GetDownStepSizeFactor() << endl;
		fs << network->GetMaxStepSize() << endl;

		fs << network->GetRobustLearning() << endl;
		fs << network->GetRobustFactor() << endl;
		fs << 1.0 + network->GetMaxPercentageRMSGrow() << endl; // rmsGrowToApplyRobustLearning

		fs << "0.0" << endl; // weightDecay

		fs << "0" << endl; // autoUpdateLearning
		fs << "0" << endl; // autoUpdateMomentum

		fs << "0.01" << endl; //percentIncDecLearnRate
		fs << "0.01" << endl; //percentIncDecMomentum
		fs << "0.01" << endl; //percentIncDecSpaceLearnRate
		fs << "0.01" << endl; //percentIncDecSpaceMomentum

		fs << INITIAL_LEARNING_RATE << endl; //mainNetLearningMomentumInformation.learningRate.value
		fs << "1000" << endl; //mainNetLearningMomentumInformation.learningRate.decayEpochs
		fs << "1" << endl; //mainNetLearningMomentumInformation.learningRate.decayPercentage

		fs << network->GetMomentum() << endl; //mainNetLearningMomentumInformation.momentum.value
		fs << "0" << endl; //mainNetLearningMomentumInformation.momentum.decayEpochs
		fs << "0" << endl; //mainNetLearningMomentumInformation.momentum.decayPercentage

		fs << INITIAL_LEARNING_RATE << endl; //spaceNetLearningMomentumInformation.learningRate.value
		fs << "1000" << endl; //spaceNetLearningMomentumInformation.learningRate.decayEpochs
		fs << "1" << endl; //spaceNetLearningMomentumInformation.learningRate.decayPercentage

		fs << network->GetMomentum() << endl; //spaceNetLearningMomentumInformation.momentum.value
		fs << "0" << endl; //spaceNetLearningMomentumInformation.momentum.decayEpochs
		fs << "0" << endl; //spaceNetLearningMomentumInformation.momentum.decayPercentage

		fs << "0" << endl; //epochsStop
		fs << rmsStop << endl;
		fs << maxEpochs << endl; // numberEpochsToStop

		fs << "0.0" << endl; //spaceRmsStop

		fs << "1" << endl; //batchTraining
		fs << "0" << endl; //randomizePatterns

		fs << (mbp ? 3 : 0) << endl; //Network Type

		//main network
		fs << network->GetNumberInputs();

		int numLayers = network->GetNumberLayers();
		for (int l = 0; l < numLayers; l++) {
			fs << "-" << network->GetNumberNeurons(l);
		}
		fs << endl;

		//space network additional layers
		fs << endl;

		// layers information
		for (int l = 0; l < numLayers; l++) {
			int numNeurons = network->GetNumberNeurons(l);

			fs << ((l == 0 && mbp) ? numNeurons : 0) << endl; // NeuronsWithSelectiveActivation

			for (int n = 0; n < numNeurons; n++) {
				fs << "0" << endl; // ActivationFunction
				fs << "1.0" << endl; //ActivationFunctionParameter
			}
		}

		// space layers information
		if (mbp) {
			int numNeurons = network->GetNumberNeurons(0);
			for (int n = 0; n < numNeurons; n++) {
				fs << "0" << endl; // ActivationFunction
				fs << "1.0" << endl; //ActivationFunctionParameter
			}
		}

		fs << "0" << endl; //ConnectInputLayerWithOutputLayer main
		fs << "0" << endl; //ConnectInputLayerWithOutputLayer space

		if (network->HasSelectiveInputs()) {
			HostArray<cudafloat> weights = network->GetSelectiveInputWeights();
			HostArray<cudafloat> bias = network->GetSelectiveInputBias();

			SaveSelectiveInputWeights(fs, weights, bias);
		}

		for (int l = 0; l < numLayers; l++) {
			HostArray<cudafloat> weights = network->GetLayerWeights(l);
			SaveWeights(fs, weights);
		}

		if (mbp) {
			MultipleBackPropagation * mbpnet = static_cast<MultipleBackPropagation *>(network);

			if (mbpnet->HasSelectiveInputs()) {
				HostArray<cudafloat> weights = mbpnet->GetSelectiveInputWeightsSpaceNetwork();
				HostArray<cudafloat> bias = mbpnet->GetSelectiveInputBiasSpaceNetwork();

				SaveSelectiveInputWeights(fs, weights, bias);
			}

			HostArray<cudafloat> weights = mbpnet->GetLayerWeightsSpaceNetwork(0);
			SaveWeights(fs, weights);
		}

		fs << "0" << endl; //epoch must be 0
		fs << "0" << endl; //rmsInterval
		fs << "0" << endl; //trainingTime
	}

	void ATSwidget::SaveSelectiveInputWeights(QTextStream & fs, GPUMLib::HostArray<cudafloat> & weights, GPUMLib::HostArray<cudafloat> & bias) {
		size_t numWeights = weights.Length();

		for (size_t w = 0; w < numWeights; w++) {
			if (weights[w] != CUDA_VALUE(0.0) || bias[w] != CUDA_VALUE(0.0)) {
				fs << bias[w] << endl;
				fs << "0.0" << endl; //delta
				fs << "0.0" << endl; //deltaWithoutLearningMomentum
				fs << INITIAL_LEARNING_RATE << endl;

				fs << weights[w] << endl;
				fs << "0.0" << endl; //delta
				fs << "0.0" << endl; //deltaWithoutLearningMomentum
				fs << INITIAL_LEARNING_RATE << endl;
			}
		}
	}

	void ATSwidget::SaveWeights(QTextStream & fs, GPUMLib::HostArray<cudafloat> & weights) {
		size_t numWeights = weights.Length();

		for (size_t w = 0; w < numWeights; w++) {
			fs << weights[w] << endl;
			fs << "0.0" << endl; //delta
			fs << "0.0" << endl; //deltaWithoutLearningMomentum
			fs << INITIAL_LEARNING_RATE << endl;
		}
	}

} // namespace GPUMLib
