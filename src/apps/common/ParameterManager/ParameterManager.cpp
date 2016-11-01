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

#include "ParameterManager.h"
#include "StringParameter.h"
#include "BoolParameter.h"
#include "IntParameter.h"
#include "FloatParameter.h"
#include "ListParameter.h"
#include "FilePathParameter.h"

#include <stdexcept>
#include <assert.h>

#include <QRegularExpression>
#include <QMessageBox>
#include <QFile>

#include <cuda_runtime.h>

namespace GPUMLib {

	ParameterManager::ParameterManager(const char * parameterFile, int argc, char ** argv) {
		assert(parameterFile != nullptr && argv != nullptr);

		mainGroupParameters = nullptr;
		propBrowser = nullptr;

		QFile file(parameterFile);

		if (!file.open(QFile::ReadOnly)) throw std::invalid_argument("Could not load the parameters file");

		QDomDocument xmldoc;
		bool parsed = xmldoc.setContent(&file);
		file.close();

		QDomNode applicationNode;
		if (parsed) applicationNode = xmldoc.firstChild();

		if (applicationNode.isNull() || applicationNode.nodeName().toLower() != "application") throw std::invalid_argument("Invalid parameters specification. Check the parameters file");

		QDomNamedNodeMap attributes = applicationNode.attributes();

		appName = attributes.namedItem("name").nodeValue();
		appAcronym = attributes.namedItem("acronym").nodeValue();

		appSupportsCPU = (attributes.namedItem("cpu").nodeValue().compare("false", Qt::CaseInsensitive) != 0);
		appSupportsGPU = (attributes.namedItem("gpu").nodeValue().compare("false", Qt::CaseInsensitive) != 0);

		if (!appSupportsCPU && !appSupportsGPU) throw std::invalid_argument("Invalid parameters specification. The application should either support CPU or GPU computing (or both).");

		QDomNode firstNode;
		firstNode = applicationNode.firstChild();
		if (firstNode.isNull()) throw std::invalid_argument("Invalid parameters specification. Check the parameters file");

		autorun = false;

		QString invalidArguments;

		QMap<QString, QString> arguments;
		for (int p = 1; p < argc; ++p) {
			if (argv[p][0] != '-') {
				invalidArguments += QString(QObject::tr("Invalid parameter %1. Parameters must start by a '-' char.\n")).arg(argv[p]);
			} else {
				QString key = argv[p] + 1;

				if (key == "run") {
					autorun = true;
				} else {
					if (arguments.contains(key)) {
						invalidArguments += QString(QObject::tr("Parameter %1 specified more than once.\n")).arg(key);
					}

					if (++p < argc) {
						arguments[key] = argv[p];
					} else {
						invalidArguments += QString(QObject::tr("Parameter %1 specified, but no value was provided.\n")).arg(key);
					}
				}
			}
		}

		info = std::move(std::unique_ptr<LogHTML>(new LogHTML(appName)));

		QString appDescription = attributes.namedItem("summary").nodeValue();

		if (!appDescription.isEmpty()) {
			info->AppendSubSection(QObject::tr("Description"));
			info->AppendParagraphWithEscapedTags(appDescription);
		}

		info->AppendSubSection(QObject::tr("Citing this work"));

		QString cite = attributes.namedItem("cite").nodeValue();

		if (cite.isEmpty()) {
			info->Append(QObject::tr("If you make use of this work, please cite it appropriately. For additional details please consult "));
			info->AppendLink("http://gpumlib.sourceforge.net/");
		} else {
			info->AppendHTMLwithEscapedTags(cite);
		}

		mainGroupParameters = new GroupParameter();
		info->AppendSubSection(QObject::tr("Parameters"));
		ProcessParameters(mainGroupParameters, firstNode, arguments);

		for (auto a = arguments.begin(); a != arguments.end(); a++) {
			invalidArguments += QString(QObject::tr("Parameter %1 does not exist.\n")).arg(a.key());
		}

		if (!invalidArguments.isEmpty()) {
			QMessageBox(QMessageBox::Warning, QObject::tr("Invalid arguments"), invalidArguments, QMessageBox::Ok).exec();
		}

		info->AppendSubSection(QObject::tr("Additional information"));
		info->AppendParagraph(QObject::tr("Use the <b>-run</b> command line parameter to start the training process automatically."));
	}

	ParameterManager::~ParameterManager() {
		if (mainGroupParameters != nullptr) delete mainGroupParameters;
	}

	QString ParameterManager::GetApplicationName() const {
		return appName;
	}

	QString ParameterManager::GetApplicationAcronym() const {
		return appAcronym;
	}

	QWidget * ParameterManager::GetGUI() {
		if (propBrowser == nullptr) CreatePropertyBrowser();
		return splitter;
	}

	bool ParameterManager::ParametersAreValid(LogHTML * log) const {
		return mainGroupParameters->IsValid(log);
	}

	bool ParameterManager::AutoRun() const {
		return autorun;
	}

	void ParameterManager::GetParameterValues(ParameterValues & parametersValues) {
		mainGroupParameters->GetParametersValues(parametersValues);
	}

	void ParameterManager::ProcessParameters(GroupParameter * group, QDomNode node, QMap<QString, QString> & arguments) {
		assert(group != nullptr);

		while (!node.isNull()) {
			QDomNamedNodeMap attributes = node.attributes();

			QString name = attributes.namedItem("name").nodeValue();
			QString summary = attributes.namedItem("summary").nodeValue();
			QString defaultValue = attributes.namedItem("value").nodeValue();
			bool optional = (attributes.namedItem("optional").nodeValue().compare("true", Qt::CaseInsensitive) == 0);

			QString fullname = group->GetFullName() + " " + name;

			QString nodetype = node.nodeName().toLower();

			QString parameterName = name;
			if (optional || !defaultValue.isEmpty()) parameterName += " (optional)";

			if (nodetype == "group") {
				GroupParameter * newGroup = new GroupParameter(fullname, name, summary, optional);
				group->AddParameter(newGroup);
				info->AppendSubSubSection(parameterName);

				info->Append("<div style='margin-left:2em;'>");
				ProcessParameters(newGroup, node.firstChild(), arguments);
				info->Append("</div>");
			} else if (nodetype == "parameter") {
				info->AppendSubSubSection(parameterName);
				info->AppendParagraphWithEscapedTags(summary);

				QString type = attributes.namedItem("type").nodeValue().toLower();
				QString commandline = attributes.namedItem("commandline").nodeValue();
				QString commandlineParameters;

				Parameter * parameter = nullptr;

				if (type == "list") {
					QStringList options;
					QStringList valuesOptions;

					QDomNode nodeOption = node.firstChild();
					while (!nodeOption.isNull()) {
						QString optionType = nodeOption.nodeName();
						if (optionType.compare("option", Qt::CaseInsensitive) != 0) {
							throw std::invalid_argument("Invalid parameter specification (expected list option)");
						}

						QDomNamedNodeMap optionsAttributes = nodeOption.attributes();

						QString option = optionsAttributes.namedItem("name").nodeValue();
						options.append(option);

						QString optionValue = optionsAttributes.namedItem("value").nodeValue();
						valuesOptions.append(optionValue);

						if (!commandlineParameters.isEmpty()) commandlineParameters += " | ";
						commandlineParameters += QString("%1 (%2)").arg(optionValue).arg(option);

						nodeOption = nodeOption.nextSibling();
					}

					parameter = new ListParameter(fullname, name, summary, commandline, optional, options, valuesOptions);
				} else if (type == "device") {
					QStringList options;
					QStringList valuesOptions;

					QString deviceName;
					QString value;

					if (appSupportsCPU) {
						deviceName = "CPU";
						value = "-1";

						options.append(deviceName);
						valuesOptions.append(value);

						commandlineParameters = QString("%1 (%2)").arg(value).arg(deviceName);
					}

					if (appSupportsGPU) {
						cudaDeviceProp deviceProperties;

						int numberDevices;
						if (cudaGetDeviceCount(&numberDevices) == cudaSuccess) {
							for (int device = 0; device < numberDevices; device++) {
								if (cudaGetDeviceProperties(&deviceProperties, device) == cudaSuccess && deviceProperties.major >= 1) {
									QString deviceName = QString("%1 (%2 Mhz) - supports CUDA %3.%4").arg(deviceProperties.name).arg(deviceProperties.clockRate / 1000).arg(deviceProperties.major).arg(deviceProperties.minor);
									value = QString::number(device);

									options.append(deviceName);
									valuesOptions.append(value);

									commandlineParameters += QString(" | %1 (%2)").arg(device).arg(deviceProperties.name);
								}
							}
						} else {
							QMessageBox(QMessageBox::Warning, QObject::tr("Cannot find CUDA devices"), cudaGetErrorString(cudaGetLastError()), QMessageBox::Ok).exec();
						}

						if (numberDevices == 0 && !appSupportsCPU) {
							throw std::runtime_error("At least one CUDA capable device is required to run this application.");
						}
					}

					parameter = new ListParameter(fullname, name, summary, commandline, optional, options, valuesOptions);
					parameter->SetValue(value);
				} else if (type == "bool") {
					parameter = new BoolParameter(fullname, name, summary, commandline, optional);
					commandlineParameters = "true | false";
				} else if (type == "string") {
					parameter = new StringParameter(fullname, name, summary, commandline, optional, attributes.namedItem("regexp").nodeValue());
					commandlineParameters = "value";
				} else if (type == "filename") {
					parameter = new FilePathParameter(fullname, name, summary, commandline, optional, attributes.namedItem("filter").nodeValue());
					commandlineParameters = "filename path";
				} else {
					QString min = attributes.namedItem("minimum").nodeValue();
					QString max = attributes.namedItem("maximum").nodeValue();

					if (type == "int") {
						parameter = new IntParameter(fullname, name, summary, commandline, optional, min, max);
						commandlineParameters = "integer number";
					} else if (type == "float") {
						parameter = new FloatParameter(fullname, name, summary, commandline, optional, min, max, attributes.namedItem("decimals").nodeValue(), attributes.namedItem("step").nodeValue());
						commandlineParameters = "number";
					}
				}

				if (parameter == nullptr) {
					throw std::invalid_argument("Invalid parameter specification (unknown parameter type)");
				}

				info->Append(QString("<p><u>command line</u>: -%1 &lt;%2&gt;").arg(commandline).arg(commandlineParameters));
				if (!defaultValue.isEmpty()) info->Append(QString(" <u>default value</u>: %1").arg(defaultValue));
				info->Append("</p>");

				QString value;
				if (arguments.contains(commandline)) {
					value = arguments[commandline];
					arguments.remove(commandline);
				} else {
					value = defaultValue;
				}

				if (!value.isEmpty()) parameter->SetValue(value);
				group->AddParameter(parameter);
			} else {
				throw std::invalid_argument("Unknown tag in the parameters specification file)");
			}

			node = node.nextSibling();
		}
	}

	void ParameterManager::CreatePropertyBrowser() {
		splitter = new QSplitter();

		propBrowser = new QtTreePropertyBrowser();
		propBrowser->setResizeMode(QtTreePropertyBrowser::ResizeToContents);
		mainGroupParameters->AddToGUIpropertyBrowser(propBrowser);

		splitter->addWidget(propBrowser);

		labelInfo = new QLabel();
		labelInfo->setStyleSheet("QLabel { background-color: white; width: 100%; height: 100%; margin: 0.3em; }");
		labelInfo->setAlignment(Qt::AlignTop);
		labelInfo->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::LinksAccessibleByMouse);
		labelInfo->setCursor(QCursor(Qt::IBeamCursor));
		labelInfo->setOpenExternalLinks(true);
		labelInfo->setWordWrap(true);
		labelInfo->setText(info->ToString());

		scrollLabel = new QScrollArea();
		scrollLabel->setWidget(labelInfo);
		scrollLabel->setWidgetResizable(true);

		splitter->addWidget(scrollLabel);
		//splitter->setStretchFactor(0,1);
		//splitter->setStretchFactor(1,1);
	}

} //namespace GPUMLib
