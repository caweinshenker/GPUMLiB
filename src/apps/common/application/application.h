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

#ifndef GPUMLIB_APPLICATION_H
#define GPUMLIB_APPLICATION_H

#include "../widget/AlgorithmWidget.h"

#include <type_traits>
#include <QApplication>
#include <QMessageBox>

namespace GPUMLib {

	class Application : public QApplication {

	public:
		Application(int & argc, char ** argv);

		template<class T> int RunAlgorithmWidget(const char * parameterFile) {
			static_assert(std::is_base_of<AlgorithmWidget, T>::value, "T must be a subclass of GPUMLib::AlgorithmWidget");

			try {
				T widget(parameterFile, argc, argv);

				if (widget.AutoRun()) widget.StartRunning();

				return this->exec();
			} catch (std::exception e) {
				QMessageBox(QMessageBox::Critical, QObject::tr("Error"), e.what(), QMessageBox::Ok).exec();
				return QMessageBox::Critical;
			}
		}

	private:
		int argc;
		char ** argv;

	};

} // namespace GPUMLib

#endif // GPUMLIB_APPLICATION_H
