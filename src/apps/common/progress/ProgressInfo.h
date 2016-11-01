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

#ifndef GPUMLIB_PROGRESS_INFO_H
#define GPUMLIB_PROGRESS_INFO_H

#include <QProgressDialog>
#include <QElapsedTimer>

namespace GPUMLib {

	class ProgressInfo {
	public:
		ProgressInfo(QWidget * parent, const char * title, int minimum, int maximum, int secondsBetweenUpdates = 7);
		~ProgressInfo();

		bool WasCanceled() const;
		bool NeedsUpdating() const;

		void Update(QString & text);
		void Update(const char * text);
		void SetValue(int value);
		void SetValue(int value, QString & text);

		void End();

	private:
		QProgressDialog progress;
		QElapsedTimer timer;
		qint64 milisecondsBetweenUpdates;
	};

} // namespace GPUMLib

#endif // GPUMLIB_PROGRESS_INFO_H
