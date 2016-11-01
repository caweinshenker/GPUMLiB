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

#include "ProgressInfo.h"

namespace GPUMLib {

	ProgressInfo::ProgressInfo(QWidget * parent, const char * title, int minimum, int maximum, int secondsBetweenUpdates) : progress(parent) {
		progress.setWindowModality(Qt::WindowModal);
		progress.setWindowTitle(title);
		progress.setRange(minimum, maximum);
		progress.setValue(minimum);

		milisecondsBetweenUpdates = secondsBetweenUpdates * 1000;
		timer.start();
	}

	ProgressInfo::~ProgressInfo() {
		End();
	}

	bool ProgressInfo::WasCanceled() const {
		return progress.wasCanceled();
	}

	bool ProgressInfo::NeedsUpdating() const {
		return timer.hasExpired(milisecondsBetweenUpdates);
	}

	void ProgressInfo::Update(QString & text) {
		progress.setLabelText(text);
		timer.restart();
	}

	void ProgressInfo::Update(const char * text) {
		QString t(text);
		Update(t);
	}

	void ProgressInfo::SetValue(int value) {
		progress.setValue(value);
		timer.restart();
	}

	void ProgressInfo::SetValue(int value, QString & text) {
		progress.setLabelText(text);
		SetValue(value);
	}

	void ProgressInfo::End() {		
		progress.cancel();
		progress.close();
	}

} // namespace GPUMLib
