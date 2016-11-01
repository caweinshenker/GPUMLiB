/*
	Noel Lopes is a Professor at the Polytechnic of Guarda, Portugal
	Copyright (C) 2009, 2010, 2011, 2012, 2013, 2014 Noel de Jesus Mendonça Lopes

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
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
	*/

#ifndef GPUMLIB_DATA_INPUT_FILE_H
#define GPUMLIB_DATA_INPUT_FILE_H

#include <QFile>
#include <QTextStream>
#include <QRegularExpression>

namespace GPUMLib {

	class DataInputFile {
	private:
		QFile file;
		QTextStream fs;
		QRegularExpression splitExpression;

	public:
		DataInputFile(const QString & filename) : file(filename), fs(&file), splitExpression(filename.endsWith(".csv", Qt::CaseInsensitive) ? ",\\s*" : "\\s+") {
			if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
				throw QString("Could not open file: <i>%1</i>.").arg(filename);
			}
		}

		bool AtEnd() const {
			return fs.atEnd();
		}

		QStringList ReadLine() {
			return fs.readLine().split(splitExpression);
		}
	};

} // namespace GPUMLib

#endif // DATAINPUTFILE_H
