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

#ifndef GPUMLIB_LOG_HTML_H
#define GPUMLIB_LOG_HTML_H

#include <memory>
#include <QFile>
#include <QTextStream>

namespace GPUMLib {

	class LogHTML {
	public:
		LogHTML();
		LogHTML(const QString & appName);
		LogHTML(const QString & appName, const QString & filename);

		~LogHTML();

		QString ToString();

		void Append(const LogHTML &loghtml);
		void Append(const QString & s);

		void AppendHTMLwithEscapedTags(QString & html);
		void AppendParagraphWithEscapedTags(QString & paragraph);

		void AppendLine();
		void AppendLine(const QString & text);

		void AppendTitle(const QString & title);        // h1
		void AppendSection(const QString & name);       // h2
		void AppendSubSection(const QString & name);    // h3
		void AppendSubSubSection(const QString & name); // h4

		void AppendParagraph(const QString & paragraph);

		void AppendTag(const QString & tag, const QString & value);
		void AppendTag(const QString & tag, const QString & value, const QString & style);

		void AppendLink(const QString & link, const char * text = nullptr);

		void BeginList();
		void EndList();
		void AddListItem(QString & item);

		void BeginTable(int numberHeaderRows = 1, int numberHeaderCols = 0);
		void EndTable();

		void BeginRow();
		void EndRow();

		void AddColumn(const QString &value);
		void AddColumn(const QString & value, int colspan, int rowspan = 1);

		template<class T> void AddColumn(T value) {
			AddColumn(QString("%1").arg(value));
		}

		void AddEmptyColumn();

		void AddPercentageColumn(double value);

		void AppendGPUMLibHeader(const QString & appName);

	private:
		std::unique_ptr<QFile> outputFile;
		QTextStream outputStream;
		QString log;

		int numberHeaderRows;
		int numberHeaderCols;
		int row;
		int col;

		bool closed;

		const char * ColTag() const;

		void Close();
	};

} // namespace GPUMLib

#endif // LOGHTML_H
