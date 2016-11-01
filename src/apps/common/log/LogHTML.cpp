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

#include <QRegularExpression>

#include "LogHTML.h"

namespace GPUMLib {

	LogHTML::LogHTML() : outputStream(&log, QIODevice::WriteOnly) {
		closed = false;
	}

	LogHTML::LogHTML(const QString & appName) : outputStream(&log, QIODevice::WriteOnly) {
		closed = false;
		AppendGPUMLibHeader(appName);
	}

	LogHTML::LogHTML(const QString & appName, const QString & filename) : outputFile(new QFile(filename)) {
		if (!outputFile->open(QIODevice::WriteOnly | QIODevice::Text)) {
			outputFile.reset(nullptr);
			log = QString("Could not open the file '%1' for writing.").arg(filename);
			throw;
		}

		outputStream.setDevice(outputFile.get());

		closed = false;

		Append("<!DOCTYPE html>");
		Append("<head>");
		
		AppendTag("title", "GPUMLib log file");

		Append("<style>");
		Append("*{margin:0;padding:0;}");
		Append("body{margin:1em;}");
		Append("th,#gpumlib{background-color:#000040;color:white;}");
		Append("#gpumlib{padding:1em;}");
		Append("a{color:yellow;}");
		Append("strong{color:red;}");
		Append("table,h1,h2,h3,h4{margin-top:1ex;}");
		Append("th,td{padding:0.2em 0.3em;}");
		Append("#gpumlib h1{margin-top:0;}");
		Append("ul{margin-left:3em;}");		
		Append("</style>");

		Append("</head>");

		Append("<body>");
		Append("<div id=\"gpumlib\">");
		AppendGPUMLibHeader(appName);
		Append("</div>");
	}

	const char * LogHTML::ColTag() const {
		return (row < numberHeaderRows || col < numberHeaderCols) ? "th" : "td";
	}

	void LogHTML::AppendGPUMLibHeader(const QString & appName) {
		AppendTag("h1", QString("GPUMLib - GPU Machine Learning Library - Version %1").arg(GPUMLIB_VERSION));
		AppendSection(appName);
		AppendLink("http://gpumlib.sourceforge.net/");
		AppendLine();
	}

	void LogHTML::Close() {
		if (!closed) {
			Append("</body>");
			Append("</html>");
			if (outputFile != nullptr) {
				outputFile->close();
				outputFile.reset(nullptr);
			}

			closed = true;
		}
	}

	LogHTML::~LogHTML() {
		Close();
	}

	QString LogHTML::ToString() {
		if (outputFile != nullptr) {
			log = QString(QObject::tr("<p>Log file saved to <a href=\"%1\">%1</a>.</p>")).arg(outputFile->fileName());
			Close();
		}

		return log;
	}

	void LogHTML::Append(const LogHTML & loghtml) {
		outputStream << loghtml.log;
	}

	void LogHTML::Append(const QString & s) {
		outputStream << s;
	}

	void LogHTML::AppendHTMLwithEscapedTags(QString & html) {
		static QRegularExpression escapedTag("\\[(.*?)\\]", QRegularExpression::DotMatchesEverythingOption);

		html.replace("[ul]", "[ul style='list-style-type:square;']");
		html.replace("[li]", "[li style='margin-top:0.75ex; padding-left:1em;']");
		Append(html.replace(escapedTag, "<\\1>"));
	}

	void LogHTML::AppendParagraphWithEscapedTags(QString & paragraph) {
		Append("<p>");
		AppendHTMLwithEscapedTags(paragraph); // It is important to the paragraph to be changed here.
		Append("</p>");
	}

	void LogHTML::AppendLine() {
		Append("<br />");
	}

	void LogHTML::AppendLine(const QString & text) {
		Append(text);
		AppendLine();
	}

	void LogHTML::AppendTitle(const QString & title) {
		AppendTag("h1", title);
	}

	void LogHTML::AppendSection(const QString & name) {
		AppendTag("h2", name);
	}

	void LogHTML::AppendSubSection(const QString & name) {
		AppendTag("h3", name);
	}

	void LogHTML::AppendSubSubSection(const QString & name) {
		AppendTag("h4", name);
	}

	void LogHTML::AppendParagraph(const QString & paragraph) {
		AppendTag("p", paragraph);
	}

	void LogHTML::AppendTag(const QString & tag, const QString & value) {
		Append(QString("<%1>%2</%1>").arg(tag, value));
	}

	void LogHTML::AppendTag(const QString & tag, const QString & value, const QString & style) {
		Append(QString("<%1 style=\"%2\">%3</%1>").arg(tag, style, value));
	}

	void LogHTML::AppendLink(const QString & link, const char * text) {
		Append(QString("<a href=\"%1\">").arg(link));
		Append(QString("%1</a>").arg((text == nullptr) ? link : QString(text)));
	}

	void LogHTML::BeginList() {
		Append("<ul>");
	}

	void LogHTML::EndList() {
		Append("</ul>");
	}

	void LogHTML::AddListItem(QString & item) {
		AppendTag("li", item);
	}

	void LogHTML::BeginTable(int numberHeaderRows, int numberHeaderCols) {
		Append("<table border=\"1\" cellpadding=\"3\" cellspacing=\"0\" style=\"border-style:solid;\">");
		this->numberHeaderRows = numberHeaderRows;
		this->numberHeaderCols = numberHeaderCols;
		row = 0;
	}

	void LogHTML::EndTable() {
		Append("</table>");
	}

	void LogHTML::BeginRow() {
		Append("<tr>");
		col = 0;
	}

	void LogHTML::EndRow() {
		Append("</tr>");
		++row;
	}

	void LogHTML::AddColumn(const QString & value) {
		AppendTag(ColTag(), value);
		++col;
	}

	void LogHTML::AddColumn(const QString &value, int colspan, int rowspan) {
		QString span;
		if (colspan > 1) span += QString("colspan=\"%1\"").arg(colspan);
		if (rowspan > 1) {
			if (!span.isEmpty()) span += " ";
			span += QString("rowspan=\"%1\"").arg(rowspan);
		}

		Append(QString("<%1 %2 style=\"text-align:center\">%4</%1>").arg(ColTag()).arg(span).arg(value));
		col += colspan;
	}

	void LogHTML::AddEmptyColumn() {
		AddColumn("");
	}

	void LogHTML::AddPercentageColumn(double value) {
		QString sValue = QString::number(value * 100.0, 'f', 2);
		AddColumn(sValue);
	}

} // namespace GPUMLib
