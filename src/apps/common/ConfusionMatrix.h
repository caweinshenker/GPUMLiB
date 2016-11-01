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

#ifndef GPUMLIB_CONFUSION_MATRIX
#define GPUMLIB_CONFUSION_MATRIX

#include "../../memory/HostMatrix.h"
#include "log/LogHTML.h"

namespace GPUMLib {

	class ConfusionMatrix {
	private:
		GPUMLib::HostMatrix<int> results;
		int classes;

		int TP(int c) const {
			return results(c, c);
		}

		int FP(int c) const {
			int fp = 0;
			for (int o = 0; o < classes; o++) if (o != c) fp += results(o, c);

			return fp;
		}

		int FN(int c) const {
			int fn = 0;
			for (int p = 0; p < classes; p++) if (p != c) fn += results(c, p);

			return fn;
		}

		int Positives(int c) const { // TP + FN
			int total = 0;
			for (int p = 0; p < classes; p++) total += results(c, p);

			return total;
		}

		double Precision(int c) const {
			int tp = TP(c);

			if (tp == 0) {
				return (Positives(c) == 0) ? 1.0 : 0.0;
			}
			else {
				return (double)tp / (tp + FP(c));
			}
		}

		double Recall(int c) const {
			int positives = Positives(c);

			if (positives == 0) {
				return 1.0;
			}
			else {
				return (double)TP(c) / positives;
			}
		}

	public:
		ConfusionMatrix(int classes) {
			this->classes = classes;
			results.ResizeWithoutPreservingData(classes, classes);

			Reset();
		}

		void Reset() {
			for (int c = 0; c < classes; c++) {
				for (int p = 0; p < classes; p++) results(c, p) = 0;
			}
		}

		void Classify(int correctClass, int predictedClass) {
			results(correctClass, predictedClass)++;
		}

		double Precision() const {
			double sum = 0.0;
			int count = 0;

			for (int c = 0; c < classes; c++) {
				if (Positives(c) > 0) {
					sum += Precision(c);
					count++;
				}
			}

			return sum / count;
		}

		double Recall() const {
			double sum = 0.0;
			int count = 0;

			for (int c = 0; c < classes; c++) {
				if (Positives(c) > 0) {
					sum += Recall(c);
					count++;
				}
			}

			return sum / count;
		}

		double FnScore(int n) const {
			assert(n > 0);

			double precision = Precision();
			double recall = Recall();

			if (precision == 0.0 && recall == 0.0) return 0.0;

			double nn = (double)n * n;

			return (1.0 + nn) * precision * recall / (nn * precision + recall);
		}

		double FMeasure() const {
			return FnScore(1);
		}

		double Accuracy() const {
			double correct = 0;
			double total = 0;

			for (int c = 0; c < classes; c++) {
				for (int p = 0; p < classes; p++) {
					int classified = results(c, p);

					if (c == p) correct += classified;
					total += classified;
				}
			}

			return correct / total;
		}

		void Log(LogHTML & log) const {
			log.BeginTable();

			log.BeginRow();
			log.AddColumn("class");
			log.AddColumn(QString("predicted"), classes);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("actual");
			for (int c = 0; c < classes; c++) log.AddColumn<int>(c);
			log.EndRow();

			for (int c = 0; c < classes; c++) {
				log.BeginRow();

				log.AddColumn<int>(c);

				for (int p = 0; p < classes; p++) {
					log.AddColumn<int>(results(c, p));
				}

				log.EndRow();
			}

			log.EndTable();
		}
	};

} // namespace GPUMLib

#endif // GPUMLIB_CONFUSION_MATRIX
