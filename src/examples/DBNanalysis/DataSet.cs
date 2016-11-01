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

using System;
using System.Collections.Generic;
using System.IO;

namespace DBNanalysis {
	class DataSet {
		private List<float[]> samples;

		private void LoadDataSet(string filename) {
			StreamReader reader = null;

			samples = new List<float[]>();

			try {
				reader = new StreamReader(filename);

				char [] colsSeparator;
				if (filename.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)) {
					colsSeparator = new char[] {','};
				} else {
					colsSeparator = new char[] {'\t', ' '};
				}

				//string[] listSeparator = { CultureInfo.CurrentCulture.TextInfo.ListSeparator };

				while (!reader.EndOfStream) {
					string[] svalues = reader.ReadLine().Split(colsSeparator);

					int columns = svalues.Length;
					while (columns > 0 && string.IsNullOrWhiteSpace(svalues[columns - 1])) columns--;

					if (columns > 0) {
						float[] values = new float[columns];
						for (int v = 0; v < columns; v++) values[v] = float.Parse(svalues[v]);
						samples.Add(values);
					}
				}
			} catch (System.Exception e) {
				samples = null;
				throw e;
			} finally {
				if (reader != null) reader.Close();
			}
		}

		public DataSet(string filename, string path) {
			try {
				LoadDataSet(Path.Combine(path, filename));
			} catch (Exception originalException) {
				try {
					LoadDataSet(filename);
				} catch (Exception) {
					throw originalException;
				}
			}
		}

		public float[] this[int index] {
			get { return samples[index]; }
		}

		public int Samples {
			get { return samples.Count; }
		}
	}
}
