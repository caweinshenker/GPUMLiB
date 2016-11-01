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
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using System.Xml;
using System.Xml.Serialization;
using System.Drawing.Imaging;

namespace DBNanalysis {
	public partial class FormAnalysisDBN : Form {
		private const int SPACE_BETWEEN_IMAGES = 4;
		private const int RECEPTIVE_FIELDS_ROWS = 10;

		private static Color BACKGROUNG_COLOR = Color.White;

		private static Random random = new Random();

		private float minWeight;
		private float maxWeight;

		private float maxWeightBias;

		private DBNmodel model = null;
		private DataSet dstrain = null;
		private DataSet dstest = null;
		private Settings settings;

		private int maxSamples = 0;

		public FormAnalysisDBN() {
			InitializeComponent();
		}

		private void FormAnalysisDBN_Load(object sender, EventArgs e) {
			settings = new Settings();
			var args = Environment.GetCommandLineArgs();
			if (args.Length > 1) settings.ModelFilename = args[1];
			proprieties.SelectedObject = settings;

			if (args.Length > 1) {
				Console.WriteLine("Loading data ...");
				LoadModel();
				Console.WriteLine("Saving model outputs ...");
				SaveModelOutputs();
				Close();
			}
		}

		private void LoadModel() {
			try {
				Cursor.Current = Cursors.WaitCursor;
				XmlSerializer xmlSerializer = new XmlSerializer(typeof(DBNmodel));
				XmlReader xmlReader = XmlReader.Create(settings.ModelFilename);

				model = (DBNmodel) xmlSerializer.Deserialize(xmlReader);
			} catch {
				model = null;
			} finally {
				Cursor.Current = Cursors.Arrow;
			}

			if (model == null || model.layers.Length == 0) {
				MessageBox.Show("Invalid model.");
				return;
			}

			RBMlayer bottomLayer = model.layers[0];

			int visibleUnits = bottomLayer.I;

			minWeight = maxWeight = bottomLayer.weights[0];
			maxWeightBias = 0.0f;

			for (int j = 0; j < bottomLayer.J; j++) {
				float bias = bottomLayer.biasHiddenLayer[j];

				for (int i = 0; i < visibleUnits; i++) {
					float w = bottomLayer.Weight(j, i);

					if (minWeight > w) {
						minWeight = w;
					} else if (maxWeight < w) {
						maxWeight = w;
					}

					if (bias + w > maxWeightBias) maxWeightBias = bias + w;
				}
			}

			if (settings.Width == 0 && settings.Height == 0) {
				int pixels = (int) Math.Sqrt(visibleUnits);
				if (visibleUnits == pixels * pixels) {
					settings.Width = settings.Height = pixels;
					proprieties.SelectedObject = settings;
				}
			}

			dstrain = new DataSet(model.TrainFilename, Path.GetDirectoryName(settings.ModelFilename));
			if (model.TestFilename != string.Empty) {
				dstest = new DataSet(model.TestFilename, Path.GetDirectoryName(settings.ModelFilename));
				maxSamples = dstest.Samples;
			} else {
				dstest = null;
				maxSamples = 0;
			}

			if (dstrain.Samples > maxSamples) maxSamples = dstrain.Samples;
			imageScroll.Maximum = maxSamples - 1;

			splitContainer.Panel2.Invalidate();
		}

		private void btLoadModel_Click(object sender, EventArgs e) {
			LoadModel();
		}

		private void DrawImages(object sender, PaintEventArgs e) {
			if (model == null || settings.Width == 0 || settings.Height == 0) return;

			SplitterPanel panelImages = (SplitterPanel) sender;

			Bitmap img = new Bitmap(panelImages.Width, panelImages.Height);
			Graphics g = Graphics.FromImage(img);

			g.Clear(BACKGROUNG_COLOR);

			int widthFactor = settings.Zoom * settings.Width + SPACE_BETWEEN_IMAGES;

			int DistanceBetweenImages = settings.Height * settings.Zoom + SPACE_BETWEEN_IMAGES;
			int numberImagesTrain = settings.K + 1;

			for (int i = imageScroll.Value; i < maxSamples; i++) {
				int x = (i - imageScroll.Value) * widthFactor;
				if (x >= panelImages.Width) break;

				if (i < dstrain.Samples) DrawSample(g, dstrain[i], x, 0);
				if (dstest != null && i < dstest.Samples) DrawSample(g, dstest[i], x, numberImagesTrain * DistanceBetweenImages);
			}

			int imagesBeforeReceptiveFields = numberImagesTrain;
			if (dstest != null) imagesBeforeReceptiveFields *= 2;

			// Receptive fields
			int imagesPerRow = model.layers[0].J / RECEPTIVE_FIELDS_ROWS;
			if (model.layers[0].J % RECEPTIVE_FIELDS_ROWS > 0) imagesPerRow++;

			for (int i = imageScroll.Value; i < 2 * imagesPerRow; i++) {
				int x = (i - imageScroll.Value) * widthFactor;

				for (int r = 0; r < RECEPTIVE_FIELDS_ROWS; r++) {
					int j = r * imagesPerRow + i;
					if (i >= imagesPerRow) j -= imagesPerRow;

					int yOrigin = SPACE_BETWEEN_IMAGES + (imagesBeforeReceptiveFields + r) * DistanceBetweenImages;

					if (j < model.layers[0].J) {
						if (i < imagesPerRow) {
							DrawReceptiveField(g, x, yOrigin, j);
						} else {
							DrawReceptiveFieldColor(g, x, yOrigin, j);
						}
					}
				}
			}

			e.Graphics.DrawImage(img, 0, 0);
		}

		private void DrawReceptiveFieldColor(Graphics graphics, int xOrigin, int yOrigin, int j) {
			RBMlayer bottomLayer = model.layers[0];

			float bias = bottomLayer.biasHiddenLayer[j];

			for (int h = 0; h < settings.Height; h++) {
				for (int w = 0; w < settings.Width; w++) {
					int i = h * settings.Width + w;

					float weight = bottomLayer.Weight(j, i);

					int color = (int) (255.0 * Sigmoid(bias + weight));
					SolidBrush brush;

					if (weight < 0) {
						color = 255 - color;
						brush = new SolidBrush(Color.FromArgb(0, color, color));
					} else {
						brush = new SolidBrush(Color.FromArgb(color, 0, 0));
					}

					int y = yOrigin + h * settings.Zoom;
					int x = xOrigin + w * settings.Zoom;

					graphics.FillRectangle(brush, x, y, settings.Zoom, settings.Zoom);
				}
			}
		}

		private void DrawReceptiveField(Graphics graphics, int xOrigin, int yOrigin, int j) {
			RBMlayer bottomLayer = model.layers[0];

			for (int h = 0; h < settings.Height; h++) {
				for (int w = 0; w < settings.Width; w++) {
					int i = h * settings.Width + w;

					int color = (int) (255.0f * (bottomLayer.Weight(j, i) - minWeight) / (maxWeight - minWeight));

					int y = yOrigin + h * settings.Zoom;
					int x = xOrigin + w * settings.Zoom;

					graphics.FillRectangle(new SolidBrush(Color.FromArgb(color, color, color)), x, y, settings.Zoom, settings.Zoom);
				}
			}
		}

		private void DrawSample(Graphics graphics, float[] imgBytes, int xOrigin, int yOrigin) {
			yOrigin += SPACE_BETWEEN_IMAGES;
			DrawSampleImage(graphics, xOrigin, yOrigin, imgBytes);

			for (int k = 0; k < settings.K; k++) {
				if (k > 0) for (int b = 0; b < imgBytes.Length; b++) imgBytes[b] = Binarize(imgBytes[b]);

				imgBytes = Reconstruction(imgBytes);
				yOrigin += settings.Height * settings.Zoom + SPACE_BETWEEN_IMAGES;
				DrawSampleImage(graphics, xOrigin, yOrigin, imgBytes);
			}
		}

		private void DrawSampleImage(Graphics graphics, int xOrigin, int yOrigin, float[] imgBytes) {
			for (int h = 0; h < settings.Height; h++) {
				for (int w = 0; w < settings.Width; w++) {
					int color = (int) (255.0f * imgBytes[h * settings.Width + w]);
					int y = yOrigin + h * settings.Zoom;
					int x = xOrigin + w * settings.Zoom;

					graphics.FillRectangle(new SolidBrush(Color.FromArgb(color, color, color)), x, y, settings.Zoom, settings.Zoom);
				}
			}
		}

		private void imageScroll_ValueChanged(object sender, EventArgs e) {
			if (settings.Position != imageScroll.Value + 1) {
				settings.Position = imageScroll.Value + 1;
				proprieties.Refresh();
			}
			splitContainer.Panel2.Invalidate();
		}

		private static double Sigmoid(float x) {
			return 1.0 / (1.0 + Math.Exp(-x));
		}

		private float Binarize(float probability) {
			if (settings.Binarize) {
				return (probability > random.NextDouble()) ? 1.0f : 0.0f;
			} else {
				return (float) probability;
			}
		}

		float[] Reconstruction(float[] originalValues) {
			float[] prevLayerValues = originalValues;

			int currentLayer = 0;
			foreach (RBMlayer layer in model.layers) {
				float[] layerValues = new float[layer.J];

				for (int j = 0; j < layer.J; j++) {
					float sum = layer.biasHiddenLayer[j];
					for (int i = 0; i < layer.I; i++) sum += prevLayerValues[i] * layer.Weight(j, i);
					layerValues[j] = Binarize((float) Sigmoid(sum));
				}

				prevLayerValues = layerValues;

				if (++currentLayer >= settings.LayersToProcess) break;
			}

			for (int l = currentLayer - 1; l >= 0; l--) {
				RBMlayer layer = model.layers[l];
				float[] layerValues = new float[layer.I];

				for (int i = 0; i < layer.I; i++) {
					float sum = layer.biasVisibleLayer[i];
					for (int j = 0; j < layer.J; j++) sum += prevLayerValues[j] * layer.Weight(j, i);
					layerValues[i] = (float) Sigmoid(sum);
				}

				prevLayerValues = layerValues;
			}

			return prevLayerValues;
		}

		private void SaveDatasetOutputs(DataSet ds, string aditionalInfo) {
			Cursor.Current = Cursors.WaitCursor;

			int numberLayers = model.layers.Length;

			labelOperation.Text = "Saving " + aditionalInfo + " datasets";
			progress.Maximum = numberLayers;
			statusBar.Visible = true;

			StreamWriter writer = null;
			StreamWriter writerAll = null;

			for (int layersToProcess = 1; layersToProcess <= numberLayers; layersToProcess++) {
				try {
					writer = new StreamWriter(string.Format("{0}-{1}-layer{2}-output.csv", Path.GetFileNameWithoutExtension(settings.ModelFilename), aditionalInfo, layersToProcess));

					if (layersToProcess == numberLayers) {
						writerAll = new StreamWriter(string.Format("{0}-{1}-all-output.csv", Path.GetFileNameWithoutExtension(settings.ModelFilename), aditionalInfo));
					}

					labelDone.Text = string.Format("{0}/{1}", layersToProcess, numberLayers);
					progress.Value = layersToProcess;
					this.Update();

					for (int s = 0; s < ds.Samples; s++) {
						float[] prevLayerValues = ds[s];

						for (int l = 0; l < layersToProcess; l++) {
							RBMlayer layer = model.layers[l];

							float[] layerValues = new float[layer.J];

							for (int j = 0; j < layer.J; j++) {
								float sum = layer.biasHiddenLayer[j];
								for (int i = 0; i < layer.I; i++) sum += prevLayerValues[i] * layer.Weight(j, i);
								layerValues[j] = (float) Sigmoid(sum);
							}

							prevLayerValues = layerValues;

							if (writerAll != null && layersToProcess == numberLayers) {
								if (l > 0) writerAll.Write(',');
								writerAll.Write("{0}", prevLayerValues[0]);
								for (int i = 1; i < prevLayerValues.Length; i++) {
									writerAll.Write(",{0}", prevLayerValues[i]);
								}
							}
						}
						if (writerAll != null) writerAll.WriteLine();

						writer.Write("{0}", prevLayerValues[0]);
						for (int i = 1; i < prevLayerValues.Length; i++) {
							writer.Write(",{0}", prevLayerValues[i]);
						}
						writer.WriteLine();
					}
				} catch (System.Exception exception) {
					throw exception;
				} finally {
					if (writer != null) writer.Close();
					if (writerAll != null) writerAll.Close();
				}
			}

			statusBar.Visible = false;
			Cursor.Current = Cursors.Arrow;
		}

		private void btSaveOutputs_Click(object sender, EventArgs e) {
			SaveModelOutputs();
		}

		private void SaveModelOutputs() {
			if (model == null) {
				MessageBox.Show("Please load a model first.");
				return;
			}

			SaveDatasetOutputs(dstrain, "train");
			if (dstest != null) SaveDatasetOutputs(dstest, "test");
		}

		private void proprieties_PropertyValueChanged(object s, PropertyValueChangedEventArgs e) {
			int imgIndex = settings.Position - 1;

			if (imgIndex > imageScroll.Maximum) {
				imgIndex = imageScroll.Maximum;
				settings.Position = imgIndex + 1;
				proprieties.Refresh();
			}
			if (imageScroll.Value != imgIndex) imageScroll.Value = imgIndex;

			splitContainer.Panel2.Invalidate();
		}

		private void btSaveImages_Click(object sender, EventArgs e) {
			if (model == null || settings.Width == 0 || settings.Height == 0) return;

			for (int i = 0; i < settings.NumberImagesWrite; i++) {
				int img = imageScroll.Value + i;

				if (img < dstrain.Samples) SaveImage(dstrain[img], string.Format("train{0}", img));
				if (dstest != null && i < dstest.Samples) SaveImage(dstest[img], string.Format("test{0}", img));
			}

			SaveReceptiveFieldsImage();
		}

		private void SaveReceptiveFieldsImage() {
			int J = model.layers[0].J;

			int imagesPerRow = J / RECEPTIVE_FIELDS_ROWS;
			if (J % RECEPTIVE_FIELDS_ROWS > 0) imagesPerRow++;

			int bWidth = settings.Width * settings.Zoom * imagesPerRow + SPACE_BETWEEN_IMAGES * (imagesPerRow - 1);
			int bHeight = settings.Height * settings.Zoom * RECEPTIVE_FIELDS_ROWS + SPACE_BETWEEN_IMAGES * (RECEPTIVE_FIELDS_ROWS - 1);

			Bitmap bitmapRFields = new Bitmap(bWidth, bHeight);
			Bitmap bitmapRFieldsColor = new Bitmap(bWidth, bHeight);
			Graphics g = Graphics.FromImage(bitmapRFields);
			Graphics gc = Graphics.FromImage(bitmapRFieldsColor);

			int widthFactor = settings.Zoom * settings.Width + SPACE_BETWEEN_IMAGES;
			int DistanceBetweenImages = settings.Height * settings.Zoom + SPACE_BETWEEN_IMAGES;

			for (int i = 0; i < imagesPerRow; i++) {
				int x = i * widthFactor;

				for (int r = 0; r < RECEPTIVE_FIELDS_ROWS; r++) {
					int j = r * imagesPerRow + i;

					if (j < J) {
						int yOrigin = SPACE_BETWEEN_IMAGES + r * DistanceBetweenImages;
						DrawReceptiveField(g, x, yOrigin, j);
						DrawReceptiveFieldColor(gc, x, yOrigin, j);
					}
				}
			}

			bitmapRFields.Save("RFields.png", ImageFormat.Png);
			bitmapRFieldsColor.Save("RFieldsColor.png", ImageFormat.Png);
		}

		private void SaveImage(float[] imgBytes, string filename) {
			Bitmap bitmap = new Bitmap(settings.Width * settings.Zoom, settings.Height * settings.Zoom);
			Graphics g = Graphics.FromImage(bitmap);
			DrawSampleImage(g, 0, 0, imgBytes);
			bitmap.Save(filename + ".png", ImageFormat.Png);

			for (int k = 0; k < settings.K; k++) {
				if (k > 0) for (int b = 0; b < imgBytes.Length; b++) imgBytes[b] = Binarize(imgBytes[b]);

				imgBytes = Reconstruction(imgBytes);
				DrawSampleImage(g, 0, 0, imgBytes);
				bitmap.Save(filename + string.Format("-cd{0}.png", k), ImageFormat.Png);
			}
		}
	}
}

