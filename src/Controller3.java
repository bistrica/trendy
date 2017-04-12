import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;

/**
 * 
 * 
 * @author Aleksandra Do³êga
 * 
 */
public class Controller3 {

	public static final int KMEANS = 0;
	public static final int WATERSHED = 1;
	public static final int THRESHOLDING = 2;
	@FXML
	private Button button;
	@FXML
	private ImageView originalFrame;
	@FXML
	private ImageView originalFrame2;
	@FXML
	private ImageView originalFrame3;
	@FXML
	private ImageView originalFrame4;
	@FXML
	private ImageView originalFrame5;

	int SIZE;
	public static int SEGMENTATION_TYPE;
	public static String redStainsFilename, blueStainsFilename;

	@FXML
	protected void load() {
		int width = 500, width2 = 300;// 500 300
		originalFrame.setFitWidth(width);
		originalFrame2.setFitWidth(width);
		originalFrame3.setFitWidth(width2);
		originalFrame4.setFitWidth(width2);
		originalFrame5.setFitWidth(width2);
		originalFrame.setPreserveRatio(true);
		originalFrame2.setPreserveRatio(true);
		originalFrame3.setPreserveRatio(true);
		originalFrame4.setPreserveRatio(true);
		originalFrame5.setPreserveRatio(true);

		Image imageToShow;
		imageToShow = getFrame(redStainsFilename);
		int SIZ_COLOR1 = SIZE;
		originalFrame.setImage(imageToShow);

		imageToShow = getFrame(blueStainsFilename);

		int SIZ_COLOR2 = SIZE;

		double s = (double) SIZ_COLOR1 / (double) SIZ_COLOR2;
		System.out.println(">>> Red stains: " + SIZ_COLOR1 + ", blue stains: " + SIZ_COLOR2 + ", result = " + s);
		originalFrame2.setImage(imageToShow);

	}

	private Image getFrame(String tit) {
		System.out.println("INS");
		BufferedImage bi = null;
		try {
			bi = ImageIO.read(new File(tit));//

		} catch (IOException e) {
			System.out.println("Err " + tit + " [[" + e);
		}

		Image imageToShow = null;
		// bi = (countOnesBI(bi));
		Mat frame = bufferedImageToMat(bi);

		try {

			if (!frame.empty()) {
				if (SEGMENTATION_TYPE == Controller.KMEANS)
					frame = this.segmentationKMeans(frame);
				else if (SEGMENTATION_TYPE == Controller.WATERSHED)
					frame = this.segmentationWatershed(frame);
				else // if (SEGMENTATION_TYPE==Controller.THRESHOLDING)
					frame = this.segmentationThresholding(frame);
				imageToShow = mat2Image(frame);
				/*
				 * BufferedImage i = countOnesBI(bi); Mat m =
				 * bufferedImageToMat(i); m = setColors(m);
				 */
				// imageToShow = mat2Image(frame);

			}

		} catch (Exception e) {

			System.err.print("ERROR");
			e.printStackTrace();
		}

		return imageToShow;
	}

	private Image countOnes(BufferedImage imm) {
		int[] pixels = new int[imm.getWidth() * imm.getHeight()];
		imm.getRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		// System.out.println("pix " + pixels.length + ",");

		for (int i = 0; i < pixels.length; i++) {

			int rgb = pixels[i];
			int r = (rgb >> 16) & 0xFF;
			int g = (rgb >> 8) & 0xFF;
			int b = (rgb & 0xFF);

			if (!(r == 0 && g == 255 && b == 255)) {
				pixels[i] = 0xFF000000;
				// r = 0;
				// b = 0;
				// g = 0;
			}

		}
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		WritableImage wr = null;
		if (imm != null) {
			wr = new WritableImage(imm.getWidth(), imm.getHeight());
			PixelWriter pw = wr.getPixelWriter();
			for (int x = 0; x < imm.getWidth(); x++) {
				for (int y = 0; y < imm.getHeight(); y++) {
					pw.setArgb(x, y, imm.getRGB(x, y));
				}
			}
		}
		return wr;
	}

	private BufferedImage countOnesBI(BufferedImage imm) {
		int[] pixels = new int[imm.getWidth() * imm.getHeight()];
		imm.getRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		// System.out.println("pix " + pixels.length + ",");

		for (int i = 0; i < pixels.length; i++) {

			int rgb = pixels[i];
			int r = (rgb >> 16) & 0xFF;
			int g = (rgb >> 8) & 0xFF;
			int b = (rgb & 0xFF);

			if (!(r == 0 && g == 255 && b == 255)) {
				pixels[i] = 0xFF000000;
				// r = 0;
				// b = 0;
				// g = 0;
			}

		}
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		return imm;
	}

	private Mat segmentationThresholding(Mat frame) {
		// if (true)
		// return frame;
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat(frame.size(), CvType.CV_8UC3);

		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

		Core.normalize(grayImage, detectedEdges, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC3);

		Mat matEllipse = Mat.ones(new Size(5, 5), 0);
		matEllipse.put(0, 0, 0);
		matEllipse.put(0, 1, 0);
		matEllipse.put(0, 3, 0);
		matEllipse.put(0, 4, 0);
		matEllipse.put(4, 0, 0);
		matEllipse.put(4, 1, 0);
		matEllipse.put(4, 3, 0);
		matEllipse.put(4, 4, 0);

		Imgproc.threshold(detectedEdges, detectedEdges, 100, 255, Imgproc.THRESH_BINARY);// 60
																							// Imgproc.morphologyEx(detectedEdges,
		// detectedEdges, Imgproc.MORPH_OPEN, matEllipse);
		Imgproc.erode(detectedEdges, detectedEdges, matEllipse); //
		Imgproc.morphologyEx(detectedEdges, detectedEdges, //
				Imgproc.MORPH_CLOSE, matEllipse);

		detectedEdges = setColors(detectedEdges);
		return detectedEdges;

	}

	private Mat segmentationWatershed(Mat frame) {
		ArrayList<Scalar> asc = new ArrayList<>();

		for (int i = 10; i < 250; i++)
			for (int j = 0; j < 250; j++)
				asc.add(new Scalar(i, j, i));
		Collections.shuffle(asc);

		Mat grayImage = new Mat();

		Mat detectedEdges = new Mat(frame.size(), CvType.CV_8UC3);

		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

		Core.normalize(grayImage, detectedEdges, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC3);

		Mat matEllipse = Mat.ones(new Size(5, 5), 0);
		matEllipse.put(0, 0, 0);
		matEllipse.put(0, 1, 0);
		matEllipse.put(0, 3, 0);
		matEllipse.put(0, 4, 0);
		matEllipse.put(4, 0, 0);
		matEllipse.put(4, 1, 0);
		matEllipse.put(4, 3, 0);
		matEllipse.put(4, 4, 0);

		Mat det = new Mat();
		detectedEdges.copyTo(det);

		Imgproc.GaussianBlur(detectedEdges, detectedEdges, new Size(17, 17), 0);

		Mat inv = new Mat();
		detectedEdges.copyTo(inv);

		Imgproc.threshold(inv, inv, 160, 255, Imgproc.THRESH_BINARY_INV);

		Imgproc.morphologyEx(inv, inv, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));

		Imgproc.threshold(detectedEdges, detectedEdges, 130, 255, Imgproc.THRESH_BINARY);

		Mat inv3 = new Mat();
		detectedEdges.copyTo(inv3);
		Core.bitwise_not(inv3, inv3);

		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1), new Point(1, 1), 3);

		Mat fin = new Mat();
		inv3.copyTo(fin);

		fin.convertTo(fin, CvType.CV_8U);

		Mat cont = new Mat();

		fin.copyTo(cont);// by³o 3/2

		final List<MatOfPoint> points = new ArrayList<>();
		final Mat hierarchy = new Mat();

		Imgproc.findContours(cont, points, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
		Mat markers = Mat.zeros(frame.size(), CvType.CV_32SC1);

		for (int i = 0; i < points.size(); i++) {
			Scalar as = asc.get(i);

			Imgproc.drawContours(markers, points, i, as, -1);
		}
		Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);

		Imgproc.cvtColor(inv, inv, Imgproc.COLOR_GRAY2RGB);
		Imgproc.cvtColor(inv3, inv3, Imgproc.COLOR_GRAY2RGB);
		Mat xx = new Mat();

		inv.copyTo(xx);// inv
		;
		// if (true)
		// return inv3;
		Imgproc.watershed(xx, markers);

		BufferedImage imm = new BufferedImage(markers.width(), markers.height(), BufferedImage.TYPE_BYTE_GRAY);
		markers.convertTo(markers, CvType.CV_8U);
		points.clear();

		final Mat hierarchy2 = new Mat();

		matToBufferedImage(markers, imm);

		int[] pixels = new int[imm.getWidth() * imm.getHeight()];
		imm.getRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());
		System.out.println("pix " + pixels.length + ",");
		HashMap<String, Integer> mm = new HashMap<>();
		for (int i = 0; i < pixels.length; i++) {
			int rgb = pixels[i];
			int r = (rgb >> 16) & 0xFF;
			int g = (rgb >> 8) & 0xFF;
			int b = (rgb & 0xFF);

			if (r == 255 && g == 255 && b == 255) {
				r = 10;
				g = 10;
				b = 10;
				pixels[i] = 0xFF0A0A0A;
			}
			if (r == 0 && g == 0 && b == 0) {
				pixels[i] = 0xFFFFFFFF;
				r = 255;
				b = 255;
				g = 255;
			}
			String pix = r + "|" + b + "|" + g;
			if (mm.containsKey(pix))
				mm.put(pix, mm.get(pix) + 1);
			else
				mm.put(pix, 1);
		}
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		BufferedImage imm2 = new BufferedImage(imm.getWidth(), imm.getHeight(), 5);
		imm2.getGraphics().drawImage(imm, 0, 0, null);
		Mat immat = bufferedImageToMat(imm2);

		// Mat gr = new Mat();
		// Imgproc.morphologyEx(immat, gr, Imgproc.MORPH_GRADIENT, matEllipse,
		// new Point(1, 1), 1);

		// Imgproc.threshold(gr, gr, 1, 255, Imgproc.THRESH_BINARY);

		// Core.bitwise_or(immat, gr, immat);

		// AGAIN
		matToBufferedImage(immat, imm);

		pixels = new int[imm.getWidth() * imm.getHeight()];
		imm.getRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());// ((DataBufferByte)
																						// imm.getRaster().getDataBuffer()).getData();
		System.out.println("pix " + pixels.length + ",");
		mm.clear();
		for (int i = 0; i < pixels.length; i++) {

			int rgb = pixels[i];
			int r = (rgb >> 16) & 0xFF;
			int g = (rgb >> 8) & 0xFF;
			int b = (rgb & 0xFF);

			if (r == 255 && g == 255 && b == 255) {
				pixels[i] = 0xFF000000;
				r = 0;
				b = 0;
				g = 0;
			}

		}
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		imm2 = new BufferedImage(imm.getWidth(), imm.getHeight(), 5);
		imm2.getGraphics().drawImage(imm, 0, 0, null);
		immat = bufferedImageToMat(imm2);

		points.clear();

		Imgproc.cvtColor(immat, immat, Imgproc.COLOR_RGB2GRAY);
		Imgproc.threshold(immat, immat, 1, 256, Imgproc.THRESH_BINARY_INV);
		immat.convertTo(immat, CvType.CV_8U);
		Mat gr = new Mat();
		Imgproc.morphologyEx(immat, gr, Imgproc.MORPH_GRADIENT, matEllipse, new Point(1, 1), 1);
		Core.bitwise_or(immat, gr, immat);

		Core.bitwise_not(immat, immat);

		Imgproc.findContours(immat, points, hierarchy2, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

		Mat markersx = Mat.zeros(immat.size(), CvType.CV_32SC3);

		ArrayList<String> color = new ArrayList<>(), c = new ArrayList<>();
		for (int i = 0; i < points.size(); i++) {
			double[] v = asc.get(i).val;
			String sv = v[0] + "|" + v[1] + "|" + v[2];
			if (color.contains(sv))
				System.out.println("CONT " + sv);
			else
				c.add(sv);
			color.add(sv);
			Imgproc.drawContours(markersx, points, i, asc.get(i), -1);
		}

		System.out.println();
		mm.clear();
		for (int r = 0; r < markersx.rows(); r++) {
			for (int cc = 0; cc < markersx.cols(); cc++) {

				int[] data = new int[3];
				markersx.get(r, cc, data);

				if (data[0] == 0 && data[1] == 0 && data[2] == 0)
					continue;
				String pix = data[0] + "|" + data[1] + "|" + data[2];
				if (mm.containsKey(pix))// pixels[i]))
					mm.put(pix, mm.get(pix) + 1);
				else
					mm.put(pix, 1);
			}

		}

		int noOfCells = mm.keySet().size();
		ArrayList<Integer> sizes = new ArrayList<>(mm.values());
		Collections.sort(sizes);
		int median = sizes.size() % 2 == 0 ? (sizes.get(sizes.size() / 2) + sizes.get(sizes.size() / 2 - 1)) / 2
				: sizes.get(sizes.size() / 2);
		System.out.println(sizes.size() + " " + median);
		int all = 0;
		for (int i : sizes)
			all += i;
		int siz = (int) ((double) all / (double) median);
		System.out.println("Size " + siz);
		SIZE = siz;

		return markersx;
	}

	public static Mat bufferedImageToMat(BufferedImage bi) {
		Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		mat.put(0, 0, data);
		return mat;
	}

	private static List<Mat> showClusters(Mat cutout, Mat labels, Mat centers) {
		centers.convertTo(centers, CvType.CV_8UC1, 255.0);
		centers.reshape(3);

		List<Mat> clusters = new ArrayList<>();
		for (int i = 0; i < centers.rows(); i++) {
			clusters.add(Mat.zeros(cutout.size(), cutout.type()));
		}

		Map<Integer, Integer> counts = new HashMap<>();
		for (int i = 0; i < centers.rows(); i++)
			counts.put(i, 0);

		int rows = 0;
		for (int y = 0; y < cutout.rows(); y++) {
			for (int x = 0; x < cutout.cols(); x++) {
				int label = (int) labels.get(rows, 0)[0];
				int r = (int) centers.get(label, 2)[0];
				int g = (int) centers.get(label, 1)[0];
				int b = (int) centers.get(label, 0)[0];
				counts.put(label, counts.get(label) + 1);
				clusters.get(label).put(y, x, b, g, r);
				rows++;
			}
		}

		return clusters;
	}

	private Mat segmentationKMeans(Mat frame) {
		ArrayList<Scalar> asc = new ArrayList<>();
		ArrayList<Integer> sizes = new ArrayList<>();
		int x = 0;
		int noOfCells = -1;
		Mat markersx = new Mat();

		for (int i = 10; i < 250; i++)
			for (int j = 0; j < 250; j++) {
				float[] hsb = Color.RGBtoHSB(i, j, i, null);
				// System.out.println(hsb[0] + ":" + hsb[1] + ":" + hsb[2]);
				asc.add(new Scalar(i, j, i));
			}
		Collections.shuffle(asc);

		Mat grayImage = new Mat();

		Mat detectedEdges = new Mat(frame.size(), CvType.CV_8UC3);
		System.out.println("DET " + detectedEdges.type());

		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

		Core.normalize(grayImage, detectedEdges, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC3);

		Mat matEllipse = Mat.ones(new Size(5, 5), 0);
		matEllipse.put(0, 0, 0);
		matEllipse.put(0, 1, 0);
		matEllipse.put(0, 3, 0);
		matEllipse.put(0, 4, 0);
		matEllipse.put(4, 0, 0);
		matEllipse.put(4, 1, 0);
		matEllipse.put(4, 3, 0);
		matEllipse.put(4, 4, 0);

		Mat samples = frame.reshape(1, frame.cols() * frame.rows());
		Mat samples32f = new Mat();
		samples.convertTo(samples32f, CvType.CV_32F, 1.0 / 255.0);

		Mat labels = new Mat();
		TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 100, 1);
		Mat centers = new Mat();

		Core.kmeans(samples32f, 3, labels, criteria, 1, Core.KMEANS_PP_CENTERS, centers);

		List<Mat> arrs = showClusters(frame, labels, centers);

		originalFrame3.setImage(mat2Image(arrs.get(0)));
		originalFrame4.setImage(mat2Image(arrs.get(1)));
		originalFrame5.setImage(mat2Image(arrs.get(2)));
		Mat immat = new Mat();
		HashMap<String, Integer> mm = new HashMap<>();
		List<MatOfPoint> points = new ArrayList<>();
		Mat hierarchy2 = new Mat();
		Mat best = null;
		int bestIndex = 0;
		int bestVal = Integer.MAX_VALUE;

		for (int iar = 0; iar < arrs.size(); iar++) {
			Mat xg1 = arrs.get(iar);

			Mat g1 = new Mat();
			xg1.copyTo(g1);

			g1.convertTo(g1, CvType.CV_32FC1);
			int x1 = 0;
			for (int cc = 0; cc < g1.cols(); cc++) {

				float[] data = new float[3];
				g1.get(0, cc, data);

				if (data[0] == 0 && data[1] == 0 && data[2] == 0)
					x1 += 1;

			}
			if (x1 < 0.5 * g1.cols())
				continue;

			points.clear();

			immat = new Mat();
			mm = new HashMap<>();

			g1.copyTo(immat);
			Imgproc.cvtColor(immat, immat, Imgproc.COLOR_RGB2GRAY);
			Imgproc.threshold(immat, immat, 1, 256, Imgproc.THRESH_BINARY);

			immat.convertTo(immat, 0);

			Imgproc.findContours(immat, points, hierarchy2, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
			markersx = Mat.zeros(immat.size(), CvType.CV_32SC3);

			ArrayList<String> color = new ArrayList<>(), c = new ArrayList<>();
			for (int i = 0; i < points.size(); i++) {
				double[] v = asc.get(i).val;
				String sv = v[0] + "|" + v[1] + "|" + v[2];
				if (color.contains(sv))
					System.out.println("CONT " + sv);
				else
					c.add(sv);
				color.add(sv);
				Imgproc.drawContours(markersx, points, i, asc.get(i), -1);
			}
			System.out.println("> > " + c.size() + "," + color.size());

			mm.clear();
			x = 0;
			for (int r = 0; r < markersx.rows(); r++) {
				for (int cc = 0; cc < markersx.cols(); cc++) {

					int[] data = new int[3];
					markersx.get(r, cc, data);

					if (data[0] == 0 && data[1] == 0 && data[2] == 0) {
						x++;
						continue;
					}
					String pix = data[0] + "|" + data[1] + "|" + data[2];
					if (mm.containsKey(pix))
						mm.put(pix, mm.get(pix) + 1);
					else
						mm.put(pix, 1);
				}

			}
			ArrayList<String> keys = new ArrayList<>();
			for (String k : mm.keySet()) {
				if (mm.get(k) < 18)
					keys.add(k);
			}
			for (String k : keys) {
				x += mm.get(k);
				mm.remove(k);
			}

			noOfCells = mm.keySet().size();
			sizes = new ArrayList<>(mm.values());
			Collections.sort(sizes);

			if (sizes.get(sizes.size() - 1) <= bestVal) {
				bestVal = sizes.get(sizes.size() - 1);

				bestIndex = iar;
			}
		}

		best = arrs.get(bestIndex);

		originalFrame3.setImage(mat2Image(arrs.get(0)));
		originalFrame4.setImage(mat2Image(arrs.get(1)));
		originalFrame5.setImage(mat2Image(arrs.get(2)));

		best.copyTo(immat);

		Imgproc.cvtColor(immat, immat, Imgproc.COLOR_RGB2GRAY);
		Imgproc.morphologyEx(immat, immat, Imgproc.MORPH_OPEN, matEllipse, new Point(3, 3), 1);// moved

		Imgproc.threshold(immat, immat, 1, 256, Imgproc.THRESH_BINARY);

		immat.convertTo(immat, 0);

		points.clear();
		hierarchy2 = new Mat();

		Imgproc.findContours(immat, points, hierarchy2, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
		markersx = Mat.zeros(immat.size(), CvType.CV_32SC3);

		ArrayList<String> color = new ArrayList<>(), c = new ArrayList<>();
		for (int i = 0; i < points.size(); i++) {
			double[] v = asc.get(i).val;
			String sv = v[0] + "|" + v[1] + "|" + v[2];
			if (color.contains(sv))
				System.out.println("CONT " + sv);
			else
				c.add(sv);
			color.add(sv);
			Imgproc.drawContours(markersx, points, i, asc.get(i), -1);
		}

		mm.clear();
		x = 0;
		for (int r = 0; r < markersx.rows(); r++) {
			for (int cc = 0; cc < markersx.cols(); cc++) {

				int[] data = new int[3];
				markersx.get(r, cc, data);

				if (data[0] == 0 && data[1] == 0 && data[2] == 0) {
					x++;
					continue;
				}
				String pix = data[0] + "|" + data[1] + "|" + data[2];
				if (mm.containsKey(pix))
					mm.put(pix, mm.get(pix) + 1);
				else
					mm.put(pix, 1);
			}

		}
		ArrayList<String> keys = new ArrayList<>();
		for (String k : mm.keySet()) {
			if (mm.get(k) < 18)
				keys.add(k);
		}
		for (String k : keys) {
			x += mm.get(k);
			mm.remove(k);
		}

		noOfCells = mm.keySet().size();
		sizes = new ArrayList<>(mm.values());
		Collections.sort(sizes);

		int median = sizes.size() % 2 == 0 ? (sizes.get(sizes.size() / 2) + sizes.get(sizes.size() / 2 - 1)) / 2
				: sizes.get(sizes.size() / 2);

		int all = 0;
		for (int i : sizes)
			all += i;
		int siz = (int) ((double) all / (double) median);
		System.out.println(all + " Size " + siz + ", " + noOfCells);
		SIZE = siz;

		return markersx;

	}

	public static BufferedImage matToBufferedImage(Mat matrix, BufferedImage bimg) {
		if (matrix != null) {
			int cols = matrix.cols();
			int rows = matrix.rows();
			int elemSize = (int) matrix.elemSize();
			byte[] data = new byte[cols * rows * elemSize];
			int type;
			matrix.get(0, 0, data);
			switch (matrix.channels()) {
			case 1:
				type = BufferedImage.TYPE_BYTE_GRAY;
				break;
			case 3:
				type = BufferedImage.TYPE_3BYTE_BGR;
				// bgr to rgb
				byte b;
				for (int i = 0; i < data.length; i = i + 3) {
					b = data[i];
					data[i] = data[i + 2];
					data[i + 2] = b;
				}
				break;
			default:
				return null;
			}

			if (bimg == null || bimg.getWidth() != cols || bimg.getHeight() != rows || bimg.getType() != type) {
				bimg = new BufferedImage(cols, rows, type);
			}
			bimg.getRaster().setDataElements(0, 0, cols, rows, data);
		} else {
			bimg = null;
		}
		return bimg;
	}

	private Image mat2Image(Mat frame) {

		MatOfByte buffer = new MatOfByte();
		Imgcodecs.imencode(".png", frame, buffer);
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}

	private Mat setColors(Mat m) {
		ArrayList<Scalar> asc = new ArrayList<>();

		for (int i = 10; i < 250; i++)
			for (int j = 0; j < 250; j++)
				asc.add(new Scalar(i, j, i));
		Collections.shuffle(asc);
		m.convertTo(m, CvType.CV_8SC3);
		BufferedImage imm = new BufferedImage(m.width(), m.height(), BufferedImage.TYPE_3BYTE_BGR);

		System.out.println("m " + m.size());
		System.out.println("w " + imm.getWidth());

		int color = -1;
		int[][] abc = new int[m.rows()][m.cols()];

		m.convertTo(m, CvType.CV_32SC3);
		for (int r = 0; r < m.rows(); r++) {
			for (int cc = 0; cc < m.cols(); cc++) {

				int[] data = new int[3];
				m.get(r, cc, data);

				if (data[0] == 0 && data[1] == 0 && data[2] == 0) {
					abc[r][cc] = 1;

				} else
					abc[r][cc] = 0;

			}
		}

		for (int i = 0; i < abc.length; i++) {
			for (int j = 0; j < abc[0].length; j++) {
				if (abc[i][j] == 1) {
					if (i == 0 && j == 0) {

						abc[i][j] = color;
					} else if (i == 0) {
						if (abc[i][j - 1] != 0) {
							abc[i][j] = color;
						}
					} else if (j == 0) {
						if (abc[i - 1][j] != 0) {
							abc[i][j] = color;
						}

						if (i + 1 < abc.length && abc[i + 1][j] != 0) {
							abc[i][j] = color;
						}
					} else if ((j + 1 < abc[0].length && i - 1 >= 0 && j - 1 >= 0) && (abc[i - 1][j - 1] != 0
							|| abc[i - 1][j] != 0 || abc[i - 1][j + 1] != 0 || abc[i][j - 1] != 0)) {
						abc[i][j] = color;

					} else {
						abc[i][j] = --color;
					}
				}
			}
		}
		boolean zmiana = true;
		int c = 0;
		while (zmiana && c < 50) {
			c++;
			// System.out.println("ZMIANA");
			zmiana = false;
			for (int i = 0; i < abc.length; i++) {
				for (int j = 0; j < abc[0].length; j++) {
					if (abc[i][j] != 0) {
						if (j + 1 < abc[0].length && abc[i][j + 1] != 0 && abc[i][j + 1] > abc[i][j]) {
							abc[i][j] = abc[i][j + 1];

							zmiana = true;
						} else {
							if (j + 1 < abc[0].length && abc[i][j + 1] != 0 && abc[i][j + 1] < abc[i][j]) {
								abc[i][j + 1] = abc[i][j];

								zmiana = true;
							}
							if (i + 1 < abc.length && j - 1 >= 0 && abc[i + 1][j - 1] != 0
									&& abc[i + 1][j - 1] < abc[i][j]) {
								abc[i + 1][j - 1] = abc[i][j];
								zmiana = true;
							}
							if (i + 1 < abc.length && abc[i + 1][j] != 0 && abc[i + 1][j] < abc[i][j]) {
								abc[i + 1][j] = abc[i][j];
								zmiana = true;
							}
							if (i + 1 < abc.length && j + 1 < abc[0].length && abc[i + 1][j + 1] != 0
									&& abc[i + 1][j + 1] < abc[i][j]) {
								abc[i + 1][j + 1] = abc[i][j];
								zmiana = true;
							}
						}
					}
				}
			}

		}

		int[] pixels2 = new int[imm.getWidth() * imm.getHeight()];
		int count = 0;
		for (int i = 0; i < abc.length; i++) {
			for (int j = 0; j < abc[0].length; j++) {
				if (Math.abs(abc[i][j]) == 0) {
					pixels2[count++] = 0;
					continue;
				}
				Scalar ss = asc.get(Math.abs(abc[i][j]));
				double[] d = ss.val;
				int rgb = (int) d[0];
				rgb = (rgb << 8) + (int) d[1];
				rgb = (rgb << 8) + (int) d[2];
				pixels2[count++] = rgb;

			}
		}
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels2, 0, imm.getWidth());

		m = bufferedImageToMat(imm);
		System.out.println("ROWS " + m.rows());
		HashMap<String, Integer> mm = new HashMap<>();
		for (int i = 0; i < abc.length; i++) {
			for (int j = 0; j < abc[0].length; j++) {
				if (abc[i][j] == 0)
					continue; // 0
				String pix = "" + abc[i][j];
				if (mm.containsKey(pix))
					mm.put(pix, mm.get(pix) + 1);
				else
					mm.put(pix, 1);
			}
		}

		//
		ArrayList<String> keys = new ArrayList<>();
		for (String k : mm.keySet()) {
			System.out.println("RES " + 0.02 * m.rows());
			if (mm.get(k) < 0.03 * m.rows())// 1536
				keys.add(k);
		}
		for (String k : keys) {
			// x += mm.get(k);
			mm.remove(k); // TODO: odkomentuj
		}
		//

		// for (String s : mm.keySet())
		// System.out.println("ss " + s + " -> " + mm.get(s));
		System.out.println("mm s " + mm.size());
		ArrayList<Integer> sizes = new ArrayList<>(mm.values());
		Collections.sort(sizes);
		int median = sizes.size() % 2 == 0 ? (sizes.get(sizes.size() / 2) + sizes.get(sizes.size() / 2 - 1)) / 2
				: sizes.get(sizes.size() / 2);
		System.out.println(sizes.size() + " " + median);
		int all = 0;
		for (int i : sizes)
			all += i;
		int siz = (int) ((double) all / (double) median);
		System.out.println("Size " + siz);
		SIZE = siz;

		// SIZE = mm.keySet().size();
		return m;
	}

}
