import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
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
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;

/**
 * 
 * 
 * @author Aleksandra Do��ga
 * 
 */
public class Controller {

	public static final int KMEANS = 0;
	public static final int WATERSHED = 1;
	public static final int THRESHOLDING = 2;
	public static final int CANNY = 3;
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
	BufferedImage redImage;

	private Mat redMat;
	boolean debug = false;

	Color CYAN = new Color(0, 255, 255);
	Color BLUE = new Color(0, 0, 255);
	double H, DAB;
	int SIZE;
	private double cells;
	private Color actualColor;
	public static int SEGMENTATION_TYPE;
	public static String redStainsFilename, blueStainsFilename;
	public static String infoPic;
	public static String densityMap;
	public static String filename;

	private Mat doBackgroundRemoval(Mat frame) {
		// init
		Mat hsvImg = new Mat();
		List<Mat> hsvPlanes = new ArrayList<>();
		Mat thresholdImg = new Mat();

		int thresh_type = Imgproc.THRESH_BINARY_INV;
		// if (this.inverse.isSelected())
		thresh_type = Imgproc.THRESH_BINARY;

		// threshold the image with the average hue value
		hsvImg.create(frame.size(), CvType.CV_8U);
		Imgproc.cvtColor(frame, hsvImg, Imgproc.COLOR_BGR2HSV);
		Core.split(hsvImg, hsvPlanes);

		// get the average hue value of the image
		double threshValue = this.getHistAverage(hsvImg, hsvPlanes.get(0));
		System.out.println("thre " + threshValue);

		Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, thresh_type);

		Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));

		// dilate to fill gaps, erode to smooth edges
		Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 1);
		Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);

		Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);

		// create the new image
		Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
		frame.copyTo(foreground, thresholdImg);

		return foreground;
	}

	private double getHistAverage(Mat hsvImg, Mat hueValues) {
		// init
		double average = 0.0;
		Mat hist_hue = new Mat();
		// 0-180: range of Hue values
		MatOfInt histSize = new MatOfInt(180);
		List<Mat> hue = new ArrayList<>();
		hue.add(hueValues);

		// compute the histogram
		Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));

		// get the average Hue value of the image
		// (sum(bin(h)*h))/(image-height*image-width)
		// -----------------
		// equivalent to get the hue of each pixel in the image, add them, and
		// divide for the image size (height and width)
		for (int h = 0; h < 180; h++) {
			// for each bin, get its value and multiply it for the corresponding
			// hue
			average += (hist_hue.get(h, 0)[0] * h);
		}

		// return the average hue of the image
		return average = average / hsvImg.size().height / hsvImg.size().width;
	}

	private void showHistogram(Mat frame, boolean gray) {
		// split the frames in multiple images
		List<Mat> images = new ArrayList<>();
		Core.split(frame, images);

		// set the number of bins at 256
		MatOfInt histSize = new MatOfInt(256);
		// only one channel
		MatOfInt channels = new MatOfInt(0);
		// set the ranges
		MatOfFloat histRange = new MatOfFloat(0, 256);

		// compute the histograms for the B, G and R components
		Mat hist_b = new Mat();
		Mat hist_g = new Mat();
		Mat hist_r = new Mat();

		// B component or gray image
		Imgproc.calcHist(images.subList(0, 1), channels, new Mat(), hist_b, histSize, histRange, false);

		// G and R components (if the image is not in gray scale)
		if (!gray) {
			Imgproc.calcHist(images.subList(1, 2), channels, new Mat(), hist_g, histSize, histRange, false);
			Imgproc.calcHist(images.subList(2, 3), channels, new Mat(), hist_r, histSize, histRange, false);
		}

		// draw the histogram
		int hist_w = 150; // width of the histogram image
		int hist_h = 150; // height of the histogram image

		int bin_w = (int) Math.round(hist_w / histSize.get(0, 0)[0]);
		System.out.println("binw " + bin_w);
		Mat histImage = new Mat(hist_h, hist_w, CvType.CV_8UC3, new Scalar(0, 0, 0));
		// normalize the result to [0, histImage.rows()]
		Core.normalize(hist_b, hist_b, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());

		// for G and R components
		if (!gray) {
			Core.normalize(hist_g, hist_g, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
			Core.normalize(hist_r, hist_r, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
		}

		// effectively draw the histogram(s)
		for (int i = 1; i < histSize.get(0, 0)[0]; i++) {
			// B component or gray image
			// i = 220;
			System.out.println("::: " + Math.round(hist_b.get(i, 0)[0]));
			Imgproc.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_b.get(i - 1, 0)[0])),
					new Point(bin_w * (i), hist_h - Math.round(hist_b.get(i, 0)[0])), new Scalar(255, 255, 255), 2, 8,
					0);
			// G and R components (if the image is not in gray scale)
			if (!gray) {
				Imgproc.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_g.get(i - 1, 0)[0])),
						new Point(bin_w * (i), hist_h - Math.round(hist_g.get(i, 0)[0])), new Scalar(0, 255, 0), 2, 8,
						0);
				Imgproc.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_r.get(i - 1, 0)[0])),
						new Point(bin_w * (i), (hist_h - Math.round(hist_r.get(i, 0)[0]))), new Scalar(0, 0, 255), 2, 8,
						0);
			}
			// break;
		}

		// display the histogram...
		Image histImg = mat2Image(histImage);
		// this.originalFrame3.setImage(histImg);

	}

	private BufferedImage countOnes(BufferedImage imm, Color color) {
		int[] pixels = new int[imm.getWidth() * imm.getHeight()];
		imm.getRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		// System.out.println("pix " + pixels.length + ",");
		int xx = 0;
		for (int i = 0; i < pixels.length; i++) {

			int rgb = pixels[i];
			int r = (rgb >> 16) & 0xFF;
			int g = (rgb >> 8) & 0xFF;
			int b = (rgb & 0xFF);

			// System.out.println("rgb " + r + " " + g + " " + b);
			if (!(r == color.getRed() && g == color.getGreen() && b == color.getBlue())) {
				pixels[i] = 0xFF000000;
				// r = 0;
				// b = 0;
				// g = 0;
			} else
				xx++;// System.out.println("ccccccccc");

		}
		System.out.println(color.getRed() + ", " + color.getGreen() + ", " + color.getBlue() + " CCC " + xx);
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());
		/*
		 * WritableImage wr = null; if (imm != null) { wr = new
		 * WritableImage(imm.getWidth(), imm.getHeight()); PixelWriter pw =
		 * wr.getPixelWriter(); for (int x = 0; x < imm.getWidth(); x++) { for
		 * (int y = 0; y < imm.getHeight(); y++) { pw.setArgb(x, y,
		 * imm.getRGB(x, y)); } } }
		 */
		return imm;// wr;
	}

	@FXML
	protected void load() {
		int width = 500, width2 = 0;// 500 300
		if (!debug) {
			originalFrame.setFitWidth(width);
			originalFrame2.setFitWidth(width);
			originalFrame3.setFitWidth(width2);
			// originalFrame4.setFitWidth(width2);
			// originalFrame5.setFitWidth(width2);
			originalFrame.setPreserveRatio(true);
			originalFrame2.setPreserveRatio(true);
			originalFrame3.setPreserveRatio(true);
			// originalFrame4.setPreserveRatio(true);
			// originalFrame5.setPreserveRatio(true);
		}

		Image imageToShow;
		actualColor = Color.RED;

		imageToShow = getFrame(redStainsFilename);
		printTruePositive(imageToShow, actualColor);
		int SIZ_COLOR1 = SIZE;
		if (!debug) {
			originalFrame.setImage(imageToShow);

		}

		saveToFile(imageToShow, "C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\SEGMENTED\\" + filename
				+ "_RED_KMEANS.png");
		actualColor = Color.BLUE;

		imageToShow = getFrame(blueStainsFilename);
		printTruePositive(imageToShow, actualColor);
		countGroundTruth();
		int SIZ_COLOR2 = SIZE;

		double s = (double) SIZ_COLOR1 / (double) SIZ_COLOR2;
		System.out.println(">>> Red stains: " + SIZ_COLOR1 + ", blue stains: " + SIZ_COLOR2 + ", result = " + s);
		if (!debug) {
			originalFrame2.setImage(imageToShow);

		}
		saveToFile(imageToShow, "C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\SEGMENTED\\" + filename
				+ "_BKUE_KMEANS.png");

	}

	public static void saveToFile(Image image, String filename) {
		File outputFile = new File(filename);
		outputFile.getParentFile().mkdirs();
		System.out.println("f " + filename);
		BufferedImage bImage = SwingFXUtils.fromFXImage(image, null);
		try {
			outputFile.createNewFile();
			ImageIO.write(bImage, "png", outputFile);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private void printTruePositive(Image imageToShow, Color actualColor2) {
		if (actualColor2 == CYAN) {

		} else {

		}

	}

	private void countGroundTruth() {
		BufferedImage bi = null, bi2 = null;
		try {
			bi = ImageIO.read(new File(infoPic));//
			bi2 = ImageIO.read(new File(infoPic));//

		} catch (IOException e) {
			System.out.println("Err " + e);
		}

		H = 0;
		DAB = 0;
		cells = 0;
		BufferedImage b2 = countOnes(bi, CYAN);
		setColors(b2, CYAN);
		H = cells;
		b2 = countOnes(bi2, BLUE);
		setColors(b2, BLUE);
		H += cells;
		DAB = cells;
		System.out.println(filename + "INFO RESULT: " + DAB + ", " + H + " = " + (DAB / H));

		String text = filename + ";" + DAB + ";" + H + ";" + (DAB / H) + "\n";
		// File summary=new File("C:\\Users\\Olusiak\\Downloads\\Ki67
		// baza-20161012T160623Z\\pods.txt");
		String file = "C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\pods.txt";
		try {
			Files.write(Paths.get(file), text.getBytes(), StandardOpenOption.APPEND);

		} catch (IOException e) {
			// exception handling left as an exercise for the reader
		}

	}

	private Image getFrame(String tit) {
		System.out.println("INS");
		BufferedImage bi = null;
		try {
			bi = ImageIO.read(new File(tit));//

			BufferedImage before = bi;

			int w = before.getWidth() / 4;
			int h = before.getHeight() / 4;

			BufferedImage resizedImg = new BufferedImage(w, h, bi.getType());
			Graphics2D g2 = resizedImg.createGraphics();
			g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			g2.drawImage(bi, 0, 0, w, h, null);
			System.out.println("before " + before.getWidth() + " " + w);
			g2.dispose();

			// BufferedImage after = new BufferedImage(w, h, bi.getType());
			// AffineTransform at = new AffineTransform();
			// at.scale(0.5, 0.5);
			// AffineTransformOp scaleOp = new AffineTransformOp(at,
			// AffineTransformOp.TYPE_BILINEAR);
			// after = scaleOp.filter(before, after);
			bi = resizedImg;

		} catch (IOException e) {
			System.out.println("Err " + e);
		}

		Image imageToShow = null;

		Mat frame = bufferedImageToMat(bi);

		try {

			if (!frame.empty()) {
				if (SEGMENTATION_TYPE == Controller.KMEANS) {
					frame = this.segmentationKMeans(frame);
					redMat = frame;
				} else if (SEGMENTATION_TYPE == Controller.WATERSHED)
					frame = this.segmentationWatershed(frame);
				else if (SEGMENTATION_TYPE == Controller.THRESHOLDING)
					frame = this.segmentationThresholdingTest(frame);
				else
					frame = this.segmentationCanny(frame);
				// frame = putPoints(frame);

				imageToShow = mat2Image(frame);

				// bi = countOnes(bi, BLUE);

				// Mat m = bufferedImageToMat(bi);
				// imageToShow = setColors(bi, BLUE);

				// imageToShow = mat2Image(m);

				WritableImage wr = null;
				if (bi != null) {
					wr = new WritableImage(bi.getWidth(), bi.getHeight());
					PixelWriter pw = wr.getPixelWriter();
					for (int x = 0; x < bi.getWidth(); x++) {
						for (int y = 0; y < bi.getHeight(); y++) {
							pw.setArgb(x, y, bi.getRGB(x, y));
						}
					}
				}
				// imageToShow = wr;
			}

		} catch (Exception e) {

			System.err.print("ERROR");
			e.printStackTrace();
		}

		return imageToShow;
	}

	private Mat putPoints(Mat frame) {

		File pic = new File(densityMap);
		System.out.println("DEN " + densityMap);
		BufferedImage img = null;
		try {
			img = ImageIO.read(pic);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			// throw e;
		}

		// BufferedImage after = new BufferedImage(img.getWidth(),
		// img.getHeight(), BufferedImage.TYPE_INT_ARGB);
		// AffineTransform at = new AffineTransform();
		// at.scale(-0.25, -0.25);
		// AffineTransformOp scale = new AffineTransformOp(at,
		// AffineTransformOp.TYPE_BILINEAR);
		// img = scale.filter(img, after);
		byte[] pixelsb = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
		Mat matImg = new Mat(img.getHeight(), img.getWidth(), CvType.CV_8UC3);
		matImg.put(0, 0, pixelsb);

		Imgproc.resize(matImg, matImg, new Size(frame.rows(), frame.cols()));

		System.out.println("W " + matImg.cols() + ", " + matImg.rows());
		int[] pixels = new int[img.getWidth() * img.getHeight()];
		int count = 0;
		frame.convertTo(frame, CvType.CV_32SC3);
		matImg.convertTo(matImg, CvType.CV_32SC3);
		originalFrame.setImage(mat2Image(matImg));
		for (int i = 0; i < frame.rows(); i++) {
			for (int j = 0; j < frame.cols(); j++) {
				int[] data = new int[3];
				matImg.get(i, j, data);// pixels[i * frame.rows() + j];

				int r = data[0];// (rgb >> 16) & 0xFF;
				int g = data[1];// (rgb >> 8) & 0xFF;
				int b = data[2];// (rgb & 0xFF);
				// System.out.println("rgb " + r + " " + g + " " + b);
				// System.out.println("rgb " + r + " " + g + " " + b);
				if ((r == CYAN.getRed() && g == CYAN.getGreen() && b == CYAN.getBlue())
						|| (r == BLUE.getRed() && g == BLUE.getGreen() && b == BLUE.getBlue())) {
					System.out.println("put");
					frame.put(i, j, data);
				}

			}
		}
		return frame;

	}

	private Mat segmentationThresholding(Mat frame) {

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

		Imgproc.threshold(detectedEdges, detectedEdges, 160, 255, Imgproc.THRESH_TRUNC);// 60
		double[] s = detectedEdges.get(0, 0);
		System.out.println(s[0] + " : " + s.length);// + s[1] + " : " + s[2]);
		Imgproc.threshold(detectedEdges, detectedEdges, 255 - s[0], 255, Imgproc.THRESH_BINARY);// 60
		// Imgproc.adaptiveThreshold(detectedEdges, detectedEdges, 255,
		// Imgproc.ADAPTIVE_THRESH_MEAN_C, // ADAPTIVE_THRESH_GAUSSIAN_C,
		// Imgproc.THRESH_BINARY, 11, 2);
		System.out.println("===");
		// Imgproc.HoughCircles(detectedEdges, detectedEdges,
		// Imgproc.HOUGH_GRADIENT, 1, 100);// dp,
		// minDist);
		// Imgproc.erode(detectedEdges, detectedEdges, matEllipse);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_CLOSE, matEllipse);// OPEN???
		/*
		 * Imgproc.erode(detectedEdges, detectedEdges, matEllipse); //
		 * Imgproc.morphologyEx(detectedEdges, detectedEdges, //
		 * Imgproc.MORPH_CLOSE, matEllipse);
		 */
		detectedEdges = setColors(detectedEdges);
		return detectedEdges;

	}

	private Mat segmentationThresholdingTest(Mat frame) {

		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat(frame.size(), CvType.CV_8UC3);
		// detectedEdges = doBackgroundRemoval(frame);

		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

		Core.normalize(grayImage, detectedEdges, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC3);

		// showHistogram(detectedEdges, true);
		// if (true)
		// return frame;
		Mat matEllipse = Mat.ones(new Size(5, 5), 0);
		Mat matEllipse2 = Mat.ones(new Size(5, 5), 0);
		Mat matt = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(6, 6));// cvCreateStructuringElementEx
		matEllipse.put(0, 0, 0);
		matEllipse.put(0, 1, 0);
		matEllipse.put(0, 3, 0);
		matEllipse.put(0, 4, 0);
		matEllipse.put(4, 0, 0);
		matEllipse.put(4, 1, 0);
		matEllipse.put(4, 3, 0);
		matEllipse.put(4, 4, 0);

		Imgproc.threshold(detectedEdges, detectedEdges, 115, 255, Imgproc.THRESH_OTSU);// 60
		if (true)
			return detectedEdges;
		double[] s = detectedEdges.get(0, 0);
		System.out.println(s[0] + " : " + s.length);// + s[1] + " : " + s[2]);
		// Imgproc.threshold(detectedEdges, detectedEdges, 255 - s[0], 255,
		// Imgproc.THRESH_OTSU);// 60
		// Imgproc.adaptiveThreshold(detectedEdges, detectedEdges, 255,
		// Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
		// Imgproc.THRESH_BINARY, 33, 2);
		if (true)
			return detectedEdges;
		// System.out.println("RED mat " + redMat);
		// if (actualColor == Color.BLUE && redMat != null)
		// Core.bitwise_and(detectedEdges, redMat, detectedEdges);

		// System.out.println("===");
		// Imgproc.HoughCircles(detectedEdges, detectedEdges,
		// Imgproc.HOUGH_GRADIENT, 1, 100);// dp,
		// minDist);

		Imgproc.morphologyEx(detectedEdges, detectedEdges, Imgproc.MORPH_CLOSE, matt);
		// Imgproc.dilate(detectedEdges, detectedEdges, matEllipse2);// , new
		// Point(-1,
		// 1), 3);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_CLOSE, matEllipse);// OPEN???
		/*
		 * Imgproc.erode(detectedEdges, detectedEdges, matEllipse); //
		 * Imgproc.morphologyEx(detectedEdges, detectedEdges, //
		 * Imgproc.MORPH_CLOSE, matEllipse);
		 */
		Mat blackWhite = new Mat();
		detectedEdges.copyTo(blackWhite);
		if (actualColor == Color.RED)
			redMat = blackWhite;
		else
			Core.bitwise_and(detectedEdges, redMat, detectedEdges);
		Imgproc.morphologyEx(detectedEdges, detectedEdges, Imgproc.MORPH_CLOSE, matt);
		detectedEdges = setColors(detectedEdges);
		return detectedEdges;

	}

	private Mat segmentationThresholdingBinary(Mat frame) {

		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat(frame.size(), CvType.CV_8UC3);

		// Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2HSV);

		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
		// Imgproc.cvtColor(frame, frame, Imgproc.COLOR_GRAY2BGR);
		// if (true)
		// return frame;
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
		// Imgproc.adaptiveThreshold(detectedEdges, detectedEdges, 255,
		// Imgproc.ADAPTIVE_THRESH_MEAN_C, // ADAPTIVE_THRESH_GAUSSIAN_C,
		// Imgproc.THRESH_BINARY, 11, 2);
		// System.out.println("===");
		// Imgproc.HoughCircles(detectedEdges, detectedEdges,
		// Imgproc.HOUGH_GRADIENT, 1, 100);// dp,
		// minDist);
		// Imgproc.erode(detectedEdges, detectedEdges, matEllipse);
		Imgproc.morphologyEx(detectedEdges, detectedEdges, Imgproc.MORPH_CLOSE, matEllipse);// OPEN???
		/*
		 * Imgproc.erode(detectedEdges, detectedEdges, matEllipse); //
		 * Imgproc.morphologyEx(detectedEdges, detectedEdges, //
		 * Imgproc.MORPH_CLOSE, matEllipse);
		 */
		detectedEdges = setColors(detectedEdges);
		return detectedEdges;

	}

	private Mat segmentationCanny(Mat frame) {
		// init
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat();

		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

		// reduce noise with a 3x3 kernel
		Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

		// canny detector, with ratio of lower:upper threshold of 3:1
		Imgproc.Canny(detectedEdges, detectedEdges, 100, 300);// ?
																// this.threshold.getValue(),
																// this.threshold.getValue()
																// * 3);

		// using Canny's output as a mask, display the result
		Mat dest = new Mat();
		frame.copyTo(dest, detectedEdges);

		return dest;
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

		// Imgproc.GaussianBlur(detectedEdges, detectedEdges, new Size(17, 17),
		// 0);//odk

		Mat inv = new Mat();
		detectedEdges.copyTo(inv);

		Imgproc.threshold(inv, inv, 160, 255, Imgproc.THRESH_BINARY_INV);

		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.HOUGH_GRADIENT, matEllipse);

		Imgproc.morphologyEx(inv, inv, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));

		Imgproc.threshold(detectedEdges, detectedEdges, 130, 255, Imgproc.THRESH_BINARY);

		Mat inv3 = new Mat();
		detectedEdges.copyTo(inv3);
		Core.bitwise_not(inv3, inv3);

		// Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3,
		// CvType.CV_8UC1), new Point(1, 1), 3);
		Imgproc.morphologyEx(inv, inv, Imgproc.MORPH_OPEN, matEllipse);// inv3,inv3

		// if (true)
		// return inv3;
		Mat fin = new Mat();
		inv.copyTo(fin);// inv3

		fin.convertTo(fin, CvType.CV_8U);

		Mat cont = new Mat();

		fin.copyTo(cont);// by�o 3/2

		final List<MatOfPoint> points = new ArrayList<>();
		final Mat hierarchy = new Mat();

		Imgproc.findContours(cont, points, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
		Mat markers = Mat.zeros(frame.size(), CvType.CV_32SC1);

		for (int i = 0; i < points.size(); i++) {
			Scalar as = asc.get(i);

			Imgproc.drawContours(markers, points, i, as, -1);
		}
		// if (true)
		// return markers;
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
		Imgproc.morphologyEx(immat, gr, Imgproc.MORPH_GRADIENT, Mat.eye(3, 3, 0));// matEllipse);//
																					// ,
		// new
		// Point(1,
		// 1),
		// 1);
		// if (true)
		// return gr;
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

		// System.out.println();
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

		// originalFrame3.setImage(mat2Image(arrs.get(0)));
		// originalFrame4.setImage(mat2Image(arrs.get(1)));
		// originalFrame5.setImage(mat2Image(arrs.get(2)));
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
				if (mm.get(k) < 10)
					keys.add(k);
			}
			for (String k : keys) {
				x += mm.get(k);
				mm.remove(k);
			}

			noOfCells = mm.keySet().size();
			// System.out.println(">>: " + noOfCells);
			sizes = new ArrayList<>(mm.values());
			Collections.sort(sizes);

			if (sizes.get(sizes.size() - 1) <= bestVal) {
				bestVal = sizes.get(sizes.size() - 1);

				bestIndex = iar;
			}
		}

		best = arrs.get(bestIndex);

		// originalFrame3.setImage(mat2Image(arrs.get(0)));
		// originalFrame4.setImage(mat2Image(arrs.get(1)));
		// originalFrame5.setImage(mat2Image(arrs.get(2)));

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

		if (actualColor == Color.BLUE) {
			System.out.println("MARKERS " + markersx.cols() + ", " + redMat.cols());
			for (int r = 0; r < markersx.rows(); r++) {
				for (int cc = 0; cc < markersx.cols(); cc++) {

					int[] data = new int[3];
					int[] data2 = new int[3];
					markersx.get(r, cc, data);
					redMat.get(r, cc, data2);
					// System.out.println("D " + data[0] + " " + data[1] + " " +
					// data[2] + " . " + data2[0] + " "
					// + data2[1] + " " + data2[2]);
					if ((data[0] == 0 && data[1] == 0 && data[2] == 0)
							&& !(data2[0] == 0 && data[1] == 0 && data[2] == 0)) {
						markersx.put(r, cc, data2);
					}
				}
			}
			// originalFrame.setImage(mat2Image(markersx));
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

		System.out.println("===");

		// for (String k : keys) {
		// x += mm.get(k);
		// mm.remove(k);
		// }

		int max = 0;
		// String key=null;
		for (String k : mm.keySet()) {
			if (mm.get(k) >= max) {
				max = mm.get(k);
				// key=k;
			}

			// System.out.println(":: " + k + " = " + mm.get(k));
			if (mm.get(k) < 10)
				// mm.remove(k);
				keys.add(k);
		}
		for (String k : keys)
			mm.remove(k);

		int places = (int) Math.ceil(max / 10.0);
		HashMap<Integer, Integer> dict = new HashMap<>();

		for (String k : mm.keySet()) {
			int val = mm.get(k);
			// System.out.println("v " + val);
			int place = (int) (val / 10.0);
			// System.out.println("pl" + place);
			if (dict.containsKey(place))
				dict.put(place, dict.get(place) + 1);
			else
				dict.put(place, 1);
		}

		max = 0;
		int key = -1;
		for (int k : dict.keySet()) {
			if (dict.get(k) >= max) {
				max = dict.get(k);
				key = k;
			}
		}

		noOfCells = mm.keySet().size();
		sizes = new ArrayList<>(mm.values());
		Collections.sort(sizes);

		int median = sizes.size() % 2 == 0 ? (sizes.get(sizes.size() / 2) + sizes.get(sizes.size() / 2 - 1)) / 2
				: sizes.get(sizes.size() / 2);
		System.out.println("MEDIAN " + median + ", " + key);
		// median = key * 10 + 5;// DODANE
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
		// originalFrame3.setImage(mat2Image(m));

		ArrayList<Scalar> asc = new ArrayList<>();

		for (int i = 10; i < 250; i++)
			for (int j = 0; j < 250; j++)
				asc.add(new Scalar(i, j, i));
		Collections.shuffle(asc);
		m.convertTo(m, CvType.CV_8SC3);
		BufferedImage imm = new BufferedImage(m.width(), m.height(), BufferedImage.TYPE_3BYTE_BGR);

		// System.out.println("m " + m.size());
		// System.out.println("w " + imm.getWidth());

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
							abc[i][j] = abc[i][j - 1];// color;
						}
					} else if (j == 0) {
						if (abc[i - 1][j] != 0) {
							abc[i][j] = abc[i - 1][j];// color;
						}

						if (i + 1 < abc.length && abc[i + 1][j] != 0) {
							abc[i][j] = abc[i + 1][j];// color;
						}
					} else if ((j + 1 < abc[0].length && i - 1 >= 0 && j - 1 >= 0) && (abc[i - 1][j - 1] != 0
							|| abc[i - 1][j] != 0 || abc[i - 1][j + 1] != 0 || abc[i][j - 1] != 0)) {

						if (abc[i - 1][j - 1] != 0)
							abc[i][j] = abc[i - 1][j - 1];
						else if (abc[i - 1][j] != 0)
							abc[i][j] = abc[i - 1][j];
						else if (abc[i - 1][j + 1] != 0)
							abc[i][j] = abc[i - 1][j + 1];
						else if (abc[i][j - 1] != 0)
							abc[i][j] = abc[i][j - 1];
						// abc[i][j] = color;

					} else {
						abc[i][j] = --color;
					}
				}
			}
		}
		boolean zmiana = true;
		int c = 0;
		while (zmiana && c < 15) {
			c++;
			// System.out.println("CCC " + c);
			ArrayList<Integer> keys = new ArrayList<>();
			ArrayList<Integer> vals = new ArrayList<>();
			HashMap<Integer, Integer> tab = new HashMap<>();
			// System.out.println("ZMIANA");
			zmiana = false;
			for (int i = 0; i < abc.length; i++) {
				for (int j = 0; j < abc[0].length; j++) {
					if (abc[i][j] != 0) {
						if (j + 1 < abc[0].length && abc[i][j + 1] != 0 && abc[i][j + 1] > abc[i][j]) {
							// abc[i][j] = abc[i][j + 1];
							// ==
							if (!keys.contains(abc[i][j])) {
								keys.add(abc[i][j]);
								vals.add(abc[i][j + 1]);
							}
							tab.put(abc[i][j], abc[i][j + 1]);

							// for (int i2 = 0; i2 < i; i2++)
							// for (int j2 = 0; j2 < abc[0].length; j2++)
							// if (abc[i2][j2] == abc[i][j])
							// abc[i2][j2] = abc[i][j + 1];

							// =
							zmiana = true;
						} else {
							if (j + 1 < abc[0].length && abc[i][j + 1] != 0 && abc[i][j + 1] < abc[i][j]) {
								// abc[i][j + 1] = abc[i][j];
								// ==
								tab.put(abc[i][j + 1], abc[i][j]);
								if (!keys.contains(abc[i][j + 1])) {
									keys.add(abc[i][j + 1]);
									vals.add(abc[i][j]);
								}
								// for (int i2 = 0; i2 < i; i2++)
								// for (int j2 = 0; j2 < abc[0].length; j2++)
								// if (abc[i2][j2] == abc[i][j + 1])
								// abc[i2][j2] = abc[i][j];
								/// =
								zmiana = true;
							}
							if (i + 1 < abc.length && j - 1 >= 0 && abc[i + 1][j - 1] != 0
									&& abc[i + 1][j - 1] < abc[i][j]) {
								// abc[i + 1][j - 1] = abc[i][j];
								// ==
								tab.put(abc[i + 1][j - 1], abc[i][j]);

								if (!keys.contains(abc[i + 1][j - 1])) {
									keys.add(abc[i + 1][j - 1]);
									vals.add(abc[i][j]);
								}
								// for (int i2 = 0; i2 < i; i2++)
								// for (int j2 = 0; j2 < abc[0].length; j2++)
								// if (abc[i2][j2] == abc[i + 1][j - 1])
								// abc[i2][j2] = abc[i][j];

								// =
								zmiana = true;
							}
							if (i + 1 < abc.length && abc[i + 1][j] != 0 && abc[i + 1][j] < abc[i][j]) {
								// abc[i + 1][j] = abc[i][j];
								// ==
								tab.put(abc[i + 1][j], abc[i][j]);
								if (!keys.contains(abc[i + 1][j])) {
									keys.add(abc[i + 1][j]);
									vals.add(abc[i][j]);
								}
								// for (int i2 = 0; i2 < i; i2++)
								// for (int j2 = 0; j2 < abc[0].length; j2++)
								// if (abc[i2][j2] == abc[i + 1][j])
								// abc[i2][j2] = abc[i][j];
								// =
								zmiana = true;
							}
							if (i + 1 < abc.length && j + 1 < abc[0].length && abc[i + 1][j + 1] != 0
									&& abc[i + 1][j + 1] < abc[i][j]) {
								// abc[i + 1][j + 1] = abc[i][j];
								// ==
								tab.put(abc[i + 1][j + 1], abc[i][j]);
								if (!keys.contains(abc[i + 1][j + 1])) {
									keys.add(abc[i + 1][j + 1]);
									vals.add(abc[i][j]);
								}
								// for (int i2 = 0; i2 < i; i2++)
								// for (int j2 = 0; j2 < abc[0].length; j2++)
								// if (abc[i2][j2] == abc[i + 1][j + 1])
								// abc[i2][j2] = abc[i][j];
								/// =
								zmiana = true;
							}
						}
					}
				}
			}
			Collections.reverse(keys);
			Collections.reverse(vals);
			tab.clear();
			// System.out.println("*** " + tab.size());
			for (int i2 = 0; i2 < abc.length; i2++)
				for (int j2 = 0; j2 < abc[0].length; j2++) {
					for (int tabb : tab.keySet()) {

						if (abc[i2][j2] == tabb)// abc[i + 1][j + 1])
							abc[i2][j2] = tab.get(tabb);
					}
					for (int k = 0; k < keys.size(); k++)
						if (abc[i2][j2] == keys.get(k))// abc[i + 1][j + 1])
							abc[i2][j2] = vals.get(k);
				}
			// System.out.println("///");

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
		// System.out.println("ROWS " + m.rows());
		HashMap<String, Integer> mm = new HashMap<>();
		for (int i = 0; i < abc.length; i++) {
			for (int j = 0; j < abc[0].length; j++) {
				if (abc[i][j] == 0)
					continue; // 0
				String pix = "" + abc[i][j];
				if (mm.containsKey(pix))
					mm.put(pix, mm.get(pix) + 1);
				else {
					mm.put(pix, 1);

				}
			}
		}

		//
		ArrayList<String> keys = new ArrayList<>();
		for (String k : mm.keySet()) {
			System.out.println("RES " + 0.03 * m.rows());
			if (mm.get(k) < 0.03 * m.rows())// 1536
				keys.add(k);
		}

		// ****
		// int[] pixels2 = new int[imm.getWidth() * imm.getHeight()];
		// int
		count = 0;
		for (int i = 0; i < abc.length; i++) {
			for (int j = 0; j < abc[0].length; j++) {

				if (Math.abs(abc[i][j]) == 0) {
					// pixels2[count++] = 0;
					count++;
					continue;
				}
				// Scalar ss = asc.get(Math.abs(abc[i][j]));
				// double[] d = ss.val;
				for (String k : keys) {
					// x += mm.get(k);
					int col = Integer.parseInt(k);
					if (col == abc[i][j]) {
						pixels2[count] = 0;
						break;
					}
					// mm.remove(k);// TODO odkomentowac?
				}
				// int rgb = 0;//(int) d[0];
				// rgb = (rgb << 8) + (int) d[1];
				// rgb = (rgb << 8) + (int) d[2];
				// pixels2[count] = rgb;
				count++;

			}
		}
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels2, 0, imm.getWidth());

		m = bufferedImageToMat(imm);

		for (String k : keys) {
			// x += mm.get(k);

			mm.remove(k);// TODO odkomentowac?
		}
		///// **************
		//

		// for (String s : mm.keySet())
		// System.out.println("ss " + s + " -> " + mm.get(s));
		// System.out.println("mm s " + mm.size());
		ArrayList<Integer> sizes = new ArrayList<>(mm.values());
		Collections.sort(sizes);
		int median = sizes.size() % 2 == 0 ? (sizes.get(sizes.size() / 2) + sizes.get(sizes.size() / 2 - 1)) / 2
				: sizes.get(sizes.size() / 2);
		// System.out.println(sizes.size() + " " + median);
		int all = 0;
		for (int i : sizes)
			all += i;
		int siz = (int) ((double) all / (double) median);
		// System.out.println("Size " + siz);
		SIZE = siz;

		// SIZE = mm.keySet().size();
		return m;
	}

	private Image setColors(BufferedImage imm, Color col) {
		// originalFrame3.setImage(((Image)imm));
		ArrayList<Scalar> asc = new ArrayList<>();

		for (int i = 10; i < 250; i++)
			for (int j = 0; j < 250; j++)
				asc.add(new Scalar(i, j, i));
		Collections.shuffle(asc);
		// m.convertTo(m, CvType.CV_8SC3);
		// BufferedImage imm = m;// new BufferedImage(m.width(), m.height(),
		// BufferedImage.TYPE_3BYTE_BGR);

		// System.out.println("m " + m.size());
		// System.out.println("w " + imm.getWidth());

		int color = -1;
		int[][] abc = new int[imm.getHeight()][imm.getWidth()];
		FastRGB fast = new FastRGB(imm);
		// m.convertTo(m, CvType.CV_32SC3);

		int[][][] result = new int[imm.getHeight()][imm.getWidth()][4];
		for (int x = 0; x < imm.getWidth(); x++) {
			for (int y = 0; y < imm.getHeight(); y++) {
				Color c = new Color(imm.getRGB(x, y), true);
				result[y][x][0] = c.getRed();
				result[y][x][1] = c.getGreen();
				result[y][x][2] = c.getBlue();
				result[y][x][3] = c.getAlpha();

				if ((c.getRed() == col.getRed() && c.getGreen() == col.getGreen() && c.getBlue() == col.getBlue()))

					abc[y][x] = 1;

				else {
					abc[y][x] = 0;

				}
			}
		}

		/*
		 * for (int r = 0; r < imm.getWidth(); r++) { for (int cc = 0; cc <
		 * imm.getHeight(); cc++) {
		 * 
		 * int[] data = new int[3]; int rgb = imm.getRGB(r, cc);// , data);
		 * 
		 * int rr = (rgb >> 16) & 0xFF; int g = (rgb >> 8) & 0xFF; int b = (rgb
		 * & 0xFF); System.out.println("rgb " + r + " " + g + " " + b); if (!(rr
		 * == 0 && g == 255 && b == 255))
		 * 
		 * abc[r][cc] = 1;
		 * 
		 * else abc[r][cc] = 0;
		 * 
		 * } }
		 */

		for (int i = 0; i < abc.length; i++) {
			for (int j = 0; j < abc[0].length; j++) {
				// System.out.println(abc[i][j] + " ");
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
			// System.out.println();
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
				// System.out.print(Math.abs(abc[i][j]) + " : ");
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
			// System.out.println();
		}
		imm = new BufferedImage(imm.getWidth(), imm.getHeight(), BufferedImage.TYPE_INT_RGB);
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels2, 0, imm.getWidth());

		// m = bufferedImageToMat(imm);
		// System.out.println("ROWS " + m.rows());

		int[] pixels = new int[imm.getWidth() * imm.getHeight()];
		imm.getRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());
		// System.out.println("pix " + pixels.length + ",");
		HashMap<String, Integer> mm = new HashMap<>();
		for (int i = 0; i < pixels.length; i++) {
			int rgb = pixels[i];
			int r = (rgb >> 16) & 0xFF;
			int g = (rgb >> 8) & 0xFF;
			int b = (rgb & 0xFF);

			/*
			 * if (r == 255 && g == 255 && b == 255) { r = 10; g = 10; b = 10;
			 * pixels[i] = 0xFF0A0A0A; } if (r == 0 && g == 0 && b == 0) {
			 * pixels[i] = 0xFFFFFFFF; r = 255; b = 255; g = 255; }
			 */
			String pix = r + "|" + b + "|" + g;
			if (mm.containsKey(pix))
				mm.put(pix, mm.get(pix) + 1);
			else
				mm.put(pix, 1);
		}
		// System.out.println("size " + mm.size());
		cells = mm.size() / 2;

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

	public void createROIwithInfo() {
		String path = "C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67 baza\\ROI wycięte\\dens\\";
		String pathROI = "C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67 baza\\ROI wycięte\\densities\\";

		File folder = new File(pathROI);
		File[] listOfFiles = folder.listFiles();
		String ending = "Markers_Counter Window -.png";

		for (File ROIfile : listOfFiles) {
			if (ROIfile.isFile()) {
				String name = ROIfile.getName().replaceAll(".png", "");
				File densFile = new File(path + name + ending);

				System.out.println(path + name + ending + " DENS FILE " + densFile);
				BufferedImage ROI = null, DENS = null;
				try {

					ROI = ImageIO.read(ROIfile);
					DENS = ImageIO.read(densFile);

					int[] pixelsROI = new int[ROI.getWidth() * ROI.getHeight()];
					ROI.getRGB(0, 0, ROI.getWidth(), ROI.getHeight(), pixelsROI, 0, ROI.getWidth());

					int[] pixelsDENS = new int[DENS.getWidth() * DENS.getHeight()];
					DENS.getRGB(0, 0, DENS.getWidth(), DENS.getHeight(), pixelsDENS, 0, DENS.getWidth());

					// System.out.println("pix " + pixels.length + ",");
					int xx = 0;
					for (int i = 0; i < pixelsROI.length; i++) {

						int rgb = pixelsROI[i];
						int r = (rgb >> 16) & 0xFF;
						int g = (rgb >> 8) & 0xFF;
						int b = (rgb & 0xFF);

						if (r == 255 && g == 255 && b == 255) {
							pixelsDENS[i] = pixelsROI[i];
						}

					}

					DENS.setRGB(0, 0, DENS.getWidth(), DENS.getHeight(), pixelsDENS, 0, DENS.getWidth());

					File outputfile = new File(
							"C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67 baza\\ROI wycięte\\densities\\"
									+ ROIfile.getName() + "_dens_map.png");
					try {
						ImageIO.write(DENS, "png", outputfile);
					} catch (IOException e) {
						e.printStackTrace();
					}

				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

		}

	}

	public void createMaps(String dir) {

		File folder = new File(dir);
		File[] listOfFiles = folder.listFiles();

		for (File file : listOfFiles) {
			if (file.isFile()) {

				BufferedImage bi = null;
				try {
					// BufferedImage tif = ImageIO.read(file);
					// System.out.println("f " + file + ", " + tif);
					// ImageIO.write(tif, "png", new File(dir + "test.png"));
					// file = new File(dir + "test.png");
					bi = ImageIO.read(file);

				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} //
				System.out.println("PR " + file.getAbsolutePath());
				if (bi != null)
					saveMap(bi, file.getName());
			}
		}

	}

	public void saveMap(BufferedImage imm, String name) {

		int[] pixels = new int[imm.getWidth() * imm.getHeight()];
		imm.getRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		// System.out.println("pix " + pixels.length + ",");
		int xx = 0;
		for (int i = 0; i < pixels.length; i++) {

			int rgb = pixels[i];
			int r = (rgb >> 16) & 0xFF;
			int g = (rgb >> 8) & 0xFF;
			int b = (rgb & 0xFF);

			// System.out.println("rgb " + r + " " + g + " " + b);
			if (!(r == CYAN.getRed() && g == CYAN.getGreen() && b == CYAN.getBlue())
					&& !(r == BLUE.getRed() && g == BLUE.getGreen() && b == BLUE.getBlue())) {
				pixels[i] = 0xFF000000;
				// r = 0;
				// b = 0;
				// g = 0;
			} else
				xx++;// System.out.println("ccccccccc");

		}
		// System.out.println(color.getRed() + ", " + color.getGreen() + ", " +
		// color.getBlue() + " CCC " + xx);
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		File outputfile = new File(
				"C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67 baza\\ROI wycięte\\dens\\" + name
						+ "_density.png");
		try {
			ImageIO.write(imm, "png", outputfile);
		} catch (IOException e) {
			e.printStackTrace();
		}
		// WritableImage wr = null;
		// //if (imm != null) {
		// wr = new WritableImage(imm.getWidth(), imm.getHeight());
		// PixelWriter pw = wr.getPixelWriter();
		// for (int x = 0; x < imm.getWidth(); x++) {
		// for (int y = 0; y < imm.getHeight(); y++) {
		// pw.setArgb(x, y, imm.getRGB(x, y));
		// }
		// }
		// //}

		// return imm;// wr;

	}

}
