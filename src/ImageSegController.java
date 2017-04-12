import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

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
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the image segmentation process.
 * 
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 1.5 (2015-11-24)
 * @since 1.0 (2013-12-20)
 * 
 */
public class ImageSegController {

	// FXML buttons
	@FXML
	private Button cameraButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	// checkbox for enabling/disabling Canny
	@FXML
	private CheckBox canny;
	// canny threshold value
	@FXML
	private Slider threshold;
	// checkbox for enabling/disabling background removal
	@FXML
	private CheckBox dilateErode;
	// inverse the threshold value for background removal
	@FXML
	private CheckBox inverse;

	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that performs the video capture
	// private VideoCapture capture = new VideoCapture();
	// a flag to change the button behavior
	// private boolean cameraActive;

	int SIZE;
	/*
	 * 
	 * public ImageSegController() { this.cameraButton.setDisable(false); }
	 */

	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	protected void startCamera() {
		// set a fixed width for the frame
		originalFrame.setFitWidth(1000);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);

		// if (!this.cameraActive) {
		// disable setting checkboxes
		// this.canny.setDisable(true);
		// this.dilateErode.setDisable(true);

		// start the video capture
		// this.capture.open(0);

		// is the video stream available?
		// if (this.capture.isOpened()) {
		// this.cameraActive = true;

		// grab a frame every 33 ms (30 frames/sec)
		Runnable frameGrabber = new Runnable() {

			@Override
			public void run() {
				Image imageToShow = grabFrame("coll1.jpg");
				int SIZ_COLOR1 = SIZE;
				originalFrame.setImage(imageToShow);
				// try {
				// Thread.sleep(1000);
				// } catch (InterruptedException e) {
				// // TODO Auto-generated catch block
				// e.printStackTrace();
				// }
				System.out.println("GRA");
				// imageToShow = grabFrame("coll2.png");
				System.out.println("B");
				// int SIZ_COLOR2 = SIZE;
				// System.out.println(">>>" + SIZ_COLOR1 + ", " + SIZ_COLOR2);
				// // 117
				// 220
				originalFrame.setImage(imageToShow);
			}
		};

		this.timer = Executors.newSingleThreadScheduledExecutor();
		this.timer.scheduleAtFixedRate(frameGrabber, 0, 260000, TimeUnit.MILLISECONDS);

		// update the button content
		// this.cameraButton.setText("Stop Camera");

		// } else {
		// log the error
		// System.err.println("Failed to open the camera connection...");
		// }
		/*
		 * } else { // the camera is not active at this point this.cameraActive
		 * = false; // update again the button content
		 * this.cameraButton.setText("Start Camera"); // enable setting
		 * checkboxes this.canny.setDisable(false);
		 * this.dilateErode.setDisable(false); // stop the timer try {
		 * this.timer.shutdown(); this.timer.awaitTermination(33,
		 * TimeUnit.MILLISECONDS); } catch (InterruptedException e) { // log the
		 * exception System.err.
		 * println("Exception in stopping the frame capture, trying to release the camera now... "
		 * + e); }
		 * 
		 * // release the camera this.capture.release();
		 */
		// clean the frame
		this.originalFrame.setImage(null);
		// }
	}

	/**
	 * Get a frame from the opened video stream (if any)
	 * 
	 * @return the {@link Image} to show
	 */
	/*
	 * private Image grabFrame() { // init everything Image imageToShow = null;
	 * Mat frame = new Mat();
	 * 
	 * // check if the capture is open if (this.capture.isOpened()) { try { //
	 * read the current frame this.capture.read(frame);
	 * 
	 * // if the frame is not empty, process it if (!frame.empty()) { // handle
	 * edge detection if (this.canny.isSelected()) { frame =
	 * this.doSegmentation(frame); } // foreground detection else if
	 * (this.dilateErode.isSelected()) { frame =
	 * this.doBackgroundRemoval(frame); }
	 * 
	 * // convert the Mat object (OpenCV) to Image (JavaFX) imageToShow =
	 * mat2Image(frame); }
	 * 
	 * } catch (Exception e) { // log the (full) error
	 * System.err.print("ERROR"); e.printStackTrace(); } }
	 * 
	 * return imageToShow; }
	 */

	private Image grabFrame(String tit) {
		System.out.println("INS");
		BufferedImage bi = null;
		try {
			bi = ImageIO.read(new File(
					"C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67 baza\\ROI wyciête\\" + tit));//
			System.out.println(
					"C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67 baza\\ROI wyciête\\" + tit);
			// java.awt.Image g = bi.getScaledInstance(bi.getWidth() / 2,
			// bi.getHeight() / 2, 0);//
			// bi = (BufferedImage) g;
			System.out.println("READ" + tit + " " + bi);

			// 124_13_001.png-(Colour_1).jpg"));//
			// surowe

			// int w = bi.getWidth();
			// int h = bi.getHeight();
			// BufferedImage after = new BufferedImage(w, h,
			// BufferedImage.TYPE_INT_ARGB);
			// AffineTransform at = new AffineTransform();
			// at.scale(2.0, 2.0);
			// AffineTransformOp scaleOp = new AffineTransformOp(at,
			// AffineTransformOp.TYPE_BILINEAR);
			// after = scaleOp.filter(bi, after);
			// i
			// oznaczone\\124_13_001.jpg-(Colour_2).jpg"));
		} catch (IOException e) {
			System.out.println("Err " + e);
		}
		System.out.println("NU");
		Image imageToShow = null;
		System.out.println("BI");
		System.out.println("BIII " + bi.getType());
		System.out.println("BI2");
		Mat frame = bufferedImageToMat(bi);// new Mat();
		// Mat frame = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC1);
		// if (!frame.empty())
		// Imgproc.cvtColor(frame2, frame, Imgproc.COLOR_RGB2GRAY);
		// Mat frame = Imgcodecs.imread(
		// "C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67
		// baza\\surowe i oznaczone\\124_13_001.jpg-(Colour_2).jpg",
		// Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
		// check if the capture is open
		// if (this.capture.isOpened()) {
		try {
			// read the current frame
			// this.capture.read(frame);

			// if the frame is not empty, process it
			if (!frame.empty()) {
				// handle edge detection
				// if (this.canny.isSelected()) {
				frame = this.doSegmentationOK(frame);

				// foreground detection
				// else if (this.dilateErode.isSelected()) {
				// frame = this.doBackgroundRemoval(frame);
				// }

				// convert the Mat object (OpenCV) to Image (JavaFX)
				imageToShow = mat2Image(frame);
			}

		} catch (Exception e) {
			// log the (full) error
			System.err.print("ERROR");
			e.printStackTrace();
		}
		// }

		return imageToShow;
	}

	public static Mat bufferedImageToMat(BufferedImage bi) {
		Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		mat.put(0, 0, data);
		return mat;
	}

	/**
	 * Perform the operations needed for removing a uniform background
	 * 
	 * @param frame
	 *            the current frame
	 * @return an image with only foreground objects
	 */
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

	/**
	 * Get the average hue value of the image starting from its Hue channel
	 * histogram
	 * 
	 * @param hsvImg
	 *            the current frame in HSV
	 * @param hueValues
	 *            the Hue component of the current frame
	 * @return the average Hue value
	 */
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

	/**
	 * Apply Canny
	 * 
	 * @param frame
	 *            the current frame
	 * @return an image elaborated with Canny
	 */
	private Mat doSegmentation2(Mat frame) {
		// init
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat(frame.size(), CvType.CV_8UC3);
		System.out.println("DET " + detectedEdges.type());
		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
		// Imgproc.distanceTransform(grayImage, detectedEdges,
		// Imgproc.CV_DIST_L2, 3);// bw,
		//
		// if (true)
		// return detectedEdges;
		Imgproc.threshold(grayImage, grayImage, 172, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);

		Imgproc.distanceTransform(grayImage, grayImage, Imgproc.CV_DIST_L1, 3);

		Core.normalize(grayImage, detectedEdges, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC3);

		Mat inc = new Mat();
		Mat bord = new Mat(), bord2 = new Mat();
		detectedEdges.copyTo(bord);
		// Imgproc.dilate(inc, bord, Mat.ones(new Size(5, 5), CvType.CV_8UC1),
		// new Point(3, 3), 5);

		// if (true)
		// return bord;
		List<MatOfPoint> p = new ArrayList<>();
		Mat Hierarchy = new Mat();
		Imgproc.findContours(bord, p, Hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

		// for( int idx = 0; idx>= 0; idx = hierarchy[idx][0], compCount++ )
		// Imgproc.drawContours(markers, contours, idx,
		// Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
		// for (int i = 0; i < points.size(); i++)
		Mat markersX = Mat.zeros(frame.size(), CvType.CV_32SC1);// CvType.CV_8UC3);//
		// CV_32SC1);//
		// CV_8UC3);//
		// CV_32SC1);
		// Scalar sc = new Scalar(128, 255, 0);

		Imgproc.drawContours(markersX, p, -1, new Scalar(255, 255, 255), -1);
		Imgproc.drawContours(bord, p, -1, new Scalar(255, 255, 255), -1);

		if (true)
			return bord;
		Imgproc.threshold(bord, inc, 70, 255, Imgproc.THRESH_BINARY);

		if (true)
			return inc;

		// detectedEdges.convertTo(detectedEdges, CvType.CV_8UC3);//
		Mat x = new Mat();

		Imgproc.GaussianBlur(detectedEdges, detectedEdges, new Size(17, 17), 0);
		Mat inv = new Mat();
		detectedEdges.copyTo(inv);

		// if (true)
		// return detectedEdges;
		Imgproc.threshold(detectedEdges, inv, 172, 255, Imgproc.THRESH_BINARY_INV);
		Imgproc.morphologyEx(inv, inv, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));
		System.out.println("INV " + inv.type() + ", " + inv.channels());
		Imgproc.distanceTransform(inv, inv, Imgproc.CV_DIST_L1, 3);

		Mat border = new Mat(), border2 = new Mat();
		inv.copyTo(border);
		Imgproc.dilate(inv, border, Mat.ones(new Size(5, 5), CvType.CV_8UC1), new Point(3, 3), 5);
		Imgproc.erode(border, border2, Mat.ones(new Size(5, 5), CvType.CV_8UC1), new Point(3, 3), 5);
		Core.subtract(border, border2, border);
		System.out.println("border " + border.type() + ", " + border.channels());
		// Imgproc.cvtColor(border, border, 0);
		System.out.println("after " + border.type() + ", " + border.channels());
		// Imgproc.cvtColor(border, border, CvType.CV_8UC1, 3);
		// Imgproc.distanceTransform(border, border, 1, 3);
		// new
		// Point(5,5),
		// 5);
		if (true)
			return border;
		Imgproc.threshold(detectedEdges, detectedEdges, 117, 255, Imgproc.THRESH_BINARY);// BINARY);//

		// _INV);//
		// BINARY);//

		// if (true)
		// return inv;
		// _INV);//
		// HRESH_BINARY);//
		// +
		// Imgproc.THRESH_OTSU);
		// CvMemStorage storage=CvMemStorage.create();
		// Seq squares = new CvContour();
		// squares = cvCreateSeq(0, sizeof(CvContour.class),
		// sizeof(CvSeq.class), storage);
		Mat cont = new Mat();
		detectedEdges.copyTo(cont);
		// if (true)
		// return detectedEdges;
		final List<MatOfPoint> points = new ArrayList<>();
		final Mat hierarchy = new Mat();
		Imgproc.findContours(cont, points, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

		// for( int idx = 0; idx>= 0; idx = hierarchy[idx][0], compCount++ )
		// Imgproc.drawContours(markers, contours, idx,
		// Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
		// for (int i = 0; i < points.size(); i++)
		Mat markers = Mat.zeros(frame.size(), CvType.CV_32SC1);// CvType.CV_8UC3);//
		// CV_32SC1);//
		// CV_8UC3);//
		// CV_32SC1);
		Scalar sc = new Scalar(128, 255, 0);

		Imgproc.drawContours(markers, points, -1, new Scalar(255, 255, 255), -1);

		// if (true)
		// return markers;
		System.out.println("PO " + points.size());
		for (int i = 0; i < points.size(); i++) {
			Imgproc.drawContours(markers, points, i, Scalar.all((i + 1) % 255), -1);
			// Imgproc.drawContours(frame, points, i, new Scalar((i * 23) % 255,
			// (i
			// * 10) % 255, (i + i * 30) % 255), -1);

		}
		Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
		// if (true)
		// return markers;
		// Imgproc.drawContours(markers, points, -1, new Scalar(128, 255, 0),
		// 3);
		// Imgproc.drawContours(detectedEdges, points, contourIdx, color);
		System.out.println(points);
		// Imgproc.cvtColor(detectedEdges, detectedEdges,
		// Imgproc.COLOR_GRAY2BGR);
		// Imgproc.findContours(detectedEdges, contours, hierarchy, mode,
		// method);//drawContours(detectedEdges, contours, contourIdx, color);
		// System.out.println(">> " + CvType.CV_8UC3 + ", d " +
		// detectedEdges.type());
		// detectedEdges.convertTo(detectedEdges, CvType.CV_8UC3);// 32SC1);
		// System.out.println(">>2) " + CvType.CV_8UC3 + ", d " +
		// grayImage.type());
		// System.out.println("ss " + detectedEdges.type() + ", " +
		// markers.type());
		Imgproc.cvtColor(grayImage, grayImage, Imgproc.COLOR_GRAY2RGB);
		// if (true) {
		// Mat dest = new Mat();
		// frame.copyTo(dest, detectedEdges);// detectedEdges;
		// return detectedEdges;
		// }
		Imgproc.cvtColor(detectedEdges, detectedEdges, Imgproc.COLOR_GRAY2RGB);// CvType.CV_8UC3);
		Imgproc.cvtColor(inv, inv, Imgproc.COLOR_GRAY2RGB);// CvType.CV_8UC3);

		Mat xx = new Mat();
		// if (true)
		// return detectedEdges;
		detectedEdges.copyTo(xx);
		;

		Imgproc.watershed(xx, markers);// detectedEdges

		Mat wshed = new Mat(markers.size(), CvType.CV_8UC3);

		// System.out.println("MA " + markers.type() + ", " + markers.depth() +
		// " " + Imgproc.COLOR_RGB2GRAY);

		// Imgproc.cvtColor(markers, markers, 0);// Imgproc.COLOR_RGB2GRAY);//
		// CvType.CV_8U);
		// UByteBufferIndexer sI = markers.createIndexer();
		// paint the watershed image

		for (int i = 0; i < markers.rows(); i++)
			for (int j = 0; j < markers.cols(); j++) {
				System.out.println(CvType.CV_8S + ", " + CvType.CV_8U);// CvType.depth(markers.type()));
				int[] cc = new int[3];

				int index = markers.get(i, j, cc);// at(i,j);
				System.out.println(" i j i " + i + ", " + j + " = " + index);
				for (int ic : cc)
					System.out.println("> " + ic);

				if (index == -1)
					wshed.put(i, j, new double[] { 255, 255, 255 }); // = //
																		// Vec3b(255,255,255);
				else if (index <= 0)//
					// || index > compCount )
					wshed.put(i, j, new double[] { 0, 0, 0 });//
				// wshed.at<Vec3b>(i,j) // = // Vec3b(0,0,0);
				else
					wshed.put(i, j, new double[] { index % 256, index % 256, index % 256 });
			}

		// byte[] data = new
		// byte[markers.rows()*markers.cols()*(int)(markers.elemSize())];
		// markers.get(0, 0, data);

		// wshed = wshed*0.5 + imgGray*0.5;
		// Imgproc.cvtColor(markers, markers, CvType.CV_8U);//
		// Imgproc.COLOR_GRAY2RGB);
		// if (true)
		// return markers;
		// Imgproc.cvtColor(detectedEdges, frame, Imgproc.COLOR_GRAY2BGR);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));
		// System.out.println(Mat.ones(3, 3, CvType.CV_8UC1));
		// System.out.println(this.threshold.getValue());// 58.730158730158735
		// canny detector, with ratio of lower:upper threshold of 3:1
		// Imgproc.Canny(detectedEdges, detectedEdges,
		// this.threshold.getValue(), this.threshold.getValue() * 3);

		// using Canny's output as a mask, display the result
		Mat dest = new Mat();
		if (true)
			return wshed;
		// if (true)
		// return markers;
		frame.copyTo(dest, xx);// detectedEdges);// detectedEdges);
		// dest = detectedEdges;
		return dest;
	}

	private Mat doSegmentation(Mat frame) {
		// init
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat(frame.size(), CvType.CV_8UC3);
		System.out.println("DET " + detectedEdges.type());
		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
		// Imgproc.distanceTransform(grayImage, detectedEdges,
		// Imgproc.CV_DIST_L2, 3);// bw,
		//
		// if (true)
		// return detectedEdges;

		Core.normalize(grayImage, detectedEdges, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC3);

		// detectedEdges.convertTo(detectedEdges, CvType.CV_8UC3);//
		Mat x = new Mat();
		// System.out.println("DDD " + grayImage.dims());

		// System.out.println(CvType.CV_8U + " DET2 " + detectedEdges.type());
		// reduce noise with a 3x3 kernel
		// Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

		// dist,
		// CV_DIST_L2,
		// 3);
		Mat matEllipse = Mat.ones(new Size(5, 5), 0);
		matEllipse.put(0, 0, 0);
		matEllipse.put(0, 1, 0);
		matEllipse.put(0, 3, 0);
		matEllipse.put(0, 4, 0);
		matEllipse.put(4, 0, 0);
		matEllipse.put(4, 1, 0);
		matEllipse.put(4, 3, 0);
		matEllipse.put(4, 4, 0);

		// Imgproc.threshold(detectedEdges, detectedEdges, 60, 255,
		// Imgproc.THRESH_BINARY);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_OPEN, matEllipse);
		// Imgproc.erode(detectedEdges, detectedEdges, matEllipse);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_GRADIENT, matEllipse);
		// if (true)
		// return detectedEdges;
		Mat det = new Mat();
		detectedEdges.copyTo(det);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_GRADIENT, matEllipse);
		// if (true)
		// return detectedEdges;
		// ####

		final List<MatOfPoint> pointss = new ArrayList<>();
		final Mat hierarchys = new Mat();
		Imgproc.threshold(detectedEdges, detectedEdges, 90, 255, Imgproc.THRESH_BINARY_INV);// INV?

		Imgproc.distanceTransform(detectedEdges, detectedEdges, Imgproc.CV_DIST_L2, 3);
		Imgproc.threshold(detectedEdges, detectedEdges, 3, 255, Imgproc.THRESH_BINARY_INV);// INV?
		Imgproc.dilate(detectedEdges, detectedEdges, matEllipse);

		detectedEdges.convertTo(detectedEdges, CvType.CV_8UC1);
		Imgproc.findContours(detectedEdges, pointss, hierarchys, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);// Imgproc.RETR_TREE,

		Mat markerss = Mat.zeros(frame.size(), CvType.CV_32SC1);
		System.out.println("POINTSs " + pointss.size());
		markerss.convertTo(markerss, CvType.CV_8U);
		ArrayList<Scalar> asc = new ArrayList<>();
		for (int i = 0; i < 256; i++)
			for (int j = 0; j < 256; j++)
				asc.add(new Scalar(i, j, 125));
		Collections.shuffle(asc);

		for (int i = 0; i < pointss.size(); i++) {
			Imgproc.drawContours(markerss, pointss, i, asc.get(i), -1);
		}
		Imgproc.circle(markerss, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);

		Imgproc.cvtColor(detectedEdges, detectedEdges, Imgproc.COLOR_GRAY2RGB);

		// if (true)
		// return markerss;

		Imgproc.threshold(det, det, 80, 255, Imgproc.THRESH_BINARY_INV);
		Imgproc.morphologyEx(det, det, Imgproc.MORPH_OPEN, matEllipse);

		// if (true)
		// return det;
		Imgproc.cvtColor(det, det, Imgproc.COLOR_GRAY2RGB);
		det.convertTo(det, CvType.CV_8UC3);
		markerss.convertTo(markerss, CvType.CV_32SC1);
		System.out
				.println("mm " + markerss.type() + ", " + det.type() + ", " + CvType.CV_8UC3 + ", " + CvType.CV_32SC1);
		Imgproc.watershed(det, markerss);

		if (true) {

			return markerss;
		}
		//
		//
		// // ######
		Imgproc.GaussianBlur(detectedEdges, detectedEdges, new Size(17, 17), 0);

		Mat inv = new Mat();
		detectedEdges.copyTo(inv);

		Imgproc.threshold(inv, inv, 160, 255, Imgproc.THRESH_BINARY_INV);

		Imgproc.morphologyEx(inv, inv, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));

		// if (true)
		// return inv;
		Imgproc.threshold(detectedEdges, detectedEdges, 117, 255, Imgproc.THRESH_BINARY);// BINARY);//

		Mat inv2 = new Mat(), inv3 = new Mat();
		inv.copyTo(inv2);
		detectedEdges.copyTo(inv3);
		Core.bitwise_not(inv3, inv3);
		// if (true)
		// return inv3;
		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));

		Imgproc.distanceTransform(inv3, inv3, Imgproc.CV_DIST_L2, 3);

		Imgproc.threshold(inv3, inv3, 3, 255, Imgproc.THRESH_BINARY);// INV?
		inv3.convertTo(inv3, CvType.CV_8UC1);
		Imgproc.distanceTransform(inv3, inv3, Imgproc.CV_DIST_L1, 5);
		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));
		Imgproc.threshold(inv3, inv3, 3, 255, Imgproc.THRESH_BINARY);// INV?

		// SAME
		inv3.convertTo(inv3, CvType.CV_8UC1);
		Imgproc.distanceTransform(inv3, inv3, Imgproc.CV_DIST_L1, 5);
		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));
		Imgproc.threshold(inv3, inv3, 1, 255, Imgproc.THRESH_BINARY);// INV?

		Imgproc.distanceTransform(inv2, inv2, Imgproc.CV_DIST_L1, 3);

		Imgproc.erode(inv2, inv2, matEllipse, new Point(3, 3), 1);

		inv2.convertTo(inv2, CvType.CV_8UC1);
		Imgproc.distanceTransform(inv2, inv2, Imgproc.CV_DIST_L1, 3);

		Imgproc.threshold(inv2, inv2, 8, 255, Imgproc.THRESH_BINARY);

		inv2.convertTo(inv2, CvType.CV_8U);
		inv3.convertTo(inv3, CvType.CV_8U);

		Mat cont = new Mat();

		inv2.copyTo(cont);
		// if (true)
		// return detectedEdges;
		final List<MatOfPoint> points = new ArrayList<>();
		final Mat hierarchy = new Mat();
		Imgproc.findContours(cont, points, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
		Mat markers = Mat.zeros(frame.size(), CvType.CV_32SC1);

		System.out.println("POINTS " + points.size());
		for (int i = 0; i < points.size(); i++) {
			Imgproc.drawContours(markers, points, i, new Scalar((i * 30) % 256, (i + 27) % 256, i), -1);

			Imgproc.drawContours(cont, points, i, new Scalar((i * 30) % 256, (i + 27) % 256, i), -1);
		}
		Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);

		Imgproc.cvtColor(inv, inv, Imgproc.COLOR_GRAY2RGB);// CvType.CV_8UC3);

		Mat xx = new Mat();
		// if (true)
		// return detectedEdges;
		inv.copyTo(xx);
		;
		// if (true)
		// return xx;
		Imgproc.watershed(xx, markers);// detectedEdges
		if (true)
			return markers;
		if (true) {
			markers.convertTo(markers, CvType.CV_8UC3);
			// System.out.println("INV c " + inv.channels());
			Imgproc.cvtColor(markers, markers, Imgproc.COLOR_GRAY2BGR);// todo
			// if (true)
			// return markers;
			Core.bitwise_not(inv, inv);
			// if (true)
			// return inv;
			markers.copyTo(markers, inv);// markers);
			return markers;
		}

		Mat wshed2 = new Mat(markers.size(), CvType.CV_8UC3);

		// if (true) {
		// System.out.println("wshed");
		return wshed2;
		// }

	}

	private Mat doSegmentationOK(Mat frame) {
		ArrayList<Scalar> asc = new ArrayList<>();
		/*
		 * Set<Integer> ss = new HashSet(); for (int i = 10; i < 256; i++) for
		 * (int j = 10; j < 256; j++) { for (int k = 10; k < 11; k++) { int x =
		 * (int) (0.299 * i + 0.587 * j + 0.114 * k);// (i + j // + k) // / 3;
		 * if (ss.contains(x)) continue; else { asc.add(new Scalar(i, j, k));
		 * ss.add(x); } } }
		 */
		for (int i = 10; i < 250; i++)
			for (int j = 0; j < 250; j++)
				asc.add(new Scalar(i, j, i));
		Collections.shuffle(asc);
		// init
		Mat grayImage = new Mat();

		Mat detectedEdges = new Mat(frame.size(), CvType.CV_8UC3);
		System.out.println("DET " + detectedEdges.type());
		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
		if (true)
			return grayImage;
		// Imgproc.distanceTransform(grayImage, detectedEdges,
		// Imgproc.CV_DIST_L2, 3);// bw,
		//
		// if (true)
		// return frame;

		Core.normalize(grayImage, detectedEdges, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC3);

		// detectedEdges.convertTo(detectedEdges, CvType.CV_8UC3);//
		Mat x = new Mat();
		// System.out.println("DDD " + grayImage.dims());

		// System.out.println(CvType.CV_8U + " DET2 " + detectedEdges.type());
		// reduce noise with a 3x3 kernel
		// Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

		// dist,
		// CV_DIST_L2,
		// 3);

		Mat matEllipse = Mat.ones(new Size(5, 5), 0);
		matEllipse.put(0, 0, 0);
		matEllipse.put(0, 1, 0);
		matEllipse.put(0, 3, 0);
		matEllipse.put(0, 4, 0);
		matEllipse.put(4, 0, 0);
		matEllipse.put(4, 1, 0);
		matEllipse.put(4, 3, 0);
		matEllipse.put(4, 4, 0);

		// Imgproc.threshold(detectedEdges, detectedEdges, 60, 255,
		// Imgproc.THRESH_BINARY);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_OPEN, matEllipse);
		// Imgproc.erode(detectedEdges, detectedEdges, matEllipse);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_GRADIENT, matEllipse);
		// if (true)
		// return detectedEdges;
		Mat det = new Mat();
		detectedEdges.copyTo(det);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_GRADIENT, matEllipse);
		// if (true)
		// return detectedEdges;
		// ####

		// final List<MatOfPoint> pointss = new ArrayList<>();
		// final Mat hierarchys = new Mat();
		// Imgproc.threshold(detectedEdges, detectedEdges, 90, 255,
		// Imgproc.THRESH_BINARY_INV);// INV?
		//
		// Imgproc.distanceTransform(detectedEdges, detectedEdges,
		// Imgproc.CV_DIST_L2, 3);
		// Imgproc.threshold(detectedEdges, detectedEdges, 3, 255,
		// Imgproc.THRESH_BINARY_INV);// INV?
		// Imgproc.dilate(detectedEdges, detectedEdges, matEllipse);
		//
		//
		// detectedEdges.convertTo(detectedEdges, CvType.CV_8UC1);
		// Imgproc.findContours(detectedEdges, pointss, hierarchys,
		// Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);// Imgproc.RETR_TREE,
		//
		// Mat markerss = Mat.zeros(frame.size(), CvType.CV_32SC1);
		// System.out.println("POINTSs " + pointss.size());
		// markerss.convertTo(markerss, CvType.CV_8U);
		// ArrayList<Scalar> asc = new ArrayList<>();
		// for (int i = 0; i < 256; i++)
		// for (int j = 0; j < 256; j++)
		// asc.add(new Scalar(i, j, 125));
		// Collections.shuffle(asc);
		//
		// for (int i = 0; i < pointss.size(); i++) {
		// Imgproc.drawContours(markerss, pointss, i, asc.get(i), -1);
		// }
		// Imgproc.circle(markerss, new Point(5, 5), 3, new Scalar(255, 255,
		// 255), -1);
		//
		// Imgproc.cvtColor(detectedEdges, detectedEdges,
		// Imgproc.COLOR_GRAY2RGB);
		// Imgproc.watershed(detectedEdges, markerss);
		//
		// if (true) {
		//
		// return markerss;
		// }
		//
		//
		// // ######
		Imgproc.GaussianBlur(detectedEdges, detectedEdges, new Size(17, 17), 0);

		Mat inv = new Mat();
		detectedEdges.copyTo(inv);

		Imgproc.threshold(inv, inv, 160, 255, Imgproc.THRESH_BINARY_INV);
		// if (true)
		// return inv;

		Imgproc.morphologyEx(inv, inv, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));// OPEN
																							// BY£O

		// if (true)
		// return inv;
		Imgproc.threshold(detectedEdges, detectedEdges, 117, 255, Imgproc.THRESH_BINARY);// BINARY);//

		Mat inv2 = new Mat(), inv3 = new Mat();
		inv.copyTo(inv2);
		detectedEdges.copyTo(inv3);
		Core.bitwise_not(inv3, inv3);
		// if (true)
		// return inv3;
		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));

		Imgproc.distanceTransform(inv3, inv3, Imgproc.CV_DIST_L2, 3);

		Imgproc.threshold(inv3, inv3, 3, 255, Imgproc.THRESH_BINARY);// INV?
		inv3.convertTo(inv3, CvType.CV_8UC1);
		Imgproc.distanceTransform(inv3, inv3, Imgproc.CV_DIST_L1, 5);
		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));
		Imgproc.threshold(inv3, inv3, 3, 255, Imgproc.THRESH_BINARY);// INV?

		// SAME
		inv3.convertTo(inv3, CvType.CV_8UC1);
		Imgproc.distanceTransform(inv3, inv3, Imgproc.CV_DIST_L1, 5);
		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));
		Imgproc.threshold(inv3, inv3, 1, 255, Imgproc.THRESH_BINARY);// INV?

		// if (true)
		// return inv3;
		// Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3,
		// CvType.CV_8UC1));
		// Imgproc.dilate(inv3, inv3, matEllipse, new Point(3, 3), 5);
		// Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3,
		// CvType.CV_8UC1));

		// Imgproc.threshold(inv3, inv3, 8, 255, Imgproc.THRESH_BINARY);// INV?
		// if (true)
		// return inv3;
		// Imgproc.dilate(inv3, inv3, matEllipse);

		// Imgproc.dilate(inv2, inv2, matEllipse, // Mat.ones(new Size(5, 5),
		// CvType.CV_8UC1),
		// new Point(3, 3), 8);
		Imgproc.distanceTransform(inv2, inv2, Imgproc.CV_DIST_L1, 3);

		Imgproc.erode(inv2, inv2, matEllipse, // Mat.ones(new Size(5, 5),
												// CvType.CV_8UC1),
				new Point(3, 3), 1);
		// Imgproc.erode(inv2, inv2, matEllipse, // Mat.ones(new Size(5, 5),
		// CvType.CV_8UC1),
		// new Point(3, 3), 1);
		// Imgproc.morphologyEx(inv2, inv2, Imgproc.MORPH_CLOSE, matEllipse);

		// if (true)
		// return inv2;
		// Imgproc.dilate(inv2, inv2, Mat.ones(new Size(5, 5), CvType.CV_8UC1),
		// new Point(3, 3), 3);
		inv2.convertTo(inv2, CvType.CV_8UC1);
		Imgproc.distanceTransform(inv2, inv2, Imgproc.CV_DIST_L1, 3);

		Imgproc.threshold(inv2, inv2, 8, 255, Imgproc.THRESH_BINARY);

		// Imgproc.threshold(inv2, inv2, 8, 255, Imgproc.THRESH_BINARY);//
		// if (true)
		// return inv2;
		// 8//odkom

		// Imgproc.morphologyEx(inv2, inv2, Imgproc.MORPH_TOPHAT, matEllipse);
		// Imgproc.threshold(inv2, inv2, 4, 255, Imgproc.THRESH_BINARY);// usun
		// if (true)
		// return inv2;

		// Imgproc.morphologyEx(inv2, inv2, Imgproc.MORPH_CLOSE, matEllipse);
		// if (true)
		// return inv2;
		// Imgproc.morphologyEx(inv2, inv2, Imgproc.MORPH_CLOSE, matEllipse);

		// if (true)
		// return inv2;
		// Imgproc.cvtColor(inv2, inv2, Imgproc.COLOR_GRAY2BGR);
		// Imgproc.cvtColor(inv2, inv2, CvType.CV_8U);//
		// Imgproc.COLOR_BGR2GRAY);
		// Imgproc.cvtColor(inv2, inv2, 0);
		inv2.convertTo(inv2, CvType.CV_8U);
		inv3.convertTo(inv3, CvType.CV_8U);
		System.out.println(
				"INV " + inv.type() + ",INV2 " + inv2.type() + " , " + CvType.CV_8UC1 + ", " + Imgproc.COLOR_GRAY2BGR);
		// if (true)
		// return inv2;
		// Mat inv3 = new Mat();
		// inv2.copyTo(inv3);
		// Core.normalize(inv3, inv3, 0, 100, Core.NORM_MINMAX, CvType.CV_8UC3);

		// if (true)
		// return inv2;
		// System.out.println("2) INV " + inv.type() + ",INV2 " + inv2.type() +
		// " , " + CvType.CV_8UC1 + ", "
		// + Imgproc.COLOR_GRAY2BGR);
		// if (true)
		// return inv2;

		Mat cont = new Mat();

		inv3.copyTo(cont);
		// if (true)
		// return detectedEdges;
		final List<MatOfPoint> points = new ArrayList<>();
		final Mat hierarchy = new Mat();
		// if (true)
		// return cont;
		Imgproc.findContours(cont, points, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);// Imgproc.RETR_TREE,
																										// Imgproc.CHAIN_APPROX_SIMPLE);
		Mat markers = Mat.zeros(frame.size(), CvType.CV_32SC1);// CvType.CV_8UC3);//
		// CV_32SC1);//
		// CV_8UC3);//
		// CV_32SC1);

		Scalar sc = new Scalar(128, 255, 0);
		System.out.println("POINTS " + points.size());
		ArrayList<Scalar> scs = new ArrayList<>();
		// int cnt=0;
		System.out.println("asc " + asc.size());

		for (int i = 0; i < points.size(); i++) {
			Scalar as = asc.get(i);
			/*
			 * while (true){ for (Scalar s: scs){ double[] w=s.val;
			 * 
			 * } }
			 */
			Imgproc.drawContours(markers, points, i, as, -1);// new
			// Scalar((i
			// * 30)
			// %
			// 256,
			// (i +
			// 27) %
			// 256,
			// i),
			// -1);

			// Imgproc.drawContours(cont, points, i, new Scalar((i * 30) % 256,
			// (i + 27) % 256, i), -1);
		}
		Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
		// if (true)
		// return markers;
		// if (true)
		// return cont;
		// Mat mark = Mat.zeros(markers.size(), CvType.CV_8UC1);
		// markers.copyTo(mark);
		// System.out.println("cc " + CvType.CV_8U);

		// Imgproc.cvtColor(markers, mark, Imgproc.COLOR_GRAY2BGR);

		// System.out.println(CvType.CV_8UC1 + " - " + mark.type());
		// if (true)
		// return mark;
		Imgproc.cvtColor(inv, inv, Imgproc.COLOR_GRAY2RGB);// CvType.CV_8UC3);

		Mat xx = new Mat();
		// if (true)
		// return detectedEdges;
		inv.copyTo(xx);
		;
		// if (true)
		// return markers;
		Imgproc.watershed(xx, markers);// detectedEdges

		// Imgproc.threshold(markers, markers, 255, 255, Imgproc.THRESH_BINARY);
		// if (true)
		// return markers;
		// Core.bitwise_not(markers, markers);

		BufferedImage imm = new BufferedImage(markers.width(), markers.height(), BufferedImage.TYPE_BYTE_GRAY);// INT_RGB);//
																												// ,
																												// BufferedImage.TYPE_BYTE_GRAY);
		markers.convertTo(markers, CvType.CV_8U);// ?
		points.clear();
		System.out.println("FIND");
		final Mat hierarchy2 = new Mat();
		// Imgproc.findContours(markers, points, hierarchy2, Imgproc.RETR_TREE,
		// Imgproc.CHAIN_APPROX_SIMPLE);// Imgproc.RETR_TREE,
		// Imgproc.CHAIN_APPROX_SIMPLE);
		/*
		 * System.out.println("FOUND" + points.size()); Mat markers2 =
		 * Mat.zeros(frame.size(), CvType.CV_32SC1); for (int i = 0; i <
		 * points.size(); i++) {
		 * 
		 * Imgproc.drawContours(markers2, points, i, asc.get(i), -1); }
		 * System.out.println("p" + points.size()); if (true) return markers2;
		 */
		matToBufferedImage(markers, imm);

		int[] pixels = new int[imm.getWidth() * imm.getHeight()];
		imm.getRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());// ((DataBufferByte)
																						// imm.getRaster().getDataBuffer()).getData();
		System.out.println("pix " + pixels.length + ",");
		HashMap<String, Integer> mm = new HashMap<>();
		for (int i = 0; i < pixels.length; i++) {
			int rgb = pixels[i];
			int r = (rgb >> 16) & 0xFF;
			int g = (rgb >> 8) & 0xFF;
			int b = (rgb & 0xFF);
			// if (r == 0 && g == 0 && b == 0) {
			// r = 10;
			// g = 10;
			// b = 10;
			// pixels[i] = 0xFF0A0A0A;
			// }
			// if (r == 255 && g == 255 && b == 255) {
			// pixels[i] = 0xFF000000;
			// r = 0;
			// b = 0;
			// g = 0;
			// }
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
			if (mm.containsKey(pix))// pixels[i]))
				mm.put(pix, mm.get(pix) + 1);
			else
				mm.put(pix, 1);
		}
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());
		/*
		 * JFrame jj = new JFrame();
		 * 
		 * JLabel picLabel = new JLabel(new ImageIcon(imm.getScaledInstance(900,
		 * 600, 0))); picLabel.setSize(new Dimension(900, 600));
		 * jj.add(picLabel); // j.add((new Component(imm)); jj.show(); JLabel
		 * picLabel2 = new JLabel(new ImageIcon(imm)); JFrame jj2 = new
		 * JFrame(); jj2.add(picLabel2); jj2.show();
		 */
		// System.out.println("k " + mm.keySet().size());
		// for (String k : mm.keySet())
		// System.out.println("K " + k + "->" + mm.get(k));

		// System.out.println(">" + BufferedImage.TYPE_INT_RGB);
		BufferedImage imm2 = new BufferedImage(imm.getWidth(), imm.getHeight(), 5);
		imm2.getGraphics().drawImage(imm, 0, 0, null);
		Mat immat = bufferedImageToMat(imm2);

		Mat gr = new Mat();
		Imgproc.morphologyEx(immat, gr, Imgproc.MORPH_GRADIENT, matEllipse, new Point(1, 1), 3);
		Imgproc.threshold(gr, gr, 1, 255, Imgproc.THRESH_BINARY);
		// gr.convertTo(gr, CvType.CV_8U);// 8UC1);
		// Imgproc.cvtColor(gr, gr, Imgproc.COLOR_BGR2GRAY);
		// immat.convertTo(immat, CvType.CV_8U);// 8UC1);
		// Imgproc.cvtColor(immat, immat, Imgproc.COLOR_BGR2GRAY);

		Core.bitwise_or(immat, gr, immat);

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
			// if (r == 0 && g == 0 && b == 0) {
			// r = 10;
			// g = 10;
			// b = 10;
			// pixels[i] = 0xFF0A0A0A;
			// }
			if (r == 255 && g == 255 && b == 255) {
				pixels[i] = 0xFF000000;
				r = 0;
				b = 0;
				g = 0;
			}

		}
		imm.setRGB(0, 0, imm.getWidth(), imm.getHeight(), pixels, 0, imm.getWidth());

		// jj.show();
		// if (true)
		// return immat;
		/*
		 * JFrame jj = new JFrame();
		 * 
		 * JLabel picLabel = new JLabel(new ImageIcon(imm.getScaledInstance(900,
		 * 600, 0))); picLabel.setSize(new Dimension(900, 600));
		 * jj.add(picLabel); // j.add((new Component(imm)); jj.show(); JLabel
		 * picLabel2 = new JLabel(new ImageIcon(imm)); JFrame jj2 = new
		 * JFrame(); jj2.add(picLabel2); jj2.show();
		 */
		System.out.println("k22 " + mm.keySet().size());
		for (String k : mm.keySet())
			System.out.println("K " + k + "->" + mm.get(k));

		System.out.println(">" + BufferedImage.TYPE_INT_RGB);
		imm2 = new BufferedImage(imm.getWidth(), imm.getHeight(), 5);
		imm2.getGraphics().drawImage(imm, 0, 0, null);
		immat = bufferedImageToMat(imm2);

		// if (true)
		// return immat;
		// markers.convertTo(markers, CvType.CV_32SC1);

		// Imgproc.cvtColor(markers, markers, Imgproc.COLOR_GRAY2BGR);

		// markers.convertTo(markers, gr.type());
		//
		// Mat grs = new Mat();
		// gr.copyTo(grs);
		// gr.convertTo(gr, CvType.CV_8UC3);
		// // Imgproc.cvtColor(grs, gr, Imgproc.COLOR_GRAY2BGR);
		//
		// gr.copyTo(gr, immat);
		// Core.addWeighted(immat, 1.0, gr, 1.0, 0.0, immat);// src1, alpha,
		// src2,
		// // beta, gamma,
		// // dst);
		points.clear();

		Imgproc.cvtColor(immat, immat, Imgproc.COLOR_RGB2GRAY);
		Imgproc.threshold(immat, immat, 1, 256, Imgproc.THRESH_BINARY_INV);
		immat.convertTo(immat, CvType.CV_8U);
		gr = new Mat();
		Imgproc.morphologyEx(immat, gr, Imgproc.MORPH_GRADIENT, matEllipse, new Point(1, 1), 1);
		Core.bitwise_or(immat, gr, immat);
		Imgproc.morphologyEx(immat, immat, Imgproc.MORPH_CLOSE, matEllipse, new Point(1, 1), 2);
		Core.bitwise_not(immat, immat);
		// if (true)
		// return immat;
		Imgproc.findContours(immat, points, hierarchy2, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);// Imgproc.RETR_TREE,
		// Imgproc.CHAIN_APPROX_SIMPLE);
		Mat markersx = Mat.zeros(immat.size(), CvType.CV_32SC3);
		System.out.println("POINTSf " + points.size());
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

		imm2 = matToBufferedImageInt(markersx, imm2);
		System.out.println();
		mm.clear();
		for (int r = 0; r < markersx.rows(); r++) {
			for (int cc = 0; cc < markersx.cols(); cc++) {

				int[] data = new int[3];
				markersx.get(r, cc, data);
				// System.out.print("[" + data[0] + "," + data[1] + "," +
				// data[2] + "]");
				if (data[0] == 0 && data[1] == 0 && data[2] == 0)
					continue;
				String pix = data[0] + "|" + data[1] + "|" + data[2];
				if (mm.containsKey(pix))// pixels[i]))
					mm.put(pix, mm.get(pix) + 1);
				else
					mm.put(pix, 1);
			}

			// System.out.println();
		}

		// for (String s : mm.keySet())
		// System.out.println("mm " + s + "->" + mm.get(s));
		System.out.println("MM " + mm.keySet().size());
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
		// System.out.print("i " + i);
		/*
		 * pixels = new int[imm2.getWidth() * imm2.getHeight()]; imm2.getRGB(0,
		 * 0, imm2.getWidth(), imm2.getHeight(), pixels, 0, imm2.getWidth());//
		 * ((DataBufferByte) // imm.getRaster().getDataBuffer()).getData();
		 * System.out.println("pix " + pixels.length + ",");
		 * 
		 * JFrame jj = new JFrame();
		 * 
		 * JLabel picLabel = new JLabel(new
		 * ImageIcon(imm2.getScaledInstance(900, 600, 0))); picLabel.setSize(new
		 * Dimension(900, 600)); jj.add(picLabel); // jj.show(); mm.clear(); for
		 * (int i = 0; i < pixels.length; i++) { int rgb = pixels[i]; int r =
		 * (rgb >> 16) & 0xFF; int g = (rgb >> 8) & 0xFF; int b = (rgb & 0xFF);
		 * String pix = r + "|" + b + "|" + g; if (mm.containsKey(pix))//
		 * pixels[i])) mm.put(pix, mm.get(pix) + 1); else mm.put(pix, 1); }
		 */

		// for (String s : mm.keySet())
		// System.out.println("ss " + s + "->" + mm.get(s));
		// System.out.println("SS " + mm.keySet().size());
		if (true)
			return markersx;
		// for (int j = 0; j < imm.getWidth(); j++)
		// System.out.print(":" + pixels[i]);
		// Get the BufferedImage's backing array and copy the pixels directly
		// into it
		// byte[] data = ((DataBufferByte)
		// imm.getRaster().getDataBuffer()).getData();
		// markers.get(0, 0, data);
		//
		// for (int i = 0; i < markers.rows(); i++)
		// for (int j = 0; j < markers.cols(); j++) {
		// double[] wyn = markers.get(i, j);
		// System.out.println(wyn.length);
		// if (wyn.length < 3)
		// continue;
		// System.out.println("w " + wyn[0] + "|" + wyn[1] + "|" + wyn[2]);
		// }
		if (true)
			return markers;
		if (true) {
			markers.convertTo(markers, CvType.CV_8UC3);
			// System.out.println("INV c " + inv.channels());
			Imgproc.cvtColor(markers, markers, Imgproc.COLOR_GRAY2BGR);// todo
			// if (true)
			// return markers;
			Core.bitwise_not(inv, inv);
			// if (true)
			// return inv;
			markers.copyTo(markers, inv);// markers);
			return markers;
		}

		Mat wshed2 = new Mat(markers.size(), CvType.CV_8UC3);

		// if (true) {
		// System.out.println("wshed");
		return wshed2;
		// }

	}

	public static BufferedImage matToBufferedImage(Mat matrix, BufferedImage bimg) {
		if (matrix != null) {
			int cols = matrix.cols();
			int rows = matrix.rows();
			int elemSize = (int) matrix.elemSize();
			byte[] data = new byte[cols * rows * elemSize];
			int type;
			System.out.println("MATR " + matrix.type());
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

			// Reuse existing BufferedImage if possible
			if (bimg == null || bimg.getWidth() != cols || bimg.getHeight() != rows || bimg.getType() != type) {
				bimg = new BufferedImage(cols, rows, type);
			}
			bimg.getRaster().setDataElements(0, 0, cols, rows, data);
		} else { // mat was null
			bimg = null;
		}
		return bimg;
	}

	public static BufferedImage matToBufferedImageInt(Mat matrix, BufferedImage bimg) {
		if (matrix != null) {
			int cols = matrix.cols();
			int rows = matrix.rows();
			int elemSize = (int) matrix.elemSize();
			int[] data = new int[cols * rows * 3];
			int type;
			System.out.println(matrix.elemSize() + " MATR " + matrix.type() + " " + matrix.channels());
			matrix.get(0, 0, data);
			switch (matrix.channels()) {
			case 1:
				type = BufferedImage.TYPE_BYTE_GRAY;
				break;
			case 3:
				type = BufferedImage.TYPE_3BYTE_BGR;
				// bgr to rgb
				// int b;
				// for (int i = 0; i < data.length; i = i + 3) {
				// b = data[i];
				// data[i] = data[i + 2];
				// data[i + 2] = b;
				// }
				break;
			default:
				return null;
			}

			// Reuse existing BufferedImage if possible
			if (bimg == null || bimg.getWidth() != cols || bimg.getHeight() != rows || bimg.getType() != type) {
				bimg = new BufferedImage(cols, rows, type);
			}
			byte[] data2 = new byte[cols * rows * elemSize];
			for (int i = 0; i < data.length; i++) {
				byte[] bb1 = new byte[4];
				int rgb = data[i];
				int rr = (rgb >> 32) & 0xFF;
				int r = (rgb >> 16) & 0xFF;
				int g = (rgb >> 8) & 0xFF;
				int b = (rgb & 0xFF);
				bb1[0] = (byte) rr;
				bb1[1] = (byte) r;// g;
				bb1[2] = (byte) g;// b;
				bb1[3] = (byte) b;
				// byte[] bb1 = ByteBuffer.allocate(4).putInt(rgb).array();
				data2[i * 3] = bb1[0];
				data2[i * 3 + 1] = bb1[1];
				data2[i * 3 + 2] = bb1[2];
				data2[i * 3 + 3] = bb1[3];

			}
			byte b;
			/*
			 * for (int i = 0; i < data2.length; i = i + 3) { b = data2[i];
			 * data2[i] = data2[i + 2]; data2[i + 2] = b; }
			 */
			bimg.getRaster().setDataElements(0, 0, cols, rows, data2);
		} else { // mat was null
			bimg = null;
		}
		return bimg;
	}

	private Mat doSegmentationOK2(Mat frame) {
		// init
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat(frame.size(), CvType.CV_8UC3);
		System.out.println("DET " + detectedEdges.type());
		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
		// Imgproc.distanceTransform(grayImage, detectedEdges,
		// Imgproc.CV_DIST_L2, 3);// bw,
		//
		// if (true)
		// return detectedEdges;

		Core.normalize(grayImage, detectedEdges, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC3);

		// detectedEdges.convertTo(detectedEdges, CvType.CV_8UC3);//
		Mat x = new Mat();
		// System.out.println("DDD " + grayImage.dims());

		// System.out.println(CvType.CV_8U + " DET2 " + detectedEdges.type());
		// reduce noise with a 3x3 kernel
		// Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

		// dist,
		// CV_DIST_L2,
		// 3);
		Mat matEllipse = Mat.ones(new Size(5, 5), 0);
		matEllipse.put(0, 0, 0);
		matEllipse.put(0, 1, 0);
		matEllipse.put(0, 3, 0);
		matEllipse.put(0, 4, 0);
		matEllipse.put(4, 0, 0);
		matEllipse.put(4, 1, 0);
		matEllipse.put(4, 3, 0);
		matEllipse.put(4, 4, 0);

		// Imgproc.threshold(detectedEdges, detectedEdges, 60, 255,
		// Imgproc.THRESH_BINARY);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_OPEN, matEllipse);
		// Imgproc.erode(detectedEdges, detectedEdges, matEllipse);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_GRADIENT, matEllipse);
		// if (true)
		// return detectedEdges;
		Mat det = new Mat();
		detectedEdges.copyTo(det);
		// Imgproc.morphologyEx(detectedEdges, detectedEdges,
		// Imgproc.MORPH_GRADIENT, matEllipse);
		// if (true)
		// return detectedEdges;
		// ####

		// final List<MatOfPoint> pointss = new ArrayList<>();
		// final Mat hierarchys = new Mat();
		// Imgproc.threshold(detectedEdges, detectedEdges, 90, 255,
		// Imgproc.THRESH_BINARY_INV);// INV?
		//
		// Imgproc.distanceTransform(detectedEdges, detectedEdges,
		// Imgproc.CV_DIST_L2, 3);
		// Imgproc.threshold(detectedEdges, detectedEdges, 3, 255,
		// Imgproc.THRESH_BINARY_INV);// INV?
		// Imgproc.dilate(detectedEdges, detectedEdges, matEllipse);
		//
		//
		// detectedEdges.convertTo(detectedEdges, CvType.CV_8UC1);
		// Imgproc.findContours(detectedEdges, pointss, hierarchys,
		// Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);// Imgproc.RETR_TREE,
		//
		// Mat markerss = Mat.zeros(frame.size(), CvType.CV_32SC1);
		// System.out.println("POINTSs " + pointss.size());
		// markerss.convertTo(markerss, CvType.CV_8U);
		// ArrayList<Scalar> asc = new ArrayList<>();
		// for (int i = 0; i < 256; i++)
		// for (int j = 0; j < 256; j++)
		// asc.add(new Scalar(i, j, 125));
		// Collections.shuffle(asc);
		//
		// for (int i = 0; i < pointss.size(); i++) {
		// Imgproc.drawContours(markerss, pointss, i, asc.get(i), -1);
		// }
		// Imgproc.circle(markerss, new Point(5, 5), 3, new Scalar(255, 255,
		// 255), -1);
		//
		// Imgproc.cvtColor(detectedEdges, detectedEdges,
		// Imgproc.COLOR_GRAY2RGB);
		// Imgproc.watershed(detectedEdges, markerss);
		//
		// if (true) {
		//
		// return markerss;
		// }
		//
		//
		// // ######
		Imgproc.GaussianBlur(detectedEdges, detectedEdges, new Size(17, 17), 0);

		Mat inv = new Mat();
		detectedEdges.copyTo(inv);

		Imgproc.threshold(inv, inv, 160, 255, Imgproc.THRESH_BINARY_INV);

		Imgproc.morphologyEx(inv, inv, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));

		// if (true)
		// return inv;
		Imgproc.threshold(detectedEdges, detectedEdges, 117, 255, Imgproc.THRESH_BINARY);// BINARY);//
		// if (true)
		// return detectedEdges;

		Mat inv2 = new Mat(), inv3 = new Mat();
		inv.copyTo(inv2);
		detectedEdges.copyTo(inv3);
		Core.bitwise_not(inv3, inv3);
		// if (true)
		// return inv3;
		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));
		if (true)
			return inv3;
		Imgproc.distanceTransform(inv3, inv3, Imgproc.CV_DIST_L2, 3);

		Imgproc.threshold(inv3, inv3, 3, 255, Imgproc.THRESH_BINARY);// INV?
		inv3.convertTo(inv3, CvType.CV_8UC1);

		Imgproc.distanceTransform(inv3, inv3, Imgproc.CV_DIST_L1, 5);
		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));
		Imgproc.threshold(inv3, inv3, 3, 255, Imgproc.THRESH_BINARY);// INV?

		// SAME
		inv3.convertTo(inv3, CvType.CV_8UC1);
		Imgproc.distanceTransform(inv3, inv3, Imgproc.CV_DIST_L1, 5);
		Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8UC1));
		Imgproc.threshold(inv3, inv3, 1, 255, Imgproc.THRESH_BINARY);// INV?

		if (true)
			return inv3;
		// Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3,
		// CvType.CV_8UC1));
		// Imgproc.dilate(inv3, inv3, matEllipse, new Point(3, 3), 5);
		// Imgproc.morphologyEx(inv3, inv3, Imgproc.MORPH_OPEN, Mat.ones(3, 3,
		// CvType.CV_8UC1));

		// Imgproc.threshold(inv3, inv3, 8, 255, Imgproc.THRESH_BINARY);// INV?
		// if (true)
		// return inv3;
		// Imgproc.dilate(inv3, inv3, matEllipse);

		// Imgproc.dilate(inv2, inv2, matEllipse, // Mat.ones(new Size(5, 5),
		// CvType.CV_8UC1),
		// new Point(3, 3), 8);
		Imgproc.distanceTransform(inv2, inv2, Imgproc.CV_DIST_L1, 3);

		Imgproc.erode(inv2, inv2, matEllipse, // Mat.ones(new Size(5, 5),
												// CvType.CV_8UC1),
				new Point(3, 3), 1);
		// Imgproc.erode(inv2, inv2, matEllipse, // Mat.ones(new Size(5, 5),
		// CvType.CV_8UC1),
		// new Point(3, 3), 1);
		// Imgproc.morphologyEx(inv2, inv2, Imgproc.MORPH_CLOSE, matEllipse);

		// if (true)
		// return inv2;
		// Imgproc.dilate(inv2, inv2, Mat.ones(new Size(5, 5), CvType.CV_8UC1),
		// new Point(3, 3), 3);
		inv2.convertTo(inv2, CvType.CV_8UC1);
		Imgproc.distanceTransform(inv2, inv2, Imgproc.CV_DIST_L1, 3);

		Imgproc.threshold(inv2, inv2, 8, 255, Imgproc.THRESH_BINARY);

		// Imgproc.threshold(inv2, inv2, 8, 255, Imgproc.THRESH_BINARY);//
		// if (true)
		// return inv2;
		// 8//odkom

		// Imgproc.morphologyEx(inv2, inv2, Imgproc.MORPH_TOPHAT, matEllipse);
		// Imgproc.threshold(inv2, inv2, 4, 255, Imgproc.THRESH_BINARY);// usun
		// if (true)
		// return inv2;

		// Imgproc.morphologyEx(inv2, inv2, Imgproc.MORPH_CLOSE, matEllipse);
		// if (true)
		// return inv2;
		// Imgproc.morphologyEx(inv2, inv2, Imgproc.MORPH_CLOSE, matEllipse);

		// if (true)
		// return inv2;
		// Imgproc.cvtColor(inv2, inv2, Imgproc.COLOR_GRAY2BGR);
		// Imgproc.cvtColor(inv2, inv2, CvType.CV_8U);//
		// Imgproc.COLOR_BGR2GRAY);
		// Imgproc.cvtColor(inv2, inv2, 0);
		inv2.convertTo(inv2, CvType.CV_8U);
		inv3.convertTo(inv3, CvType.CV_8U);
		System.out.println(
				"INV " + inv.type() + ",INV2 " + inv2.type() + " , " + CvType.CV_8UC1 + ", " + Imgproc.COLOR_GRAY2BGR);
		// if (true)
		// return inv2;
		// Mat inv3 = new Mat();
		// inv2.copyTo(inv3);
		// Core.normalize(inv3, inv3, 0, 100, Core.NORM_MINMAX, CvType.CV_8UC3);

		// if (true)
		// return inv2;
		// System.out.println("2) INV " + inv.type() + ",INV2 " + inv2.type() +
		// " , " + CvType.CV_8UC1 + ", "
		// + Imgproc.COLOR_GRAY2BGR);
		// if (true)
		// return inv2;

		Mat cont = new Mat();

		inv3.copyTo(cont);
		// if (true)
		// return detectedEdges;
		final List<MatOfPoint> points = new ArrayList<>();
		final Mat hierarchy = new Mat();
		Imgproc.findContours(cont, points, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);// Imgproc.RETR_TREE,
																										// Imgproc.CHAIN_APPROX_SIMPLE);
		Mat markers = Mat.zeros(frame.size(), CvType.CV_32SC1);// CvType.CV_8UC3);//
		// CV_32SC1);//
		// CV_8UC3);//
		// CV_32SC1);

		Scalar sc = new Scalar(128, 255, 0);
		System.out.println("POINTS " + points.size());
		for (int i = 0; i < points.size(); i++) {
			Imgproc.drawContours(markers, points, i, new Scalar((i * 30) % 256, (i + 27) % 256, i), -1);

			Imgproc.drawContours(cont, points, i, new Scalar((i * 30) % 256, (i + 27) % 256, i), -1);
		}
		Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
		// if (true)
		// return markers;
		// if (true)
		// return cont;
		// Mat mark = Mat.zeros(markers.size(), CvType.CV_8UC1);
		// markers.copyTo(mark);
		// System.out.println("cc " + CvType.CV_8U);

		// Imgproc.cvtColor(markers, mark, Imgproc.COLOR_GRAY2BGR);

		// System.out.println(CvType.CV_8UC1 + " - " + mark.type());
		// if (true)
		// return mark;
		Imgproc.cvtColor(inv, inv, Imgproc.COLOR_GRAY2RGB);// CvType.CV_8UC3);

		Mat xx = new Mat();
		// if (true)
		// return detectedEdges;
		inv.copyTo(xx);
		;
		// if (true)
		// return markers;
		Imgproc.watershed(xx, markers);// detectedEdges

		if (true)
			return markers;
		if (true) {
			markers.convertTo(markers, CvType.CV_8UC3);
			// System.out.println("INV c " + inv.channels());
			Imgproc.cvtColor(markers, markers, Imgproc.COLOR_GRAY2BGR);// todo
			// if (true)
			// return markers;
			Core.bitwise_not(inv, inv);
			// if (true)
			// return inv;
			markers.copyTo(markers, inv);// markers);
			return markers;
		}

		Mat wshed2 = new Mat(markers.size(), CvType.CV_8UC3);

		// if (true) {
		// System.out.println("wshed");
		return wshed2;
		// }

	}

	/*
	 * mo¿e siê przydaæ private Mat doSegmentationOK2(Mat frame) { // init Mat
	 * grayImage = new Mat(); Mat detectedEdges = new Mat(frame.size(),
	 * CvType.CV_8UC3); System.out.println("DET " + detectedEdges.type()); //
	 * convert to grayscale Imgproc.cvtColor(frame, grayImage,
	 * Imgproc.COLOR_BGR2GRAY); // Imgproc.distanceTransform(grayImage,
	 * detectedEdges, // Imgproc.CV_DIST_L2, 3);// bw, // // if (true) // return
	 * detectedEdges;
	 * 
	 * Core.normalize(grayImage, detectedEdges, 0, 255, Core.NORM_MINMAX,
	 * CvType.CV_8UC3);
	 * 
	 * // detectedEdges.convertTo(detectedEdges, CvType.CV_8UC3);// Mat x = new
	 * Mat(); // System.out.println("DDD " + grayImage.dims());
	 * 
	 * // System.out.println(CvType.CV_8U + " DET2 " + detectedEdges.type()); //
	 * reduce noise with a 3x3 kernel // Imgproc.blur(grayImage, detectedEdges,
	 * new Size(3, 3));
	 * 
	 * // dist, // CV_DIST_L2, // 3); Mat matEllipse = Mat.ones(new Size(5, 5),
	 * 0); matEllipse.put(0, 0, 0); matEllipse.put(0, 1, 0); matEllipse.put(0,
	 * 3, 0); matEllipse.put(0, 4, 0); matEllipse.put(4, 0, 0);
	 * matEllipse.put(4, 1, 0); matEllipse.put(4, 3, 0); matEllipse.put(4, 4,
	 * 0);
	 * 
	 * Imgproc.threshold(detectedEdges, detectedEdges, 60, 255,
	 * Imgproc.THRESH_BINARY); Imgproc.morphologyEx(detectedEdges,
	 * detectedEdges, Imgproc.MORPH_OPEN, matEllipse);
	 * Imgproc.erode(detectedEdges, detectedEdges, matEllipse); //
	 * Imgproc.morphologyEx(detectedEdges, detectedEdges, //
	 * Imgproc.MORPH_GRADIENT, matEllipse); if (true) return detectedEdges;
	 */

	private Scalar getScalar(int i) {
		int a = 0, b = 0, c = 0;
		if (i < 256) {
			c = i;
		} else if (i < 256 * 256) {
			int d = (int) Math.ceil(((double) i) / 256);
			b = d;
			c = i - d;
		}
		Scalar sc = new Scalar(a, b, c);
		return sc;
	}

	/**
	 * Action triggered when the Canny checkbox is selected
	 * 
	 */
	@FXML
	protected void cannySelected() {
		// check whether the other checkbox is selected and deselect it
		// if (this.dilateErode.isSelected()) {
		// this.dilateErode.setSelected(false);
		// this.inverse.setDisable(true);
		// }

		// enable the threshold slider
		/*
		 * if (this.canny.isSelected()) this.threshold.setDisable(false); else
		 * this.threshold.setDisable(true);
		 */
		// now the capture can start
		// this.cameraButton.setDisable(false);
	}

	/**
	 * Action triggered when the "background removal" checkbox is selected
	 */
	@FXML
	protected void dilateErodeSelected() {
		// check whether the canny checkbox is selected, deselect it and disable
		// its slider
		if (this.canny.isSelected()) {
			this.canny.setSelected(false);
			this.threshold.setDisable(true);
		}

		if (this.dilateErode.isSelected())
			this.inverse.setDisable(false);
		else
			this.inverse.setDisable(true);

		// now the capture can start
		this.cameraButton.setDisable(false);
	}

	/**
	 * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
	 * 
	 * @param frame
	 *            the {@link Mat} representing the current frame
	 * @return the {@link Image} to show
	 */
	private Image mat2Image(Mat frame) {
		// create a temporary buffer
		MatOfByte buffer = new MatOfByte();
		// encode the frame in the buffer, according to the PNG format
		Imgcodecs.imencode(".png", frame, buffer);
		// build and return an Image created from the image encoded in the
		// buffer
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}

}
