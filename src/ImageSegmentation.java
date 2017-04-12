
import java.io.File;
import java.util.ArrayList;

import org.opencv.core.Core;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class ImageSegmentation extends Application {

	@Override
	public void start(Stage primaryStage) {
		try {

			BorderPane root = (BorderPane) FXMLLoader.load(getClass().getResource("ImageSeg.fxml"));
			Scene scene = new Scene(root, 1000, 800);
			primaryStage.setScene(scene);

			primaryStage.show();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		Controller.SEGMENTATION_TYPE = Controller.KMEANS;
		ArrayList<String> filenames = new ArrayList<>();
		// filenames.add("5341_13_001");
		// filenames.add("5412_13_001");
		// filenames.add("4519_13_002");
		filenames.add("4638_13_001");
		// filenames.add("4975_13_001");
		// filenames.add("5409_13_001");
		// filenames.add("4517_13_001");
		// filenames.add("4978_13_001");
		Controller cc = new Controller();
		// cc.createMaps("C:\\Users\\Olusiak\\Downloads\\Ki67
		// baza-20161012T160623Z\\Ki67 baza\\ROI wyciête\\dens");
		// cc.createROIwithInfo();
		String mainPath = "C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67 baza\\ROI wyciête\\densities\\chosen\\chosen\\";
		String path = "C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67 baza\\ROI wyciête\\densities\\chosen\\deconv\\";
		File dir = new File(mainPath);
		for (File fileEntry : dir.listFiles()) {

			// for (String f : filenames) {
			String filename = fileEntry.getName();// "5341_13_001";
			System.out.println("FI" + filename);

			Controller.redStainsFilename = path + filename + "-(Colour_2).jpg";
			System.out.println("red " + Controller.redStainsFilename);
			Controller.blueStainsFilename = path + filename + "-(Colour_1).jpg";
			filename = filename.replaceAll(".png", "");
			String pathDens = "C:\\Users\\Olusiak\\Downloads\\Ki67 baza-20161012T160623Z\\Ki67 baza\\ROI wyciête\\dens\\"
					+ filename + "Markers_Counter Window -.png_density.png";

			Controller.densityMap = pathDens;
			Controller.infoPic = Controller.densityMap;// path + filename +
														// "Markers.png";
			Controller.filename = filename;
			Controller c = new Controller();
			c.debug = true;
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			if (c.debug)
				c.load();
			else
				launch(args);
		}

	}
}