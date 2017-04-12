
//package br.com.pontov.frame;

//import com.atul.JavaOpenCV.Imshow;
//import com.atul.JavaOpenCV.Imshow;

public class Watershed {
	/*
	 * public static void main(String[] args) {
	 * System.loadLibrary("opencv_java249"); Mat goldimgc = Highgui.imread(
	 * "/Users/alexandrermello/Documents/GoldenImages/PCB_ID15V0/GoldenImages/Img4Defect0a.png"
	 * ); Mat gray = new Mat(goldimgc.size(),CvType.CV_8UC3);
	 * 
	 * Imgproc.cvtColor(goldimgc, gray, Imgproc.COLOR_RGB2GRAY);
	 * 
	 * 
	 * Imgproc.threshold(gray, gray, 100, 255, Imgproc.THRESH_BINARY_INV +
	 * Imgproc.THRESH_OTSU);
	 * 
	 * // Imshow ima1 = new Imshow("Drawing"); //ima1.Window.setResizable(true);
	 * // ima1.showImage(gray);
	 * 
	 * 
	 * //Noise removal Mat kernel =
	 * Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3,3));
	 * //19,19 Mat ret = new Mat(goldimgc.size(),CvType.CV_8U);
	 * Imgproc.morphologyEx(gray, ret, Imgproc.MORPH_OPEN, kernel);
	 * 
	 * // Imshow ima2 = new Imshow("Drawing2"); //
	 * ima2.Window.setResizable(true); // ima2.showImage(ret);
	 * 
	 * //Sure background area Mat sure_bg = new
	 * Mat(goldimgc.size(),CvType.CV_8U); Imgproc.dilate(ret,sure_bg,new
	 * Mat(),new Point(-1,-1),3); Imgproc.threshold(sure_bg,sure_bg,1,
	 * 128,Imgproc.THRESH_BINARY_INV);
	 * 
	 * Mat sure_fg = new Mat(goldimgc.size(),CvType.CV_8U);
	 * Imgproc.erode(gray,sure_fg,new Mat(),new Point(-1,-1),2); /* Mat fg = new
	 * Mat(goldimgc.size(),CvType.CV_8U); Imgproc.erode(goldimgc,fg,new
	 * Mat(),new Point(-1,-1),2);
	 * 
	 * Mat bg = new Mat(goldimgc.size(),CvType.CV_8U);
	 * Imgproc.dilate(goldimgc,bg,new Mat(),new Point(-1,-1),3);
	 * Imgproc.threshold(bg,bg,1, 128,Imgproc.THRESH_BINARY_INV);
	 */
	/*
	 * Mat markers = new Mat(goldimgc.size(),CvType.CV_8U, new Scalar(0));
	 * Core.add(sure_fg, sure_bg, markers);
	 * 
	 * // Imshow ima3 = new Imshow("Drawing3");
	 * //ima3.Window.setResizable(true); //ima3.showImage(markers);
	 * 
	 * markers.convertTo(markers, CvType.CV_32SC1);
	 * 
	 * Imgproc.watershed(goldimgc, markers); Core.convertScaleAbs(markers,
	 * markers);
	 * 
	 * 
	 * //Imshow dra4 = new Imshow("Drawing4"); //
	 * dra4.Window.setResizable(true); // dra4.showImage(markers);
	 * 
	 * 
	 * //------------------------------ACHAR
	 * CONTORNO------------------------------------------------------
	 * ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>(); int thresh
	 * =1; double largest_contour=0; int largest_contour_index=0;
	 * 
	 * double largest_contour2=0; int largest_contour_index2=0;
	 * 
	 * for (int i=0;i<250;i++) { Imgproc.Canny(markers, markers, i,255);
	 * 
	 * Imgproc.findContours(markers, contours, new Mat(),
	 * Imgproc.RETR_EXTERNAL+Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
	 * //RETR_LIST
	 * 
	 * } for( int x = 0; x<contours.size(); x++ ) // iterate through each
	 * contour. {
	 * 
	 * double a=Imgproc.contourArea(contours.get(x)); // Find the area of
	 * contour
	 * 
	 * 
	 * if(a>largest_contour){ largest_contour=a; //SAIDA: MAIOR CONTORNO
	 * largest_contour_index=x; //Store the index of largest contour }
	 * 
	 * 
	 * }
	 * 
	 * for( int x = 0; x<contours.size(); x++ ) // iterate through each contour.
	 * {
	 * 
	 * double a=Imgproc.contourArea(contours.get(x)); // Find the area of
	 * contour
	 * 
	 * 
	 * if(a>largest_contour2 & a<largest_contour){ largest_contour2=a; //SAIDA:
	 * SEGUNDO MAIOR CONTORNO largest_contour_index2=x; //Store the index of
	 * largest contour }
	 * 
	 * }
	 * 
	 * System.out.print(largest_contour); System.out.print(largest_contour2);
	 * 
	 * boolean key=false; if ((largest_contour - largest_contour2) <
	 * 0.3*largest_contour) { key=true; }
	 * 
	 * 
	 * //---------------------VETORES E
	 * MOMENTOS1-------------------------------- if (key == false) {
	 * Vector<Moments> mu = new Vector<Moments>(contours.size()); for(int i=0;
	 * i<contours.size(); i++) { mu.add(Imgproc.moments( (contours.get(i)),
	 * false )); }
	 * 
	 * 
	 * Vector<Point> mc = new Vector<Point>( contours.size() ); // valor de
	 * centroid for( int i = 0; i < contours.size(); i++ ) { mc.add(new Point(
	 * (mu.get(i)).get_m10()/mu.get(i).get_m00(),
	 * mu.get(i).get_m01()/mu.get(i).get_m00() )); }
	 * 
	 * Mat drawing = Mat.zeros(markers.size(), largest_contour_index);
	 * 
	 * Mat hierarchy = new Mat();
	 * 
	 * 
	 * for( int i = 0; i< contours.size(); i++ ) { if (mu.get(i).get_m00() >
	 * (largest_contour-20)) //-50 {
	 * 
	 * Imgproc.drawContours(markers, contours, i, new
	 * Scalar(255,0,0),1,8,hierarchy,1,new Point());
	 * 
	 * 
	 * Core.circle(markers, mc.get(i), 4, new Scalar(255,0,0), 3,0, 0); //4 } }
	 * Imshow dra5 = new Imshow("Contorno"); dra5.Window.setResizable(true);
	 * dra5.showImage(markers); } //---------------------VETORES E
	 * MOMENTOS2-------------------------------- if (key==true) {
	 * Vector<Moments> mu2 = new Vector<Moments>(contours.size()); for(int i=0;
	 * i<contours.size(); i++) { mu2.add(Imgproc.moments( (contours.get(i)),
	 * false )); }
	 * 
	 * 
	 * Vector<Point> mc2 = new Vector<Point>( contours.size() ); // valor de
	 * centroid for( int i = 0; i < contours.size(); i++ ) { mc2.add(new Point(
	 * (mu2.get(i)).get_m10()/mu2.get(i).get_m00(),
	 * mu2.get(i).get_m01()/mu2.get(i).get_m00() )); }
	 * 
	 * Mat drawing2 = Mat.zeros(markers.size(), largest_contour_index2);
	 * 
	 * Mat hierarchy2 = new Mat();
	 * 
	 * 
	 * for( int i = 0; i< contours.size(); i++ ) { if (mu2.get(i).get_m00() >
	 * (largest_contour2-20)) //-50 {
	 * 
	 * Imgproc.drawContours(markers, contours, i, new
	 * Scalar(255,0,0),1,8,hierarchy2,1,new Point());
	 * 
	 * 
	 * Core.circle(markers, mc2.get(i), 4, new Scalar(255,0,0), 3,0, 0); //4 } }
	 * Imshow dra9 = new Imshow("Contorno2"); dra9.Window.setResizable(true);
	 * dra9.showImage(markers);
	 * 
	 * 
	 * } //---------------------------------
	 * 
	 * 
	 * 
	 * if(key==false) { MatOfPoint2f approxCurve = new MatOfPoint2f(); Rect
	 * rect=new Rect(); MatOfPoint2f contour2f = new MatOfPoint2f(
	 * contours.get(largest_contour_index).toArray() ); RotatedRect
	 * rectrot=null; rectrot = Imgproc.minAreaRect(contour2f);
	 * 
	 * Rect box = rectrot.boundingRect(); // System.out.print(box);
	 * 
	 * //Desenhar o retangulo // Core.rectangle(drawing, box.tl(), box.br(), new
	 * Scalar(0,0,255),1);
	 * 
	 * float W1,L1,elo1; W1 = box.width; L1 = box.height; elo1=1-W1/L1; //SAIDA:
	 * elongation // arr.setElongation(elo1);
	 * 
	 * //System.out.print(elo1); //ELLIPSE
	 * FIT----------------------------------------------------------------------
	 * ------------------ RotatedRect elip = Imgproc.fitEllipse(contour2f);
	 * 
	 * //Desenhar a elipse // Core.ellipse(drawing, elip, new Scalar(255,0,0));
	 * Core.ellipse(markers, rectrot, new Scalar(255,0,0), 1);
	 * 
	 * Imshow dra6 = new Imshow("Elipse"); dra6.Window.setResizable(true);
	 * dra6.showImage(markers);
	 * 
	 * float angulo = (float) elip.angle; //SAIDA: angulo de rotação da imagem
	 * //System.out.print(angulo); //arr.setAngle(angulo); }
	 * 
	 * if(key==true) { MatOfPoint2f approxCurve = new MatOfPoint2f(); Rect
	 * rect=new Rect(); RotatedRect rectrot=null; MatOfPoint2f contour2f = new
	 * MatOfPoint2f( contours.get(largest_contour_index).toArray() );
	 * MatOfPoint2f contour2f2 = new MatOfPoint2f(
	 * contours.get(largest_contour_index2).toArray() );
	 * contour2f.push_back(contour2f2);
	 * 
	 * rectrot = Imgproc.minAreaRect(contour2f);
	 * 
	 * Rect box = rectrot.boundingRect(); // System.out.print(box);
	 * 
	 * //Desenhar o retangulo // Core.rectangle(drawing, box.tl(), box.br(), new
	 * Scalar(0,0,255),1);
	 * 
	 * float W1,L1,elo1; W1 = box.width; L1 = box.height; elo1=1-W1/L1; //SAIDA:
	 * elongation // arr.setElongation(elo1);
	 * 
	 * //System.out.print(elo1); //ELLIPSE
	 * FIT----------------------------------------------------------------------
	 * ------------------ RotatedRect elip = Imgproc.fitEllipse(contour2f);
	 * 
	 * //Desenhar a elipse // Core.ellipse(drawing, elip, new Scalar(255,0,0));
	 * Core.ellipse(markers, rectrot, new Scalar(255,0,0), 1);
	 * 
	 * Imshow dra6 = new Imshow("Elipse"); dra6.Window.setResizable(true);
	 * dra6.showImage(markers);
	 * 
	 * float angulo = (float) elip.angle; //SAIDA: angulo de rotação da imagem
	 * //System.out.print(angulo); //arr.setAngle(angulo); }
	 * 
	 * }
	 */}
