package com.cristian.detectexam;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

public class Rectangle {

    private Mat src;
    private List<Rect> detectedListIdColumns;
    private List<Rect> sortedListCells;
    private List<Rect> detectedListExternalRects;
    private List<Rect> totalListRects;

    public void setTotalListRects(List<Rect> totalListRects) {
        this.totalListRects = totalListRects;
    }

    public Rectangle(String imageUrl) {
        this.src = Imgcodecs.imread(imageUrl);
        this.detectedListIdColumns = new ArrayList<>();
        this.sortedListCells = new ArrayList<>();
        this.detectedListExternalRects = new ArrayList<>();
        this.totalListRects = new ArrayList<>();
    }

    public boolean loadImage() {
        if (src.empty()) {
            System.out.println("No se puede cargar la imagen");
            return false;
        }
        return true;
    }

    public void applyFiltersAndFindExternalRects() {
        Mat srcWithFilters = new Mat();
        Imgproc.cvtColor(src, srcWithFilters, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(srcWithFilters, srcWithFilters, 3);
        Imgproc.adaptiveThreshold(srcWithFilters, srcWithFilters, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 1);
        Imgproc.Canny(srcWithFilters, srcWithFilters, 50, 150);
        Imgcodecs.imwrite("I:\\FPDAM\\PRACTICAS_EMPRESA\\recursosOpencv\\canny.jpg", srcWithFilters);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(srcWithFilters, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            double peri = Imgproc.arcLength(contour2f, true);
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, 0.02 * peri, true);

            if (approx.total() == 4 && Imgproc.isContourConvex(new MatOfPoint(approx.toArray()))) {
                Rect rect = Imgproc.boundingRect(new MatOfPoint(approx.toArray()));

                detectedListExternalRects.add(rect);
            }
        }
        
        if (detectedListExternalRects.size() < 6) {
            System.out.println("Error: La imagen no se ha tomado adecuadamente. Haz otra foto.");
            System.exit(0);
        }
    }

    public void organizeExternalRects() {

        detectedListExternalRects.sort(new Comparator<Rect>() {
            @Override
            public int compare(Rect r1, Rect r2) {
                return Double.compare(r2.area(), r1.area());
            }
        });

        detectedListExternalRects = detectedListExternalRects.subList(0, 6);

        detectedListExternalRects.sort(new Comparator<Rect>() {
            @Override
            public int compare(Rect r1, Rect r2) {
                int yCompare = Integer.compare(r1.y, r2.y);
                if (yCompare != 0) {
                    return yCompare;
                }
                return Integer.compare(r1.x, r2.x);

            }
        });
        
        applyFiltersAndFindInternalRects();
    }
    
    
    private void applyFiltersAndFindInternalRects() {
       for (int i = 0; i < detectedListExternalRects.size(); i++) {
            Rect externalRect = detectedListExternalRects.get(i);
            Mat externalRectMat = src.submat(externalRect);
            Imgproc.cvtColor(externalRectMat, externalRectMat, Imgproc.COLOR_BGR2GRAY);
            Imgproc.medianBlur(externalRectMat, externalRectMat, 3);
            Imgproc.adaptiveThreshold(externalRectMat, externalRectMat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 1);

            List<MatOfPoint> internalContours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(externalRectMat, internalContours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
            
            for (MatOfPoint contour : internalContours) {
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                double peri = Imgproc.arcLength(contour2f, true);
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(contour2f, approx, 0.02 * peri, true);

                if (approx.total() == 4 && Imgproc.isContourConvex(new MatOfPoint(approx.toArray()))) {
                    Rect internalRect = Imgproc.boundingRect(new MatOfPoint(approx.toArray()));

                    internalRect.x += externalRect.x;
                    internalRect.y += externalRect.y;

                    if (i <= 1) {
                        if (!isDuplicate(internalRect, detectedListIdColumns) && internalRect.height > (externalRect.height / 3) && internalRect.height > internalRect.width) {
                            detectedListIdColumns.add(internalRect);
                        }

                    } else if (!isDuplicate(internalRect, sortedListCells) && internalRect.width > internalRect.height) {
                        sortedListCells.add(internalRect);

                        if (sortedListCells.size() > 40) {
                            sortedListCells.sort(new Comparator<Rect>() {
                                @Override
                                public int compare(Rect r1, Rect r2) {
                                    return Double.compare(r2.area(), r1.area());
                                }
                            });
                            sortedListCells = sortedListCells.subList(0, 40);
                        }
                    }
                }
            }
        }
        organizeInternalRectangles();
    }

    private void organizeInternalRectangles() {
        detectedListIdColumns.sort((r1, r2) -> {
            int yCompare = Integer.compare(r1.y, r2.y);
            if (yCompare != 0) {
                return yCompare;
            }
            return Integer.compare(r1.x, r2.x);
        });

        for (Rect rect : detectedListIdColumns) {
            totalListRects.add(rect);

        }

        int threshold = 2;
        sortedListCells.sort((r1, r2) -> {
            int xCompare = Integer.compare(r1.x, r2.x);
            if (Math.abs(r1.x - r2.x) > threshold) {
                return xCompare;
            }
            return Integer.compare(r1.y, r2.y);
        });

        for (Rect rect : sortedListCells) {
            totalListRects.add(rect);

        }
    }
     private boolean isDuplicate(Rect rect, List<Rect> detected) {
        for (Rect pos : detected) {
            if (Math.abs(rect.x - pos.x) < 4 && Math.abs(rect.y - pos.y) < 4) {
                return true;
            }
        }
        return false;
    }

    //Debugging method 
    private void drawListRects(List<Rect> sortedListRects) {
        for (int i = 0; i < sortedListRects.size(); i++) {
            Rect rect = sortedListRects.get(i);
            Imgproc.rectangle(src, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255), 2);
            Imgproc.putText(src, String.valueOf(i + 1), new Point(rect.x, rect.y - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 0, 255), 2);
            Imgcodecs.imwrite("I:\\FPDAM\\PRACTICAS_EMPRESA\\recursosOpencv\\respuestas\\rect" + (i + 1) + ".jpg", src);
        }
        HighGui.imshow("rectangulos detectados", src);
        HighGui.waitKey();
    }

    public List<Rect> getTotalListRects() {
        return totalListRects;
    }

    public Mat getSrc() {
        return src;
    }


}
