package com.cristian.detectexam;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.apache.commons.lang3.tuple.Pair;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class Circle {

    private final List<Pair<Double, Integer>> CIRCLES_ORDERED_LIST;
    private final List<Character> ANSWERS_DETECTED_LIST;
    private static final double MIN_DISTANCE = 25.0;
    private int rectIndex;
    private int numberToChar;

    public Circle() {
        this.CIRCLES_ORDERED_LIST = new ArrayList<>();
        this.ANSWERS_DETECTED_LIST = new ArrayList<>();
    }

    public void detectCircles(Mat src, List<Rect> orderedListRects) {

        for (Rect rect : orderedListRects) {
            CIRCLES_ORDERED_LIST.clear();
            Mat subMat = src.submat(rect);
            Mat gray = new Mat();
            Imgproc.cvtColor(subMat, gray, Imgproc.COLOR_BGR2GRAY);

            Mat blurred = new Mat();
            Imgproc.medianBlur(gray, blurred, 3);

            Mat circles = new Mat();
            Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, 1.0, (double) blurred.rows() / 50, 150.0, 30.0, 15, 20);

            for (int x = 0; x < circles.cols(); x++) {
                double[] circleData = circles.get(0, x);
                if (circleData == null) {
                    continue;
                }
                double currentX = circleData[0];
                double currentY = circleData[1];

                if (!isDuplicate(currentX, currentY, CIRCLES_ORDERED_LIST, circles)) {
                    CIRCLES_ORDERED_LIST.add(Pair.of(currentX, x));
                }
            }

            organizeCirclesByType(subMat, circles, CIRCLES_ORDERED_LIST, rectIndex);
            
            // Guardar la imagen después de dibujar los círculos
            Imgcodecs.imwrite("I:\\FPDAM\\PRACTICAS_EMPRESA\\recursosOpencv\\examenDetectado.jpg", src);
            rectIndex++;
        }
    }

    public boolean isDuplicate(double currentX, double currentY, List<Pair<Double, Integer>> circlesOrdered, Mat circles) {
        for (Pair<Double, Integer> pair : circlesOrdered) {
            double storedX = circles.get(0, pair.getRight())[0];
            double storedY = circles.get(0, pair.getRight())[1];

            double distance = Math.sqrt(Math.pow(currentX - storedX, 2) + Math.pow(currentY - storedY, 2));
            if (distance < MIN_DISTANCE) {
                return true;
            }
        }
        return false;
    }

    private void organizeCirclesByType(Mat subMat, Mat circles, List<Pair<Double, Integer>> circlesOrdered, int rectIndex) {
        int totalCircles = circlesOrdered.size();
        int coordinateDifferenceThreshold = 2;
        switch (totalCircles) {

            case 4:
                circlesOrdered.sort(new Comparator<Pair<Double, Integer>>() {
                    @Override
                    public int compare(Pair<Double, Integer> p1, Pair<Double, Integer> p2) {
                        double x1 = circles.get(0, p1.getRight())[0];
                        double x2 = circles.get(0, p2.getRight())[0];
                        if (Math.abs(x1 - x2) > coordinateDifferenceThreshold) {
                            return Double.compare(x1, x2);
                        }
                        return Integer.compare(p1.getRight(), p2.getRight());
                    }
                });
                detectAnswersFromExam(subMat, circles, circlesOrdered);
                break;
            case 26:
                circlesOrdered.sort(new Comparator<Pair<Double, Integer>>() {
                    @Override
                    public int compare(Pair<Double, Integer> p1, Pair<Double, Integer> p2) {
                        double y1 = circles.get(0, p1.getRight())[1];
                        double y2 = circles.get(0, p2.getRight())[1];
                        if (Math.abs(y1 - y2) > coordinateDifferenceThreshold) {
                            return Double.compare(y1, y2);
                        } else {
                            double x1 = circles.get(0, p1.getRight())[0];
                            double x2 = circles.get(0, p2.getRight())[0];
                            return Double.compare(x1, x2);
                        }
                    }
                });
                detectAndSetIdLetter(subMat, circles, circlesOrdered, rectIndex);
                break;
            case 30:
                circlesOrdered.sort(new Comparator<Pair<Double, Integer>>() {
                    @Override
                    public int compare(Pair<Double, Integer> p1, Pair<Double, Integer> p2) {
                        double x1 = circles.get(0, p1.getRight())[0];
                        double x2 = circles.get(0, p2.getRight())[0];
                        if (Math.abs(x1 - x2) > coordinateDifferenceThreshold) {
                            return Double.compare(x1, x2);
                        } else {
                            double y1 = circles.get(0, p1.getRight())[1];
                            double y2 = circles.get(0, p2.getRight())[1];
                            return Double.compare(y1, y2);
                        }
                    }
                });
                classifyAndDetectAnswers(subMat, circles, circlesOrdered, rectIndex);
                break;
            case 80:
                circlesOrdered.sort(new Comparator<Pair<Double, Integer>>() {
                    @Override
                    public int compare(Pair<Double, Integer> p1, Pair<Double, Integer> p2) {
                        double x1 = circles.get(0, p1.getRight())[0];
                        double x2 = circles.get(0, p2.getRight())[0];
                        if (Math.abs(x1 - x2) > coordinateDifferenceThreshold) {
                            return Double.compare(x1, x2);
                        } else {
                            double y1 = circles.get(0, p1.getRight())[1];
                            double y2 = circles.get(0, p2.getRight())[1];
                            return Double.compare(y1, y2);
                        }
                    }
                });
                classifyAndDetectAnswers(subMat, circles, circlesOrdered, rectIndex);
                break;
            default:
                System.out.println("Número de círculos no manejado en " + rectIndex + ": " + totalCircles);
                break;
        }
    }

    public void detectAnswersFromExam(Mat subMat, Mat circles, List<Pair<Double, Integer>> circlesOrdered) {
        char[] letterAnswers = {'A', 'B', 'C', 'D'};
        int letterIndex = 0;
        List<Character> answeredQuestion = new ArrayList<>();

        for (Pair<Double, Integer> pair : circlesOrdered) {
            if (pair.getRight() >= circles.cols()) {
                continue;
            }
            double[] c = circles.get(0, pair.getRight());
            if (c == null) {
                continue;
            }
            Point center = new Point(Math.round(c[0]), Math.round(c[1]));
            int radius = (int) Math.round(c[2]);

            Rect roiRect = new Rect((int) center.x - radius, (int) center.y - radius, radius * 2, radius * 2);
            Mat roi = new Mat(subMat, roiRect);
            Scalar mean = Core.mean(roi);

            char letterAnswer = letterAnswers[letterIndex];

            if (mean.val[0] < 200) {
                answeredQuestion.add(letterAnswer);
                Imgproc.circle(subMat, center, radius, new Scalar(0, 255, 0), 3, 8, 0);
            }

            letterIndex++;
            if (letterIndex >= 4) {
                letterIndex = 0;
            }
        }

        if (answeredQuestion.isEmpty()) {
            ANSWERS_DETECTED_LIST.add('N');
        } else if (answeredQuestion.size() == 1) {
            ANSWERS_DETECTED_LIST.add(answeredQuestion.get(0));
        } else if (answeredQuestion.size() == 2) {
            ANSWERS_DETECTED_LIST.add('N');
        } else if (answeredQuestion.size() > 2) {
            ANSWERS_DETECTED_LIST.add('N');
        }
        
        DetectExam.setDetectedAnswers(ANSWERS_DETECTED_LIST);
    }

    public void detectAndSetIdLetter(Mat subMat, Mat circles, List<Pair<Double, Integer>> circlesOrdered, int rectIndex) {
        int counter = 0;

        for (Pair<Double, Integer> pair : circlesOrdered) {
            if (pair.getRight() >= circles.cols()) {
                continue;
            }
            double[] c = circles.get(0, pair.getRight());
            if (c == null) {
                continue;
            }
            Point center = new Point(Math.round(c[0]), Math.round(c[1]));
            int radius = (int) Math.round(c[2]);
            Rect roiRect = new Rect((int) center.x - radius, (int) center.y - radius, radius * 2, radius * 2);
            Mat roi = new Mat(subMat, roiRect);
            Scalar mean = Core.mean(roi);
            numberToChar = counter + 1;

            if (mean.val[0] < 200) {
                if (rectIndex == 0) {
                    DetectExam.setNumberToNieLetter(numberToChar);
                } else {
                    DetectExam.setNumberToDniLetter(numberToChar);

                }
                Imgproc.circle(subMat, center, radius, new Scalar(0, 255, 0), 3, 8, 0);
            } 
            counter++;
        }
    }

    public void classifyAndDetectAnswers(Mat subMat, Mat circles, List<Pair<Double, Integer>> circlesOrdered, int rectIndex) {

        List<Integer> foundNumbers = new ArrayList<>();

        for (int i = 0; i < circlesOrdered.size(); i += 10) {
            List<Pair<Double, Integer>> subList = circlesOrdered.subList(i, Math.min(i + 10, circlesOrdered.size()));
            int counter = 0;
            for (Pair<Double, Integer> pair : subList) {
                if (pair.getRight() >= circles.cols()) {
                    continue;
                }
                double[] c = circles.get(0, pair.getRight());
                if (c == null) {
                    continue;
                }
                Point center = new Point(Math.round(c[0]), Math.round(c[1]));
                int radius = (int) Math.round(c[2]);
                Rect roiRect = new Rect((int) center.x - radius, (int) center.y - radius, radius * 2, radius * 2);
                Mat roi = new Mat(subMat, roiRect);
                Scalar mean = Core.mean(roi);

                if (mean.val[0] < 200) {
                    foundNumbers.add(counter);
                    if (rectIndex < 2) {
                        DetectExam.setNumbersOfIdentification(foundNumbers);

                    } else {
                        DetectExam.setNumbersOfExam(foundNumbers);
                    }

                    Imgproc.circle(subMat, center, radius, new Scalar(0, 255, 0), 3, 8, 0);
                }
                counter++;
            }
        }
    }

    //Debugging method ,actually doesnt work
    public void drawCircles(Mat subMat, Mat circles, List<Character> answeredQuestion) {
        char[] letterAnswers = {'A', 'B', 'C', 'D'};
        int letterIndex = 0;

        for (Pair<Double, Integer> pair : CIRCLES_ORDERED_LIST) {
            if (pair.getRight() >= circles.cols()) {
                continue;
            }
            double[] c = circles.get(0, pair.getRight());
            if (c == null) {
                continue;
            }
            Point center = new Point(Math.round(c[0]), Math.round(c[1]));
            int radius = (int) Math.round(c[2]);

            char letterAnswer = letterAnswers[letterIndex];
            Imgproc.putText(subMat, String.valueOf(letterAnswer), new Point(center.x - radius, center.y), Core.Formatter_FMT_DEFAULT, 1, new Scalar(0, 0, 255), 2);

            if (answeredQuestion.contains(letterAnswer)) {
                Imgproc.circle(subMat, center, radius, new Scalar(0, 255, 0), 3, 8, 0);
            }
            letterIndex++;
            if (letterIndex >= 4) {
                letterIndex = 0;
            }

            HighGui.imshow("Rectángulos detectados", subMat);
            HighGui.waitKey();
        }

    }
}
