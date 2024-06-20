package com.cristian.detectexam;

import org.opencv.core.*;
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

    private final List<Pair<Double, Integer>> circlesOrdered;
    private int rectIndex;

    private int numberToChar;
    private final List<Character> answersList;

    public int getNumberOfLetter() {
        return numberToChar;
    }

    public List<Character> getAnswersList() {
        return answersList;
    }

    public Circle() {
        this.circlesOrdered = new ArrayList<>();
        this.answersList = new ArrayList<>();
    }

    public void detectCircles(Mat src, List<Rect> orderedListRects) {
        
        for (Rect rect : orderedListRects) {
            circlesOrdered.clear();
            Mat subMat = src.submat(rect);
            Mat gray = new Mat();
            Imgproc.cvtColor(subMat, gray, Imgproc.COLOR_BGR2GRAY);

            Mat blurred = new Mat();
            Imgproc.medianBlur(gray, blurred, 3);

            Mat circles = new Mat();
            Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, 1.0, (double) blurred.rows() / 50, 150.0, 30.0, 15, 20);

            double minDistance = 25;
            for (int x = 0; x < circles.cols(); x++) {
                double[] c = circles.get(0, x);
                if (c == null) {
                    continue;
                }
                double xCoordinate = c[0];
                double yCoordinate = c[1];

                boolean isDuplicate = false;
                for (Pair<Double, Integer> pair : circlesOrdered) {
                    double existingX = circles.get(0, pair.getRight())[0];
                    double existingY = circles.get(0, pair.getRight())[1];

                    double distance = Math.sqrt(Math.pow(xCoordinate - existingX, 2) + Math.pow(yCoordinate - existingY, 2));
                    if (distance < minDistance) {
                        isDuplicate = true;
                        break;
                    }
                }

                if (!isDuplicate) {
                    circlesOrdered.add(Pair.of(xCoordinate, x));
                }

            }

            organizeCirclesByType(subMat, circles, circlesOrdered, rectIndex);
            rectIndex++;

        }
    }

    private void organizeCirclesByType(Mat subMat, Mat circles, List<Pair<Double, Integer>> circlesOrdered, int rectIndex) {
        int totalCircles = circlesOrdered.size();
        int threshold = 2;
        switch (totalCircles) {

            case 4:
                circlesOrdered.sort(new Comparator<Pair<Double, Integer>>() {
                    @Override
                    public int compare(Pair<Double, Integer> p1, Pair<Double, Integer> p2) {
                        double x1 = circles.get(0, p1.getRight())[0];
                        double x2 = circles.get(0, p2.getRight())[0];
                        if (Math.abs(x1 - x2) > threshold) {
                            return Double.compare(x1, x2);
                        }
                        return Integer.compare(p1.getRight(), p2.getRight());
                    }
                });
                detectAnswersQuestions(subMat, circles, circlesOrdered);
                break;
            case 26:
                circlesOrdered.sort(new Comparator<Pair<Double, Integer>>() {
                    @Override
                    public int compare(Pair<Double, Integer> p1, Pair<Double, Integer> p2) {
                        double y1 = circles.get(0, p1.getRight())[1];
                        double y2 = circles.get(0, p2.getRight())[1];
                        if (Math.abs(y1 - y2) > threshold) {
                            return Double.compare(y1, y2);
                        } else {
                            double x1 = circles.get(0, p1.getRight())[0];
                            double x2 = circles.get(0, p2.getRight())[0];
                            return Double.compare(x1, x2);
                        }
                    }
                });
                detectAnswersToAlphabet(subMat, circles, circlesOrdered, rectIndex);
                break;
            case 30:
                circlesOrdered.sort(new Comparator<Pair<Double, Integer>>() {
                    @Override
                    public int compare(Pair<Double, Integer> p1, Pair<Double, Integer> p2) {
                        double x1 = circles.get(0, p1.getRight())[0];
                        double x2 = circles.get(0, p2.getRight())[0];
                        if (Math.abs(x1 - x2) > threshold) {
                            return Double.compare(x1, x2);
                        } else {
                            double y1 = circles.get(0, p1.getRight())[1];
                            double y2 = circles.get(0, p2.getRight())[1];
                            return Double.compare(y1, y2);
                        }
                    }
                });
                detectAnswers(subMat, circles, circlesOrdered, 0);
                break;
            case 80:
                circlesOrdered.sort(new Comparator<Pair<Double, Integer>>() {
                    @Override
                    public int compare(Pair<Double, Integer> p1, Pair<Double, Integer> p2) {
                        double x1 = circles.get(0, p1.getRight())[0];
                        double x2 = circles.get(0, p2.getRight())[0];
                        if (Math.abs(x1 - x2) > threshold) {
                            return Double.compare(x1, x2);
                        } else {
                            double y1 = circles.get(0, p1.getRight())[1];
                            double y2 = circles.get(0, p2.getRight())[1];
                            return Double.compare(y1, y2);
                        }
                    }
                });
                detectAnswers(subMat, circles, circlesOrdered, 1);
                break;
            default:
                System.out.println("Número de círculos no manejado en " + rectIndex + ": " + totalCircles);
                break;
        }
    }

    public List<Character> detectAnswersQuestions(Mat subMat, Mat circles, List<Pair<Double, Integer>> circlesOrdered) {
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
            }

            letterIndex++;
            if (letterIndex >= 4) {
                letterIndex = 0;
            }
        }

        if (answeredQuestion.isEmpty()) {
            answersList.add('N');
        } else if (answeredQuestion.size() == 1) {
            answersList.add(answeredQuestion.get(0));
        } else if (answeredQuestion.size() == 2) {
            answersList.add('N');
        } else if (answeredQuestion.size() > 2) {
            answersList.add('N');
        }

        return answeredQuestion;

    }

    public void detectAnswersToAlphabet(Mat subMat, Mat circles, List<Pair<Double, Integer>> circlesOrdered, int flag) {
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
                if (flag == 0) {
                    DetectExam.setNumberToNieLetter(numberToChar);
                } else {
                    DetectExam.setNumberToDniLetter(numberToChar);

                }
                Imgproc.circle(subMat, center, radius, new Scalar(0, 255, 0), 3, 8, 0);
                Imgproc.putText(subMat, String.valueOf(counter + 1), new Point(center.x - radius, center.y), Core.Formatter_FMT_DEFAULT, 0.5, new Scalar(0, 0, 255), 2);
            } else {
                Imgproc.circle(subMat, center, radius, new Scalar(255, 0, 255), 3, 8, 0);
            }
            counter++;
        }
    }

    public void detectAnswers(Mat subMat, Mat circles, List<Pair<Double, Integer>> circlesOrdered, int flag) {

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
                    if (flag == 0) {
                        DetectExam.setNumbersOfExam(foundNumbers);
                    } else {
                        DetectExam.setNumbersOfIdentification(foundNumbers);
                    }

                    Imgproc.circle(subMat, center, radius, new Scalar(0, 255, 0), 3, 8, 0);
                    Imgproc.putText(subMat, String.valueOf(counter), new Point(center.x - radius, center.y), Core.Formatter_FMT_DEFAULT, 0.5, new Scalar(0, 0, 255), 2);
                } else {
                    Imgproc.circle(subMat, center, radius, new Scalar(255, 0, 255), 3, 8, 0);
                }
                counter++;
            }
        }
    }

    public void drawCircles(Mat subMat, Mat circles, List<Character> answeredQuestion) {
        char[] letterAnswers = {'A', 'B', 'C', 'D'};
        int letterIndex = 0;

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

            char letterAnswer = letterAnswers[letterIndex];
            Imgproc.putText(subMat, String.valueOf(letterAnswer), new Point(center.x - radius, center.y), Core.Formatter_FMT_DEFAULT, 1, new Scalar(0, 0, 255), 2);

            if (answeredQuestion.contains(letterAnswer)) {
                Imgproc.circle(subMat, center, radius, new Scalar(0, 255, 0), 3, 8, 0);
            } else {
                Imgproc.circle(subMat, center, radius, new Scalar(255, 0, 255), 3, 8, 0);
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
