/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */
package com.cristian.detectexam;

import com.opencsv.CSVWriterBuilder;
import com.opencsv.ICSVWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.Rect;


public class DetectExam {

    private static int wrongAnswerCounter;
    private static int correctAnswerCounter;
    private static int nullAnswerCounter;
    private static int numberToDniLetter;
    private static int numberToNieLetter;
    private static String examFile;
    private static String idFile;
    public static String idStudentDni;
    public static String idStudentNie;
    public static String idStudent;
    public static String idExam;
    public static double score;
    private static List<Integer> numbersOfIdentification = new ArrayList<>();
    private static List<Integer> numbersOfExam = new ArrayList<>();
    private static List<Character> detectedAnswers = new ArrayList<>();


    public static void main(String[] args) throws IOException {
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
        start();
    }

    private static void start() throws IOException {
//        String examFile = "I:\\FPDAM\\PRACTICAS_EMPRESA\\recursosOpencv\\ExamenPruebaNie.jpg";
        String urlCsv = "I:\\FPDAM\\PRACTICAS_EMPRESA\\recursosOpencv\\dataFromStudents2.csv";

        Rectangle rectangleProcessor = new Rectangle(examFile);

        if (!rectangleProcessor.loadImage()) {
            return;
        }

        rectangleProcessor.applyFiltersAndFindExternalRects();
        List<Rect> detectedCells = rectangleProcessor.getFinalListRects();
        Mat src = rectangleProcessor.getSourceImageMat();

        Circle circleProcessor = new Circle();
        circleProcessor.detectCircles(src, detectedCells);

        List<Character> correctAnswers = Arrays.asList('A', 'B', 'C', 'D', 'N', 'A', 'N', 'D', 'C', 'A', 'A', 'B', 'C', 'D','C', 'B', 'A', 'N', 'C', 'D', 'C', 'B', 'A', 'B', 'C', 'D', 'C', 'B','A', 'B', 'N', 'C', 'D', 'B', 'B', 'C', 'D', 'A', 'C', 'B');
        setScore(calculateScore(detectedAnswers, correctAnswers));
        setIdStudentDni(listToString(numbersOfIdentification) + String.valueOf(numberToChar(numberToDniLetter)));
        setIdStudentNie(String.valueOf(numberToChar(numberToNieLetter)) + listToString(numbersOfIdentification));
        setIdExam(listToString(numbersOfExam));
        writeOpenCsv(urlCsv);
    }

    private static double calculateScore(List<Character> detectedAnswers, List<Character> correctAnswers) {
        for (int i = 0; i < detectedAnswers.size(); i++) {
            char detected = detectedAnswers.get(i);
            char correct = correctAnswers.get(i);

            if (detected == 'N') {
                nullAnswerCounter++;
                continue;
            }

            if (detected == correct) {
                score += 0.25;
                correctAnswerCounter++;
            } else {
                score -= 0.08;
                wrongAnswerCounter++;
            }
        }

        return score;
    }

    public static char numberToChar(int number) {
        char[] letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".toCharArray();

        if (number == 0) {
            return '0';
        } else if (number < 1 || number > 26) {
            throw new IllegalArgumentException("El número debe estar entre 1 y 26");
        } else {
            return letters[number - 1];
        }
    }

    private static int listToInt(List<Integer> list) {
        StringBuilder builder = new StringBuilder();
        for (Integer num : list) {
            builder.append(num);
        }
        return Integer.parseInt(builder.toString());
    }

    private static String listToString(List<Integer> list) {
        StringBuilder builder = new StringBuilder();
        for (Integer num : list) {
            builder.append(num);
        }
        return builder.toString();
    }

    private static String listToStringCharacters(List<Character> list) {
        StringBuilder builder = new StringBuilder();
        for (Character cha : list) {
            if (builder.length() > 0) {
                builder.append(',');
            }
            builder.append(cha);
        }
        return builder.toString();
    }

    public static void writeOpenCsv(String url) throws IOException {
        try (ICSVWriter writer = new CSVWriterBuilder(new FileWriter(url, true))
                .withSeparator(';')
                .withQuoteChar(ICSVWriter.NO_QUOTE_CHARACTER)
                .build()) {

            if (numberToNieLetter == 0 && numberToDniLetter == 0) {
                System.out.println("No se sabe el tipo de ID");
            } else if (numberToNieLetter > 0 && numberToDniLetter > 0) {
                System.out.println("Documento incorrecto");
            } else if (numberToNieLetter > 0) {
                String[] data = new String[]{
                    String.valueOf(numberToChar(numberToNieLetter)) + listToString(numbersOfIdentification),
                    listToString(numbersOfExam),
                    listToStringCharacters(detectedAnswers),
                    String.valueOf(score)
                };
                writer.writeNext(data);
                setIdStudent(String.valueOf(numberToChar(numberToNieLetter)) + listToString(numbersOfIdentification));
            } else {
                String[] data = new String[]{
                    listToString(numbersOfIdentification) + String.valueOf(numberToChar(numberToDniLetter)),
                    listToString(numbersOfExam),
                    listToStringCharacters(detectedAnswers),
                    String.valueOf(score)
                };
                writer.writeNext(data);
                setIdStudent(listToString(numbersOfIdentification) + String.valueOf(numberToChar(numberToDniLetter)));
            }
        }

    }

    public static int getWrongAnswerCounter() {
        return wrongAnswerCounter;
    }

    public static void setWrongAnswerCounter(int wrongAnswerCounter) {
        DetectExam.wrongAnswerCounter = wrongAnswerCounter;
    }

    public static int getCorrectAnswerCounter() {
        return correctAnswerCounter;
    }

    public static void setCorrectAnswerCounter(int correctAnswerCounter) {
        DetectExam.correctAnswerCounter = correctAnswerCounter;
    }

    public static int getNullAnswerCounter() {
        return nullAnswerCounter;
    }

    public static void setNullAnswerCounter(int nullAnswerCounter) {
        DetectExam.nullAnswerCounter = nullAnswerCounter;
    }

    public static int getNumberToDniLetter() {
        return numberToDniLetter;
    }

    public static void setNumberToDniLetter(int numberToDniLetter) {
        DetectExam.numberToDniLetter = numberToDniLetter;
    }

    public static int getNumberToNieLetter() {
        return numberToNieLetter;
    }

    public static void setNumberToNieLetter(int numberToNieLetter) {
        DetectExam.numberToNieLetter = numberToNieLetter;
    }

    public static String getExamFile() {
        return examFile;
    }

    public static void setExamFile(String examFile) {
        DetectExam.examFile = examFile;
    }

    public static String getIdFile() {
        return idFile;
    }

    public static void setIdFile(String idFile) {
        DetectExam.idFile = idFile;
    }

    public static String getIdStudentDni() {
        return idStudentDni;
    }

    public static void setIdStudentDni(String idStudentDni) {
        DetectExam.idStudentDni = idStudentDni;
    }

    public static String getIdStudentNie() {
        return idStudentNie;
    }

    public static void setIdStudentNie(String idStudentNie) {
        DetectExam.idStudentNie = idStudentNie;
    }

    public static String getIdStudent() {
        return idStudent;
    }

    public static void setIdStudent(String idStudent) {
        DetectExam.idStudent = idStudent;
    }

    public static String getIdExam() {
        return idExam;
    }

    public static void setIdExam(String idExam) {
        DetectExam.idExam = idExam;
    }

    public static double getScore() {
        return score;
    }

    public static void setScore(double score) {
        DetectExam.score = score;
    }

    public static List<Integer> getNumbersOfIdentification() {
        return numbersOfIdentification;
    }

    public static void setNumbersOfIdentification(List<Integer> numbersOfIdentification) {
        DetectExam.numbersOfIdentification = numbersOfIdentification;
    }

    public static List<Integer> getNumbersOfExam() {
        return numbersOfExam;
    }

    public static void setNumbersOfExam(List<Integer> numbersOfExam) {
        DetectExam.numbersOfExam = numbersOfExam;
    }

    public static List<Character> getDetectedAnswers() {
        return detectedAnswers;
    }

    public static void setDetectedAnswers(List<Character> detectedAnswers) {
        DetectExam.detectedAnswers = detectedAnswers;
    }
}
