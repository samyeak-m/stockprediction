import database.DatabaseHelper;
import util.DataPreprocessor;
import lstm.LSTMNetwork;

import java.io.File;
import java.sql.SQLException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    private static final String MODEL_FILE_PATH = "lstm_model.ser".replace("/", File.separator);
    private static final Logger LOGGER = Logger.getLogger(Main.class.getName());
    static String RESET = "\u001B[0m";
    static String GREEN = "\u001B[32m";
    static String BLUE = "\u001B[34m";
    static String YELLOW = "\u001B[33m";

    public static void main(String[] args) throws SQLException {
        LOGGER.setLevel(Level.INFO);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.INFO);
        LOGGER.addHandler(handler);

        DatabaseHelper dbHelper = new DatabaseHelper();

        LSTMNetwork lstm = LSTMNetwork.loadModel(MODEL_FILE_PATH);
        if (lstm == null) {
            lstm = new LSTMNetwork(6, 365, 1);  // Assuming 7 features: date, open, high, low, close, volume, turnover
        }

        List<String> tableNames = dbHelper.getAllStockTableNames();
        List<double[]> allStockData = new ArrayList<>();

        for (String tableName : tableNames) {
            allStockData.addAll(dbHelper.loadStockData(tableName));
        }

        double[][] stockDataArray = allStockData.toArray(new double[0][]);

        double[][][] preprocessedData = DataPreprocessor.preprocessData(stockDataArray, 0.6);
        double[][] trainData = preprocessedData[0];
        double[][] testData = preprocessedData[1];

        System.out.println(BLUE+"Training data size: " + trainData.length+RESET);
        System.out.println(BLUE+"Test data size: " + testData.length+RESET);

        trainModel(lstm, trainData, 1000, 0.001);

        double accuracy = testModel(lstm, testData);
        while (accuracy < 0.90) {
            System.out.println(GREEN+"Accuracy below 90%, retraining model..."+RESET);
            lstm = new LSTMNetwork(7, 365, 1);
            trainModel(lstm, trainData, 1000, 0.01);
            accuracy = testModel(lstm, testData);
        }

        lstm.saveModel(MODEL_FILE_PATH);

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("Enter the stock symbol to predict: ");
                String stockSymbol = scanner.nextLine();
                System.out.print("Enter the number of days for prediction: ");
                int days = scanner.nextInt();
                scanner.nextLine();

                predictAndSave(dbHelper, lstm, stockSymbol, days);

                System.out.print(BLUE+"Do you want to predict for another stock? (yes/no): "+RESET);
                String response = scanner.nextLine();
                if (!response.equalsIgnoreCase("yes")) {
                    break;
                }
            }
        }
    }

    private static void trainModel(LSTMNetwork lstm, double[][] trainData, int epochs, double learningRate) {
        try {
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (double[] point : trainData) {
                    double[] input = new double[point.length - 1];
                    System.arraycopy(point, 1, input, 0, point.length - 1);
                    double[] target = new double[]{point[1]};

                    System.out.println(YELLOW+"Input: " + Arrays.toString(input) + ", Target: "+BLUE + Arrays.toString(target)+RESET);

                    lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
                    lstm.backpropagate(input, target, learningRate);

                    System.out.println(YELLOW+"Forward and Backpropagation completed for input: "+BLUE + Arrays.toString(input)+RESET);
                }
                System.out.println(GREEN+"Epoch " + epoch + " completed."+RESET);
            }
            System.out.println(GREEN+"Training completed successfully."+RESET);
        } catch (Exception e) {
            System.err.println("Exception during training: " + e.getMessage());
            e.printStackTrace();
        }
    }



    private static double testModel(LSTMNetwork lstm, double[][] testData) {
        int correctPredictions = 0;
        for (double[] point : testData) {
            double[] input = new double[point.length - 1];
            System.arraycopy(point, 1, input, 0, point.length - 1);
            double[] output = lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
            double prediction = output[0];
            double actual = point[4];

            if (Math.abs(prediction - actual) / actual < 0.1) {  // Considering within 10% deviation as correct
                correctPredictions++;
            }
        }
        double accuracy = (double) correctPredictions / testData.length;
        System.out.println(YELLOW+"Model accuracy: " + (accuracy * 100) + "%"+RESET);
        return accuracy;
    }

    private static void predictAndSave(DatabaseHelper dbHelper, LSTMNetwork lstm, String stockSymbol, int days) throws SQLException {
        List<double[]> stockData = dbHelper.loadStockData("daily_data_" + stockSymbol);

        double[] hiddenState = new double[lstm.getHiddenSize()];
        double[] cellState = new double[lstm.getHiddenSize()];

        for (int i = 0; i < days; i++) {
            double[] input = new double[stockData.get(stockData.size() - 1).length - 1];
            System.arraycopy(stockData.get(stockData.size() - 1), 1, input, 0, input.length);
            double[] output = lstm.forward(input, hiddenState, cellState);

            double prediction = output[0];
            stockData.add(new double[]{stockData.get(stockData.size() - 1)[0] + 86400000, prediction});

            LocalDate today = LocalDate.now();
            LocalDate predictionDate = today.plusDays(i + 1);
            String predict = (prediction - input[4]) / input[4] > 0.1 ? "up" : (prediction - input[4]) / input[4] < -0.1 ? "down" : "neutral";
            double priceChange = prediction - input[4];
            String pointChangeStr = String.format("%.2f", (priceChange / input[4]) * 100);

            if (Double.parseDouble(pointChangeStr) > 10) {
                pointChangeStr = "10";
                prediction = input[4] * 1.10;
            } else if (Double.parseDouble(pointChangeStr) < -10) {
                pointChangeStr = "-10";
                prediction = input[4] * 0.90;
            }

            dbHelper.savePrediction(stockSymbol, predict, pointChangeStr, priceChange, prediction, input[4], today, predictionDate);

            System.out.println(BLUE+"Predicted next closing price for " + stockSymbol + " on " + predictionDate + ": " + prediction);
            System.out.println(GREEN+"Prediction saved successfully."+RESET);
        }
    }
}
