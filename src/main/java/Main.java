import database.DatabaseHelper;
import util.ChartUtils;
import util.DataPreprocessor;
import lstm.LSTMNetwork;

import java.time.Duration;
import java.time.Instant;
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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Main {
    private static final String MODEL_FILE_PATH = "lstm_model.ser".replace("/", File.separator);
    private static final Logger LOGGER = Logger.getLogger(Main.class.getName());
    static String RESET = "\u001B[0m";
    static String GREEN = "\u001B[32m";
    static String BLUE = "\u001B[34m";
    static String YELLOW = "\u001B[33m";

    private static List<Integer> epochList = new ArrayList<>();
    private static List<Double> accuracyList = new ArrayList<>();

    public static void main(String[] args) throws SQLException, IOException {
        LOGGER.setLevel(Level.INFO);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.INFO);
        LOGGER.addHandler(handler);

        DatabaseHelper dbHelper = new DatabaseHelper();

        LSTMNetwork lstm = LSTMNetwork.loadModel(MODEL_FILE_PATH);
        if (lstm == null) {
            lstm = new LSTMNetwork(6, 3650, 1);
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

        LOGGER.log(Level.INFO, BLUE + "Training data size: " + trainData.length + RESET);
        LOGGER.log(Level.INFO, BLUE + "Test data size: " + testData.length + RESET);

        trainModel(lstm, trainData, 10000, 0.001);

        double accuracy = testModel(lstm, testData);
        while (accuracy < 0.99) {
            LOGGER.log(Level.INFO, GREEN + "Accuracy below 90%, retraining model..." + RESET);
            lstm = new LSTMNetwork(7, 365, 1);
            trainModel(lstm, trainData, 1000, 0.01);
            accuracy = testModel(lstm, testData);
        }

        lstm.saveModel(MODEL_FILE_PATH);

        // Create directories for saving charts
        String accuracyChartDir = "charts/accuracy";
        String predictionChartDir = "charts/predictions";
        createDirectory(accuracyChartDir);
        createDirectory(predictionChartDir);

        // Save the accuracy chart
        ChartUtils.saveAccuracyChart("Model Accuracy", epochList, accuracyList, accuracyChartDir + "/model_accuracy.png");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("Enter the stock symbol to predict: ");
                String stockSymbol = scanner.nextLine();
                System.out.print("Enter the number of days for prediction: ");
                int days = scanner.nextInt();
                scanner.nextLine();

                predictAndSave(dbHelper, lstm, stockSymbol, days, predictionChartDir);

                System.out.print(BLUE + "Do you want to predict for another stock? (yes/no): " + RESET);
                String response = scanner.nextLine();
                if (!response.equalsIgnoreCase("yes")) {
                    break;
                }
            }
        }
    }

    private static void trainModel(LSTMNetwork lstm, double[][] trainData, int epochs, double learningRate) {
        Instant start = Instant.now();
        try {
            for (int epoch = 0; epoch < epochs; epoch++) {
                Instant epochStart = Instant.now();
                for (double[] point : trainData) {
                    double[] input = new double[point.length - 1];
                    System.arraycopy(point, 1, input, 0, point.length - 1);
                    double[] target = new double[]{point[1]};
                    LOGGER.log(Level.INFO, YELLOW + "Input: " + Arrays.toString(input) + ", Target: " + BLUE + Arrays.toString(target) + RESET);

                    lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
                    lstm.backpropagate(input, target, learningRate);
                    LOGGER.log(Level.INFO, YELLOW + "Forward and Backpropagation completed for input: " + BLUE + Arrays.toString(input) + RESET);

                }
                Instant epochEnd = Instant.now();
                Duration epochDuration = Duration.between(epochStart, epochEnd);
                Duration totalDuration = Duration.between(start, Instant.now());
                long remainingEpochs = epochs - (epoch + 1);
                Duration estimatedRemainingTime = epochDuration.multipliedBy(remainingEpochs);

                LOGGER.log(Level.INFO, GREEN + "Epoch " + epoch + " completed in " + formatDuration(epochDuration) + ". Estimated time left: " + formatDuration(estimatedRemainingTime) + RESET);

                // Test model after each epoch and record accuracy
                double accuracy = testModel(lstm, trainData);
                epochList.add(epoch + 1);
                accuracyList.add(accuracy);
            }
            LOGGER.log(Level.INFO, GREEN + "Training completed successfully." + RESET);
        } catch (Exception e) {
            System.err.println("Exception during training: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static double testModel(LSTMNetwork lstm, double[][] testData) {
        Instant start = Instant.now();
        int correctPredictions = 0;
        for (double[] point : testData) {
            double[] input = new double[point.length - 1];
            System.arraycopy(point, 1, input, 0, point.length - 1);
            double[] output = lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
            double prediction = output[0];
            double actual = point[4];

            if (Math.abs(prediction - actual) / actual < 0.1) {
                correctPredictions++;
            }
        }
        double accuracy = (double) correctPredictions / testData.length;
        Instant end = Instant.now();
        Duration duration = Duration.between(start, end);
        LOGGER.log(Level.INFO, YELLOW + "Model accuracy: " + (accuracy * 100) + "%, tested in " + formatDuration(duration) + RESET);
        return accuracy;
    }

    private static String formatDuration(Duration duration) {
        long seconds = duration.getSeconds();
        long absSeconds = Math.abs(seconds);
        String positive = String.format(
                "%d hours %02d minutes %02d seconds",
                absSeconds / 3600,
                (absSeconds % 3600) / 60,
                absSeconds % 60);
        return seconds < 0 ? "-" + positive : positive;
    }

    private static void predictAndSave(DatabaseHelper dbHelper, LSTMNetwork lstm, String stockSymbol, int days, String predictionChartDir) throws SQLException, IOException {
        List<double[]> stockData = dbHelper.loadStockData("daily_data_" + stockSymbol);

        double[] hiddenState = new double[lstm.getHiddenSize()];
        double[] cellState = new double[lstm.getHiddenSize()];

        List<LocalDate> dates = new ArrayList<>();
        List<Double> actualPrices = new ArrayList<>();
        List<Double> predictedPrices = new ArrayList<>();

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

            LOGGER.log(Level.INFO, "Predicted stock price for " + predictionDate + " is " + prediction + " (change: " + pointChangeStr + "%, " + predict + ")");

            dates.add(predictionDate);
            actualPrices.add(input[4]);
            predictedPrices.add(prediction);
        }

        // Create directory for specific stock if it does not exist
        String stockPredictionDir = predictionChartDir + "/" + stockSymbol;
        createDirectory(stockPredictionDir);

        // Save the prediction chart
        String predictionChartFilePath = stockPredictionDir + "/prediction_" + stockSymbol + ".png";
        ChartUtils.savePredictionChart("Stock Price Prediction - " + stockSymbol, dates, actualPrices, predictedPrices, predictionChartFilePath);
    }

    private static void createDirectory(String path) {
        try {
            Files.createDirectories(Paths.get(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
