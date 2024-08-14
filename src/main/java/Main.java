import util.CustomChartUtils;
import util.DataPreprocessor;
import util.TechnicalIndicators;
import lstm.LSTMNetwork;
import lstm.LSTMTrainer;
import database.DatabaseHelper;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.SQLException;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    static String version = "v1";

    private static final Logger LOGGER = Logger.getLogger(Main.class.getName());
    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String BLUE = "\u001B[34m";
    private static final String YELLOW = "\u001B[33m";

    static int hiddenSize = 75;
    static int inputSize = 8;
    static int outputSize = 1;
    static int epoch = 30;
    static int batch = 16;
    static double trainingRate = 0.001;

    private static final String MODEL_FILE_PATH = "lstm_model" + version +"_"+epoch+".ser";


    private static final List<Integer> epochList = new ArrayList<>();
    private static final List<Double> accuracyList = new ArrayList<>();
    private static final List<Double> lossList = new ArrayList<>();

    private static double[] min;
    private static double[] max;

    private static BufferedWriter fileWriter;

    static double threshold = 0.5;

    public static void main(String[] args) throws SQLException, IOException, ClassNotFoundException {
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.INFO);

        FileWriter writer = new FileWriter("normal1.txt", true); // Append mode
        fileWriter = new BufferedWriter(writer);

        DatabaseHelper dbHelper = new DatabaseHelper();

        LSTMNetwork lstm = LSTMNetwork.loadModel(MODEL_FILE_PATH);
        if (lstm == null) {
            lstm = new LSTMNetwork(inputSize, hiddenSize, outputSize);
        }

        List<String> tableNames = dbHelper.getAllStockTableNames();
        List<double[]> allStockData = new ArrayList<>();

        for (String tableName : tableNames) {
            allStockData.addAll(dbHelper.loadStockData(tableName));
        }

        double[][] stockDataArray = allStockData.toArray(new double[0][]);

        double[][] technicalIndicators = TechnicalIndicators.calculate(stockDataArray, 16, 3);

        double[][] extendedData = DataPreprocessor.addFeatures(stockDataArray, technicalIndicators);

        double[][][] preprocessedData = preprocessData(extendedData, 0.6);
        double[][] trainData = preprocessedData[0];
        double[][] testData = preprocessedData[1];

        double[][] validationData = Arrays.copyOfRange(testData, 0, testData.length / 5);
        double[][] finalTestData = Arrays.copyOfRange(testData, testData.length / 5, testData.length);


        LOGGER.log(Level.INFO, BLUE + "Training data size: " + trainData.length + RESET);
        LOGGER.log(Level.INFO, BLUE + "Validation data size: " + validationData.length + RESET);
        LOGGER.log(Level.INFO, BLUE + "Final test data size: " + finalTestData.length + RESET);

        trainModel(lstm, trainData, validationData, epoch, trainingRate);

        double testAccuracy = testModel(lstm, finalTestData) * 100;
        LOGGER.log(Level.INFO, String.format(GREEN + "Final Test Accuracy: %.2f%%" + RESET, testAccuracy));

        double finalTestLoss = calculateLoss(lstm, finalTestData) * 100;
        LOGGER.log(Level.INFO, String.format(GREEN + "Final Test Loss: %.2f%%" + RESET, finalTestLoss));

        double[] targets = new double[finalTestData.length];
        for (int i = 0; i < targets.length; i++) {
            targets[i] = finalTestData[i][1]; // Use finalTestData[i][1] for targets
        }
        int[][] confusionMatrix = lstm.computeConfusionMatrix(finalTestData, finalTestData[finalTestData.length - 1][1], threshold);
        printConfusionMatrix(confusionMatrix);

        lstm.saveModel(MODEL_FILE_PATH);


        String accuracyChartDir = "charts" + version + "_" + epoch + File.separator + "accuracy";
        String predictionChartDir = "charts" + version + "_" + epoch + File.separator + "predictions";
        createDirectory(accuracyChartDir);
        createDirectory(predictionChartDir);

        CustomChartUtils.saveAccuracyChart("Model Accuracy", epochList, accuracyList, validationAccuracyList, accuracyChartDir + File.separator + "model_accuracy.png", "Epochs", "Accuracy");
        CustomChartUtils.saveLossChart("Model Loss", epochList, lossList, validationLossList, accuracyChartDir + File.separator + "model_loss.png", "Epochs", "Loss");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("Enter the stock symbol to predict: ");
                String stockSymbol = scanner.nextLine();

                predictAndSave(dbHelper, lstm, stockSymbol, predictionChartDir);

                System.out.print(BLUE + "Do you want to predict for another stock? (yes/no): " + RESET);
                String response = scanner.nextLine();
                if (!response.equalsIgnoreCase("yes")) {
                    break;
                }
            }
        }

        System.out.println(GREEN + "Program execution finished." + RESET);

        fileWriter.close();
    }

    private static void printConfusionMatrix(int[][] matrix) {
        System.out.println("Confusion Matrix:");
        System.out.println("TP: " + matrix[0][0] + ", FN: " + matrix[1][0]);
        System.out.println("FP: " + matrix[0][1] + ", TN: " + matrix[1][1]);
    }

    private static final List<Double> validationAccuracyList = new ArrayList<>();
    private static final List<Double> validationLossList = new ArrayList<>();

    private static void trainModel(LSTMNetwork lstm, double[][] trainData, double[][] validationData, int epochs, double learningRate) {
        LSTMTrainer trainer = new LSTMTrainer(lstm, learningRate);
        double prevAccuracy = 0;
        int sameCount = 0;

        double totalAccuracy = 0;
        double totalLoss = 0;
        int epochCount = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            long startTime = System.currentTimeMillis();
            int totalDataPoints = trainData.length;
            int batchSize = batch;
            int batches = totalDataPoints / batchSize;

            double totalEpochLoss = 0;

            for (int batch = 0; batch < batches; batch++) {
                double[][] batchData = Arrays.copyOfRange(trainData, batch * batchSize, (batch + 1) * batchSize);
                for (double[] data : batchData) {
                    double[] input = Arrays.copyOfRange(data, 0, data.length);
                    double[] target = new double[]{data[data.length - 1]};
                    lstm.backpropagate(input, target, learningRate);

                    double[] output = lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());

                    if (output == null) {
                        LOGGER.severe("NaN value encountered during forward pass. Stopping training.");
                        return;
                    }
                }
            }

            double accuracy = testModel(lstm, trainData);
            double epochLoss = calculateLoss(lstm, trainData);

            double validationAccuracy = testModel(lstm, validationData);
            double validationLoss = calculateValidationLoss(lstm, validationData);

            long endTime = System.currentTimeMillis();
            long elapsedTimeMillis = endTime - startTime;
            String elapsedTime = String.format("%02d:%02d:%02d",
                    (elapsedTimeMillis / (1000 * 60 * 60)) % 24,
                    (elapsedTimeMillis / (1000 * 60)) % 60,
                    (elapsedTimeMillis / 1000) % 60);

            epochList.add(epoch);
            accuracyList.add(accuracy);
            lossList.add(epochLoss);
            validationAccuracyList.add(validationAccuracy);
            validationLossList.add(validationLoss);

            LOGGER.log(Level.INFO, String.format(YELLOW + "Epoch %d: Accuracy = %.2f, Loss = %.2f, Validation Accuracy = %.2f, Validation Loss = %.2f, Time = %s" + RESET,
                    epoch, accuracy, epochLoss, validationAccuracy, validationLoss, elapsedTime));

            totalAccuracy += accuracy;
            totalLoss += epochLoss;
            epochCount++;

            prevAccuracy = validationAccuracy;
        }

        double averageAccuracy = totalAccuracy / epochCount;
        double averageLoss = totalLoss / epochCount;

        LOGGER.log(Level.INFO, String.format(GREEN + "Overall Average Accuracy: %.2f" + RESET, averageAccuracy));
        LOGGER.log(Level.INFO, String.format(GREEN + "Overall Average Loss: %.2f" + RESET, averageLoss));
    }


    private static double calculateValidationLoss(LSTMNetwork lstm, double[][] validationData) {
        return calculateLoss(lstm, validationData);
    }


    private static double testModel(LSTMNetwork lstm, double[][] testData) {
        double totalAccuracy = 0;

        for (int i = 0; i < testData.length - 1; i++) {
            double[] input = Arrays.copyOf(testData[i], testData[i].length - 1);
            double[] output = lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
            if (output == null) {
                continue;
            }
            double prediction = output[0];
            double actual = testData[i + 1][1];
            double currentClosePrice = testData[i][1];


            double lastClosePrice = testData[i][1];
            prediction = applyPredictionConstraints(prediction, lastClosePrice);

            // Calculate the prediction accuracy based on proximity to the actual value
            double accuracy = calculatePredictionAccuracy(prediction, actual, currentClosePrice);
            totalAccuracy += accuracy;
        }
        return totalAccuracy / (testData.length - 1);
    }

    private static double applyPredictionConstraints(double prediction, double lastClosePrice) {
        double maxChange = 0.10 * lastClosePrice;
        double minPrice = lastClosePrice * 0.90;
        double maxPrice = lastClosePrice * 1.10;

        if (prediction < minPrice) {
            prediction = minPrice;
        } else if (prediction > maxPrice) {
            prediction = maxPrice;
        }

        prediction = Math.max(0, Math.min(1, prediction));

        return prediction;
    }


    private static double calculatePredictionAccuracy(double prediction, double actual, double currentClosePrice) {
        double maxChange = 0.10 * currentClosePrice;
        double diff = Math.abs(prediction - actual);;

        if (diff > maxChange) {
            return 0;
        }

        double accuracy = 1 - (diff / maxChange);
        return accuracy;
    }



    private static double calculateLoss(LSTMNetwork lstm, double[][] data) {
        double totalLoss = 0;
        double maxChange = 0.10; // 10% tolerance

        for (int i = 0; i < data.length - 1; i++) {
            double[] input = Arrays.copyOf(data[i], data[i].length - 1);
            double lastClosePrice = data[i][1];
            double[] output = lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
            double prediction = output[0];

            double minPrice = lastClosePrice * (1 - maxChange);
            double maxPrice = lastClosePrice * (1 + maxChange);

            if (prediction < minPrice) {
                prediction = minPrice;
            } else if (prediction > maxPrice) {
                prediction = maxPrice;
            }

            double actual = data[i + 1][1];
            double diff = Math.abs(prediction - actual);
            double tolerance = maxChange * actual;

            double loss;
            if (diff > tolerance) {
                loss = 1.0;
            } else {
                loss = diff / tolerance;
            }

            totalLoss += loss;
        }

        return totalLoss / (data.length - 1);
    }

    private static void createDirectory(String directory) {
        File dir = new File(directory);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }

    private static void predictAndSave(DatabaseHelper dbHelper, LSTMNetwork lstm, String stockSymbol, String predictionChartDir) throws SQLException, IOException {
        List<double[]> stockData = dbHelper.loadStockData(stockSymbol);
        double[][] stockDataArray = stockData.toArray(new double[0][]);

        double[][] technicalIndicators = TechnicalIndicators.calculate(stockDataArray,  16, 3);

        double[][] extendedData = DataPreprocessor.addFeatures(stockDataArray, technicalIndicators);

        extendedData = DataPreprocessor.normalize(extendedData, min, max);

        int days = 1;

        double[] predictions = new double[days];
        for (int i = 0; i < days; i++) {
            double[] input = Arrays.copyOfRange(extendedData[extendedData.length - 1], 0, extendedData[0].length);
            double[] output = lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
            double prediction = output[0];

            double lastClosePrice = extendedData[extendedData.length - 1][1];

            prediction = applyPredictionConstraints(prediction, lastClosePrice);

            predictions[i] = prediction;

            double[] newInput = new double[extendedData[0].length];
            System.arraycopy(input, 1, newInput, 0, input.length - 1);
            newInput[newInput.length - 1] = predictions[i];
            extendedData = Arrays.copyOf(extendedData, extendedData.length + 1);
            extendedData[extendedData.length - 1] = newInput;
        }

        predictions = DataPreprocessor.denormalize(predictions, min[min.length - 1], max[max.length - 1]);

        double[] actualPrices = new double[days];
        for (int i = 0; i < days; i++) {
            actualPrices[i] = stockDataArray[stockDataArray.length - 1][1];
        }

        // Save predictions to chart
        dbHelper.savePredictions(stockSymbol, predictions, actualPrices);
        System.out.println(GREEN + "Predictions saved for stock: " + stockSymbol + RESET);
    }

    private static double[][][] preprocessData(double[][] data, double trainSplitRatio) {
        int trainSize = (int) (data.length * trainSplitRatio);
        double[][] trainData = Arrays.copyOfRange(data, 0, trainSize);
        double[][] testData = Arrays.copyOfRange(data, trainSize, data.length);

        min = DataPreprocessor.calculateMin(data);
        max = DataPreprocessor.calculateMax(data);

        double bufferPercentage = 0.40;

        for (int i = 1; i < min.length; i++) {
            double actualMin = min[i];
            double actualMax = max[i];

            double bufferedMin = actualMin - (bufferPercentage * (actualMax - actualMin));
            double bufferedMax = actualMax + (bufferPercentage * (actualMax - actualMin));

            if (bufferedMin < 0) {
                bufferedMin = 0;
            }

            min[i] = bufferedMin;
            max[i] = bufferedMax;
        }


        trainData = DataPreprocessor.normalize(trainData, min, max);
        testData = DataPreprocessor.normalize(testData, min, max);

        return new double[][][]{trainData, testData};
    }


}
