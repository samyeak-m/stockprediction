package util;

import java.util.Arrays;

public class DataPreprocessor {

    public static double[][] normalize(double[][] data) {
        // Calculate min and max values across columns
        double[] min = new double[data[0].length];
        double[] max = new double[data[0].length];

        for (int i = 0; i < data[0].length; i++) {
            min[i] = Double.MAX_VALUE;
            max[i] = Double.MIN_VALUE;
        }

        // Find min and max values
        for (double[] row : data) {
            for (int i = 0; i < row.length; i++) {
                if (row[i] < min[i]) {
                    min[i] = row[i];
                }
                if (row[i] > max[i]) {
                    max[i] = row[i];
                }
            }
        }

        // Normalize data
        double[][] normalizedData = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                normalizedData[i][j] = (max[j] - min[j]) != 0 ? (data[i][j] - min[j]) / (max[j] - min[j]) : 0;
            }
        }

        return normalizedData;
    }

    public static double[][][] preprocessData(double[][] data, double trainingSplit) {
        double[][] normalizedData = normalize(data);

        int trainSize = (int) (normalizedData.length * trainingSplit);
        double[][] trainData = new double[trainSize][normalizedData[0].length];
        double[][] testData = new double[normalizedData.length - trainSize][normalizedData[0].length];

        System.arraycopy(normalizedData, 0, trainData, 0, trainSize);
        System.arraycopy(normalizedData, trainSize, testData, 0, normalizedData.length - trainSize);

        return new double[][][]{trainData, testData};
    }

    public static double[] removeNaNs(double[] data) {
        return Arrays.stream(data).filter(Double::isFinite).toArray();
    }

    public static void checkDataForNaNs(double[] data) {
        if (Arrays.stream(data).anyMatch(Double::isNaN)) {
            throw new IllegalArgumentException("Data contains NaN values!");
        }
    }

    public static double[][] addFeatures(double[][] stockData, double[][] technicalIndicators) {
        int numRows = stockData.length;
        int numCols = stockData[0].length + technicalIndicators[0].length;
        double[][] extendedData = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            System.arraycopy(stockData[i], 0, extendedData[i], 0, stockData[i].length);
            System.arraycopy(technicalIndicators[i], 0, extendedData[i], stockData[i].length, technicalIndicators[i].length);
        }

        return extendedData;
    }
}
