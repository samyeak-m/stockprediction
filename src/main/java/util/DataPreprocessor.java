package util;

import java.util.Arrays;

public class DataPreprocessor {

    public static double[][] normalize(double[][] data) {
        double[] min = new double[data[0].length];
        double[] max = new double[data[0].length];
        Arrays.fill(min, Double.MAX_VALUE);
        Arrays.fill(max, Double.MIN_VALUE);

        // Find min and max values
        for (double[] row : data) {
            for (int i = 0; i < row.length; i++) {
                if (row[i] < min[i]) min[i] = row[i];
                if (row[i] > max[i]) max[i] = row[i];
            }
        }

        // Normalize data, avoiding division by zero
        double[][] normalizedData = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                if (max[j] != min[j]) {
                    normalizedData[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
                } else {
                    normalizedData[i][j] = 0; // Avoid division by zero if max equals min
                }
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

//        print2DArray(extendedData);

        return extendedData;
    }

    public static void print2DArray(double[][] array) {
        for (double[] row : array) {
            System.out.println(Arrays.toString(row));
        }
    }

}