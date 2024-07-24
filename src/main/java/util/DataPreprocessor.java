package util;

import java.util.Arrays;

public class DataPreprocessor {

    public static double[][] normalize(double[][] data, double[] min, double[] max) {
        double[][] normalizedData = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                normalizedData[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
            }
        }
        return normalizedData;
    }

    public static double[][] addFeatures(double[][] data, double[][] technicalIndicators) {
        double[][] extendedData = new double[data.length][data[0].length + technicalIndicators[0].length];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, extendedData[i], 0, data[i].length);
            System.arraycopy(technicalIndicators[i], 0, extendedData[i], data[i].length, technicalIndicators[i].length);
        }
        return extendedData;
    }

    public static double[] calculateMin(double[][] data) {
        double[] min = Arrays.copyOf(data[0], data[0].length);
        for (double[] row : data) {
            for (int i = 0; i < row.length; i++) {
                if (row[i] < min[i]) {
                    min[i] = row[i];
                }
            }
        }
        return min;
    }

    public static double[] calculateMax(double[][] data) {
        double[] max = Arrays.copyOf(data[0], data[0].length);
        for (double[] row : data) {
            for (int i = 0; i < row.length; i++) {
                if (row[i] > max[i]) {
                    max[i] = row[i];
                }
            }
        }
        return max;
    }

    public static double[] denormalize(double[] normalizedValues, double min, double max) {
        double[] denormalizedValues = new double[normalizedValues.length];
        for (int i = 0; i < normalizedValues.length; i++) {
            denormalizedValues[i] = normalizedValues[i] * (max - min) + min;
        }
        return denormalizedValues;
    }
}
