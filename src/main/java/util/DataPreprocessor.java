package util;

public class DataPreprocessor {

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

    public static double[][][] preprocessData(double[][] data, double trainSplitRatio) {
        int trainSize = (int) (data.length * trainSplitRatio);
        double[][] trainData = new double[trainSize][];
        double[][] testData = new double[data.length - trainSize][];

        System.arraycopy(data, 0, trainData, 0, trainSize);
        System.arraycopy(data, trainSize, testData, 0, data.length - trainSize);

        return new double[][][]{trainData, testData};
    }
}
