package util;

import org.jfree.chart.*;
import org.jfree.chart.plot.*;
import org.jfree.chart.renderer.xy.*;
import org.jfree.chart.axis.*;
import org.jfree.data.xy.*;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.data.category.DefaultCategoryDataset;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class CustomChartUtils {


    public static void saveAccuracyChart(String title, List<Integer> epochs, List<Double> accuracy, List<Double> validationAccuracy, String filePath, String xAxisLabel, String yAxisLabel,int interval) {
        int tickInterval = interval;

        XYSeries accuracySeries = new XYSeries("Training Accuracy");
        XYSeries validationAccuracySeries = new XYSeries("Validation Accuracy");

        for (int i = 0; i < epochs.size(); i += tickInterval) {
            int end = Math.min(i + tickInterval, epochs.size());
            double maxAccuracy = findMaxInRange(accuracy, i, end);
            double maxValidationAccuracy = findMaxInRange(validationAccuracy, i, end);
            accuracySeries.add((double)epochs.get(end - 1), maxAccuracy);
            validationAccuracySeries.add((double)epochs.get(end - 1), maxValidationAccuracy);
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(accuracySeries);
        dataset.addSeries(validationAccuracySeries);

        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, dataset, PlotOrientation.VERTICAL, true, true, false);

        XYPlot plot = chart.getXYPlot();
        NumberAxis xAxis = new NumberAxis(xAxisLabel);
        xAxis.setTickUnit(new org.jfree.chart.axis.NumberTickUnit(tickInterval));  // Use the variable for tick interval
        plot.setDomainAxis(xAxis);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesLinesVisible(0, true);
        renderer.setSeriesShapesVisible(0, false);
        renderer.setSeriesLinesVisible(1, true);
        renderer.setSeriesShapesVisible(1, false);
        renderer.setSeriesPaint(0, Color.GREEN);
        renderer.setSeriesPaint(1, Color.RED);
        plot.setRenderer(renderer);

        saveChart(chart, filePath);
    }

    public static void saveLossChart(String title, List<Integer> epochs, List<Double> loss, List<Double> validationLoss, String filePath, String xAxisLabel, String yAxisLabel,int interval) {
        int tickInterval = interval;

        XYSeries lossSeries = new XYSeries("Training Loss");
        XYSeries validationLossSeries = new XYSeries("Validation Loss");

        for (int i = 0; i < epochs.size(); i += tickInterval) {
            int end = Math.min(i + tickInterval, epochs.size());
            double maxLoss = findMaxInRange(loss, i, end);
            double maxValidationLoss = findMaxInRange(validationLoss, i, end);
            lossSeries.add((double)epochs.get(end - 1), maxLoss);
            validationLossSeries.add((double)epochs.get(end - 1), maxValidationLoss);
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(lossSeries);
        dataset.addSeries(validationLossSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, dataset, PlotOrientation.VERTICAL, true, true, false);

        XYPlot plot = chart.getXYPlot();
        NumberAxis xAxis = new NumberAxis(xAxisLabel);
        xAxis.setTickUnit(new org.jfree.chart.axis.NumberTickUnit(tickInterval));  // Use the variable for tick interval
        plot.setDomainAxis(xAxis);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesLinesVisible(0, true);
        renderer.setSeriesShapesVisible(0, false);
        renderer.setSeriesLinesVisible(1, true);
        renderer.setSeriesShapesVisible(1, false);
        renderer.setSeriesPaint(0, Color.GREEN);
        renderer.setSeriesPaint(1, Color.RED);
        plot.setRenderer(renderer);

        saveChart(chart, filePath);
    }

    private static double findMaxInRange(List<Double> data, int start, int end) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = start; i < end; i++) {
            if (data.get(i) > max) {
                max = data.get(i);
            }
        }
        return max;
    }

    public static void saveConfusionMatrixChart(String title, int[][] confusionMatrix, String filePath) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        // Populate dataset with confusion matrix values
        String[] classes = {"Positive", "Negative"};
        dataset.addValue(confusionMatrix[0][0], "True Positive", classes[0]);  // TP
        dataset.addValue(confusionMatrix[0][1], "False Positive", classes[1]);  // FP
        dataset.addValue(confusionMatrix[1][0], "False Negative", classes[0]);  // FN
        dataset.addValue(confusionMatrix[1][1], "True Negative", classes[1]);  // TN

        // Create the chart
        JFreeChart chart = ChartFactory.createBarChart(title, "Predicted Class", "Count", dataset);

        // Customize plot for confusion matrix (color and appearance)
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setRangeGridlinePaint(Color.BLACK);

        BarRenderer renderer = new BarRenderer();
        renderer.setSeriesPaint(0, Color.GREEN);  // True Positive
        renderer.setSeriesPaint(1, Color.RED);    // False Positive / False Negative
        renderer.setSeriesPaint(2, Color.BLUE);   // True Negative
        plot.setRenderer(renderer);

        // Customize axis
        CategoryAxis domainAxis = plot.getDomainAxis();
        domainAxis.setCategoryMargin(0.2);

        ValueAxis rangeAxis = plot.getRangeAxis();
        rangeAxis.setUpperMargin(0.15);  // Adjust for visual clarity

        saveChart(chart, filePath);
    }

    private static void saveChart(JFreeChart chart, String filePath) {
        File chartFile = new File(filePath);
        try {
            ChartUtils.saveChartAsPNG(chartFile, chart, 800, 600);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
