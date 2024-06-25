package util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class CustomChartUtils {
    public static void plotTrainingProgress(List<Double> trainingLoss, List<Double> validationLoss) {
        XYSeries trainingSeries = new XYSeries("Training Loss");
        XYSeries validationSeries = new XYSeries("Validation Loss");

        for (int i = 0; i < trainingLoss.size(); i++) {
            trainingSeries.add(i + 1, trainingLoss.get(i));
            validationSeries.add(i + 1, validationLoss.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(trainingSeries);
        dataset.addSeries(validationSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Training and Validation Loss",
                "Epoch",
                "Loss",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        XYPlot plot = chart.getXYPlot();
        plot.getRenderer().setSeriesPaint(0, new java.awt.Color(0xFF6600));
        plot.getRenderer().setSeriesPaint(1, new java.awt.Color(0x0066FF));

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));

        JFrame frame = new JFrame("Training Progress");
        frame.setContentPane(chartPanel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public static void saveAccuracyChart(String title, List<Integer> epochs, List<Double> accuracies, String filePath, String xLabel, String yLabel) throws IOException {
        XYSeries accuracySeries = new XYSeries("Accuracy");

        for (int i = 0; i < epochs.size(); i++) {
            accuracySeries.add(epochs.get(i), accuracies.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(accuracySeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                xLabel,
                yLabel,
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        ChartUtils.saveChartAsPNG(new File(filePath), chart, 800, 600);
    }

    public static void savePredictionChart(String title, double[] predictions, String filePath, String xLabel, String yLabel) throws IOException {
        XYSeries predictionSeries = new XYSeries("Predictions");

        for (int i = 0; i < predictions.length; i++) {
            predictionSeries.add(i + 1, predictions[i]);
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(predictionSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                xLabel,
                yLabel,
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        ChartUtils.saveChartAsPNG(new File(filePath), chart, 800, 600);
    }
}
