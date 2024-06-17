package util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.time.LocalDate;
import java.util.List;

public class ChartUtils {

    public static void saveAccuracyChart(String title, List<Integer> epochs, List<Double> accuracies, String filePath) throws IOException {
        XYSeries accuracySeries = new XYSeries("Accuracy");
        for (int i = 0; i < epochs.size(); i++) {
            accuracySeries.add(epochs.get(i), accuracies.get(i));
        }
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(accuracySeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                "Epochs",
                "Accuracy",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        org.jfree.chart.ChartUtils.saveChartAsPNG(new File(filePath), chart, 800, 600);
    }

    public static void savePredictionChart(String title, List<LocalDate> dates, List<Double> actualPrices, List<Double> predictedPrices, String filePath) throws IOException {
        XYSeries actualSeries = new XYSeries("Actual Prices");
        XYSeries predictedSeries = new XYSeries("Predicted Prices");

        for (int i = 0; i < dates.size(); i++) {
            actualSeries.add(i, actualPrices.get(i));
            predictedSeries.add(i, predictedPrices.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(actualSeries);
        dataset.addSeries(predictedSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                "Days",
                "Price",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        org.jfree.chart.ChartUtils.saveChartAsPNG(new File(filePath), chart, 800, 600);
    }
}
