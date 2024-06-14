package database;

import util.PropertyLoader;

import java.sql.*;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class DatabaseHelper {
    private final String url;
    private final String username;
    private final String password;

    public DatabaseHelper() {
        Properties properties = PropertyLoader.loadProperties("application.properties");
        this.url = properties.getProperty("db.url");
        this.username = properties.getProperty("db.username");
        this.password = properties.getProperty("db.password");
    }

    public Connection connect() throws SQLException {
        return DriverManager.getConnection(url, username, password);
    }

    public List<String> getAllStockTableNames() throws SQLException {
        List<String> tableNames = new ArrayList<>();
        String query = "SHOW TABLES LIKE 'daily_data_%'";

        try (Connection conn = connect();
             PreparedStatement pstmt = conn.prepareStatement(query);
             ResultSet rs = pstmt.executeQuery()) {
            while (rs.next()) {
                String tableName = rs.getString(1);
                tableNames.add(tableName);
            }
        }
        return tableNames;
    }

    public List<double[]> loadStockData(String tableName) throws SQLException {
        List<double[]> stockData = new ArrayList<>();
        String query = "SELECT * FROM " + tableName + " ORDER BY date";

        try (Connection conn = connect();
             PreparedStatement pstmt = conn.prepareStatement(query);
             ResultSet rs = pstmt.executeQuery()) {
            while (rs.next()) {
                double[] data = new double[2];
                data[0] = rs.getDate("date").getTime();
                data[1] = rs.getDouble("close");
                stockData.add(data);
            }
        }
        return stockData;
    }

    public void savePrediction(String stockName, String predict, String pointChangeStr, double priceChange, double finalPrice, double currentPrice, LocalDate today, LocalDate predictionDate) throws SQLException {
        String createTableQuery = "CREATE TABLE IF NOT EXISTS nepse_price (" +
                "symbol VARCHAR(50), " +
                "predict VARCHAR(10), " +
                "point VARCHAR(10), " +
                "price DOUBLE, " +
                "final_price DOUBLE, " +
                "current_price DOUBLE, " +
                "date DATE, " +
                "prediction_date DATE" +
                ")";
        String insertQuery = "INSERT INTO nepse_price (symbol, predict, point, price, final_price, current_price, date, prediction_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";

        try (Connection conn = connect();
             Statement stmt = conn.createStatement()) {
            stmt.execute(createTableQuery);
        }

        try (Connection conn = connect();
             PreparedStatement pstmt = conn.prepareStatement(insertQuery)) {
            pstmt.setString(1, stockName);
            pstmt.setString(2, predict);
            pstmt.setString(3, pointChangeStr);
            pstmt.setDouble(4, priceChange);
            pstmt.setDouble(5, finalPrice);
            pstmt.setDouble(6, currentPrice);
            pstmt.setDate(7, Date.valueOf(today));
            pstmt.setDate(8, Date.valueOf(predictionDate));
            pstmt.executeUpdate();
        }
    }
}
