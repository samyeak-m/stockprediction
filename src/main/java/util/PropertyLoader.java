package util;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class PropertyLoader {
    public static Properties loadProperties(String fileName) {
        Properties properties = new Properties();

        try (InputStream input = PropertyLoader.class.getClassLoader().getResourceAsStream(fileName)) {
            if (input == null) {
                System.out.println("Sorry, unable to find " + fileName);
                return null;
            }

            properties.load(input);
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return properties;
    }
}
