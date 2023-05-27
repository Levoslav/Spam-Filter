/**
 * Provides classes for reading machine learning data.
 * This package includes various readers for different data formats and sources.
 */
package com.example.ml.data.reader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


/**
 * Reads data from a text file and provides methods to access the data and labels.
 */
public class TxtDataReader {
    double[][] data;
    int[] labels;


    /**
     * Loads data from the specified text file.
     *
     * @param filePath The path to the text file.
     * @param n The number of features + 1.
     */
    public void load(String filePath, int n) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            int numLines = 0;
            while ((line = reader.readLine()) != null) {
                numLines++;
            }

            data = new double[numLines][n];
            labels = new int[numLines];

            reader.close();

            BufferedReader newReader = new BufferedReader(new FileReader(filePath));
            int lineIndex = 0;
            while ((line = newReader.readLine()) != null) {
                String[] tokens = line.split(" ");

                for (int i = 0; i < n; i++) {
                    data[lineIndex][i] = Double.parseDouble(tokens[i]);
                }

                labels[lineIndex] = Integer.parseInt(tokens[n]);
                lineIndex++;
            }
            newReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    /**
     * Returns the list of data content.
     *
     * @return The list of data content.
     */
    public double[][] get_data() {
        return data;
    }


    /**
     * Returns the list of labels corresponding to the data.
     *
     * @return The list of labels.
     */
    public int[] get_labels() {
        return labels;
    }


    /**
     * Clears the  data and labels lists to free space.
     */
    public void clear() {
        data = null;
        labels = null;
    }
}
