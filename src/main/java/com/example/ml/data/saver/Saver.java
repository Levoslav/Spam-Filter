/**
 * Provides classes for saving machine learning data.
 * This package includes various savers for different data formats and types.
 */
package com.example.ml.data.saver;

import java.io.FileWriter;
import java.io.IOException;


/**
 * A class for saving data to a text file.
 */
public class Saver {

    /**
     * Saves the data array and corresponding targets to a text file.
     *
     * @param arr     The data array to be saved.
     * @param targets The targets array corresponding to the data.
     * @param path    The path of the file to save the data.
     */
    public static void saveToTXT(double[][] arr,double[] targets ,String path) {
        int rows = arr.length;
        int cols = arr[0].length;

        try {
            FileWriter writer = new FileWriter(path);

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    writer.write(String.valueOf(arr[i][j]));
                    writer.write(" ");
                }
                writer.write(String.valueOf(targets[i]));
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
}
