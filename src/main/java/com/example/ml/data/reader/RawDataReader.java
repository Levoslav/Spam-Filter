/**
 * Provides classes for reading machine learning data.
 * This package includes various readers for different data formats and sources.
 */
package com.example.ml.data.reader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;


/**
 * A class for reading raw data files and extracting the content and labels.
 */
public class RawDataReader {
    // List of documents in String form
    ArrayList<String> RawData;
    // List of corresponding labels
    ArrayList<Integer> labels;


    /**
     * Loads the raw data files from the specified directory path and populates the RawData and labels lists.
     *
     * @param file_path The path to the directory containing the raw data files.
     * @throws IOException If an I/O error occurs while reading the files.
     */
    public void load(String file_path) throws IOException {
        String directoryPath = file_path;

        File directory = new File(directoryPath);
        File[] files = directory.listFiles();

        RawData = new ArrayList<>();
        labels = new ArrayList<>();
        for (File file : files) {
            if (file.isFile() && file.getName().endsWith(".txt")) {
                Path filePath = Paths.get(file.getAbsolutePath());
                String content = new String(Files.readAllBytes(filePath));

                if (file.getName().endsWith(".ham.txt")) {
                    labels.add(0); // 0 = ham
                    RawData.add(content);
                } else if(file.getName().endsWith(".spam.txt")) {
                    labels.add(1); // 1 = spam
                    RawData.add(content);
                }
            }
        }

    }


    /**
     * Returns the list of raw data content.
     *
     * @return The list of raw data content.
     */
    public ArrayList<String> get_data() {
        return RawData;
    }


    /**
     * Returns the list of labels corresponding to the raw data.
     *
     * @return The list of labels.
     */
    public ArrayList<Integer> get_labels() {
        return labels;
    }


    /**
     * Clears the raw data and labels lists to free space.
     */
    public void clear() {
        RawData = null;
        labels = null;
    }
}