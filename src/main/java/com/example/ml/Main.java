package com.example.ml;

import java.io.IOException;
import java.util.*;
import com.example.ml.data.reader.RawDataReader;
import com.example.ml.data.saver.Saver;
import com.example.ml.feature.extraction.tfidf.TfIdf;
import com.example.ml.classifier.naivebayes.GaussianNaiveBayes;
import com.example.ml.evaluation.metrics.Evaluator;
import java.net.URL;

/**
 *
 * The Main class serves as the entry point for the program and demonstrates the usage of machine learning components for text classification.
 * It loads the training and testing data, applies the TF-IDF feature extraction, trains a Gaussian Naive Bayes classifier, and evaluates its performance.
 * The steps performed in the main method are as follows:
 * Initialize the TF-IDF transformer, raw data reader, and Gaussian Naive Bayes model.
 * Load the training data using the raw data reader and fit the TF-IDF transformer on it.
 * Transform the training data using the fitted transformer to obtain the TF-IDF features.
 * Get the training labels from the raw data reader.
 * Clear unnecessary data from memory using the clear method of the raw data reader and setting variables to null.
 * Train the Gaussian Naive Bayes model on the transformed training data and labels.
 * Load the testing data using the raw data reader and apply the fitted transformer to transform the testing data.
 * Get the testing labels from the raw data reader.
 * Clear unnecessary data from memory using the clear method of the raw data reader and setting variables to null.
 * Predict the labels for the testing data using the trained model.
 * Evaluate the performance of the model by comparing the predicted labels with the true labels using the Evaluator class.
 * This class serves as an example of how to use the different components together to build a text classification pipeline.
 * Developers can modify and extend this class to suit their specific needs for text classification tasks.
 */
public class Main {

    public static void main(String[] args) throws IOException {

        TfIdf transformer = new TfIdf();
        RawDataReader reader = new RawDataReader();
        GaussianNaiveBayes model = new GaussianNaiveBayes();

        URL TrainUrl = Main.class.getResource("/data/enron/hamnspam_train");
        URL TestUrl = Main.class.getResource("/data/enron/hamnspam_test");

        reader.load(TrainUrl.getFile());
        transformer.fit(reader.get_data());
        double[][] train_data = transformer.transform();
        int[] train_labels = reader.get_labels().stream().mapToInt(i -> i).toArray();
        reader.clear();  // To free up heap space

        System.out.println("Training model...");
        model.fit(train_data, train_labels);
        System.out.println("Model trained");

        // Free up the space
        train_labels = null;
        train_data = null;
        transformer.data = null;

        reader.load(TestUrl.getFile());
        double[][] test_data = transformer.fit_transform_testdata(reader.get_data());
        int[] test_labels = reader.get_labels().stream().mapToInt(i -> i).toArray();
        reader.clear();  // To free heap space

        int[] predictions = model.predict(test_data);

        Evaluator.evaluate(test_labels,predictions);

    }

}