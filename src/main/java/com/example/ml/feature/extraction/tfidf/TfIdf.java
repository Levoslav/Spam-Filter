/**
 * This package contains classes related to TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction.
 * TF-IDF is a numerical statistic used to reflect the importance of a term in a collection of documents.
 */
package com.example.ml.feature.extraction.tfidf;

import com.example.ml.feature.extraction.Extractor;
import java.util.*;


/**
 * The TfIdf class implements the Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction technique.
 * TF-IDF is a numerical statistic that reflects the importance of a term in a document within a collection
 * of documents. It combines the term frequency (TF) and inverse document frequency (IDF) measures to compute
 * feature vectors representing the textual data.
 *
 * This class implements the Extractor interface and provides the fit and transform methods for fitting the TF-IDF model
 * on a collection of documents and transforming the documents into their TF-IDF feature representations.
 */
public class TfIdf implements Extractor{
    // Place to save the created feature vectors
    public double[][] data;
    // Vocabulary of all used tokens in training data with corresponding index in feature vector
    public HashMap<String, Integer> vocabulary;
    // IDF vector
    public double[] idf_vector;
    // Pointer to "data"
    private int pointer = 0;


    /**
     * Calculates the Term Frequency (TF) feature vector for a given document based on a dictionary.
     *
     * @param document   The input document as a String.
     * @param dictionary A HashMap representing the dictionary of terms and their corresponding indices.
     * @return A double array representing the TF feature vector for the document.
     */
    private double[] TF(String document, HashMap<String, Integer> dictionary) {
        // Create feature vector of zeros
        double[] new_sample = new double[dictionary.size()];
        int term_count = 0;
        for (String term : document.split("\\W+")) {
            if (dictionary.containsKey(term)) {
                new_sample[dictionary.get(term)] += 1;
            }
            term_count++;
        }
        // divide all counts of terms from document with the term_count
        for (int i = 0; i < dictionary.size(); i++) {
            new_sample[i] = new_sample[i]/term_count;
        }
        return new_sample;
    }


    /**
     * Computes the Inverse Document Frequency (IDF) feature vector using the document count and dictionary.
     *
     * @param documents_count      The total number of documents.
     * @param dictionary_doc_count A map representing the document count for each term in the dictionary.
     * @param dictionary           A map representing the dictionary of terms and their corresponding indices.
     * @return A double array representing the IDF feature vector.
     */
    private static double[] IDF(int documents_count, Map<String, Integer> dictionary_doc_count, Map<String, Integer> dictionary) {
        // Create feature vector of zeros
        double[] new_sample = new double[dictionary.size()];

        // Go through all terms and for each compute inverse document frequency(IDF) and save it to "correct place" in resulting vector
        for (String term: dictionary.keySet()) {
            int term_doc_count = dictionary_doc_count.getOrDefault(term, 0);
            new_sample[dictionary.get(term)] = Math.log((double) documents_count / (term_doc_count + 1));
        }
        return new_sample;
    }


    /**
     * Performs element-wise multiplication between two double arrays and returns the result.
     *
     * @param x The first double array.
     * @param y The second double array.
     * @return A double array resulting from element-wise multiplication of x and y.
     */
    private static double[] multiply(double[] x, double[] y) {
        double[] z = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            z[i] = x[i] * y[i];
        }
        return z;
    }


    /**
     * Fits the feature extraction model on the provided list of test documents and transforms the test data.
     *
     * @param X An ArrayList of strings representing the input test documents.
     * @return A double array representing the transformed test data.
     */
    public double[][] fit_transform_testdata(ArrayList<String> X) {
        double[][] test_data = new double[X.size()][vocabulary.size()];
        int i = 0;
        for (String document: X) {
            test_data[i] = multiply(TF(document, vocabulary), idf_vector);
            i++;
        }
        return test_data;
    }


    /**
     * Fits the feature extraction model on the provided list of documents.
     *
     * @param X An ArrayList of strings representing the input documents.
     */
    @Override
    public void fit(ArrayList<String> X) {
        // Dictionary where key = term t, value = number of occurrences of the term t in all documents
        HashMap<String, Integer> dictionary = new HashMap<>();
        // Dictionary where key = term t, value = number of documents containing t
        HashMap<String, Integer> dictionary_doc_count = new HashMap<>();
        // Set of terms in document d
        HashSet<String> used_terms = new HashSet<>();

        // Place to store already created feature vectors
        data = new double[X.size()][];

        for (String document : X) {
            used_terms.clear();
            for (String term : document.split("\\W+")) {
                if (dictionary.containsKey(term)) {
                    dictionary.put(term, dictionary.get(term) + 1);
                } else {
                    dictionary.put(term, 1);
                    dictionary_doc_count.put(term, 0);
                }
                used_terms.add(term);
            }
            for (String term : used_terms) {
                dictionary_doc_count.put(term, dictionary_doc_count.get(term) + 1);
            }
        }

        // Remove all terms which occurred only once
        List<String> delete = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : dictionary.entrySet()) {
            if (dictionary_doc_count.get(entry.getKey()) < 2) {
                delete.add(entry.getKey());
            }
        }
        for (String key : delete) {
            dictionary.remove(key);
            dictionary_doc_count.remove(key);
        }

        // We don't need counts of terms from "dictionary". We will recycle "dictionary" for indexing feature vectors
        int i = 0;
        for (String key : dictionary.keySet()) {
            dictionary.put(key, i++);
        }

        vocabulary = dictionary;

        // compute term frequency for each document (create feature vector for each document)
        for (String document: X) {
            data[pointer] = TF(document, dictionary);
            pointer++;
        }

        // compute idf vector
        idf_vector = IDF(X.size(), dictionary_doc_count, dictionary);

        // apply idf vector to all data
        for (int j = 0; j < X.size();j++) {
            data[j] = multiply(data[j],idf_vector);
        }
    }


    /**
     * Transforms the input data using the fitted feature extraction model.
     *
     * @return A double array representing the transformed data.
     */
    @Override
    public double[][] transform() {
        return data;
    }
}