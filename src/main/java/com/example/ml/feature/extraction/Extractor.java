/**
 * This package contains classes related to feature extraction techniques used in machine learning.
 * Various feature extraction methods such as TF-IDF, word embeddings(not yet but might include), etc., can be found within this package.
 */
package com.example.ml.feature.extraction;
import java.util.ArrayList;


/**
 * The Extractor interface represents a feature extraction method in machine learning.
 * Classes implementing this interface are responsible for fitting the feature extraction model
 * on the input data and transforming the data based on the fitted model.
 */
public interface Extractor {
    void fit(ArrayList<String> X);
    double[][] transform();
}