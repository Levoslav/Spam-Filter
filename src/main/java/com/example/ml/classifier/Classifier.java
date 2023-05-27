/**
 * The "com.example.ml.classifier" package contains various machine learning classifiers(so far Gaussian naive bayes only).
 * This package provides an interface and implementations for different types of classifiers
 * used for training models and making predictions on new data.
 */
package com.example.ml.classifier;


/**
 * The Classifier interface represents a machine learning classifier.
 * Classes implementing this interface are responsible for fitting the classifier model
 * on the training data and making predictions on new data.
 *
 * Implementing classes must provide implementations for the `fit` and `predict` methods.
 */
public interface Classifier {
    public void fit(double[][] X, int[] y);
    public int[] predict(double[][] X);

}
