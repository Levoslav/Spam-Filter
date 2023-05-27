/**
 * The "com.example.ml.classifier.naivebayes" package contains implementations of the Naive Bayes classifier.
 * This package provides classes and utilities for training and using Naive Bayes classifiers,
 * which are probabilistic models based on the Bayes' theorem and the assumption of feature independence.
 *
 * Implementations of the Naive Bayes classifier, such as Gaussian Naive Bayes
 * can be found within this package or in its sub-packages.
 */
package com.example.ml.classifier.naivebayes;

import com.example.ml.classifier.Classifier;
import java.lang.Math;


/**
 * The GaussianNaiveBayes class is an implementation of the Gaussian Naive Bayes classifier.
 * It is used for probabilistic classification tasks based on the assumption of Gaussian distribution
 * for the features in each class. The classifier learns the mean and variance of each feature
 * in each class during the training phase and uses them to make predictions on new data.
 *
 * This class implements the Classifier interface and provides the fit and predict methods
 * required for training the classifier and making predictions.
 */
public class GaussianNaiveBayes implements Classifier {
    // Small constant added to every computed variance to avoid division by zero
    private double var_smoothing = 1e-9;
    // Vector of variances for each feature (ham emails)
    private double[] ham_variances;
    // Vector of means for each feature (ham emails)
    private double[] ham_means;
    // Vector of variances for each feature (spam emails)
    private double[] spam_variances;
    // Vector of means for each feature (spam emails)
    private double[] spam_means;
    // Prior ham probability computed from the train data
    private double ham_probability;
    // Prior spam probability computed from the train data
    private double spam_probability;


    /**
     * Fits the Gaussian Naive Bayes classifier by estimating the mean and variance parameters for each feature in the training data.
     * The method computes the mean and variance for each feature separately for spam and ham classes.
     * It calculates the class probabilities based on the number of spam and ham instances in the training data.
     *
     * @param X The input training data matrix with shape [num_samples, num_features].
     * @param y The target labels array indicating the class of each training sample.
     *          The labels are binary, where 0 represents the ham class and 1 represents the spam class.
     */
    @Override
    public void fit(double[][] X, int[] y) {

        int num_columns = X[0].length;
        int num_rows = X.length;
        double spam_sum;
        double ham_sum;
        int spam_count = 0;
        int ham_count = 0;

        ham_variances = new double[num_columns];
        ham_means = new double[num_columns];
        spam_variances = new double[num_columns];
        spam_means = new double[num_columns];

        for (int j = 0; j < num_columns; j++) {
            spam_sum = 0;
            ham_sum = 0;
            spam_count = 0;
            ham_count = 0;
            for (int i = 0; i < num_rows; i++) {
                if (y[i] == 0) { // Ham
                    ham_count += 1;
                    ham_sum += X[i][j];
                } else { // Spam
                    spam_count += 1;
                    spam_sum += X[i][j];
                }
            }
            spam_means[j] = (spam_sum/spam_count);
            ham_means[j] = (ham_sum/ham_count);
        }


        for (int j = 0; j < num_columns; j++) {
            spam_sum = 0;
            ham_sum = 0;
            for (int i = 0; i < num_rows; i++) {
                if (y[i] == 0) { // Ham
                    ham_sum += Math.pow((X[i][j] - ham_means[j]), 2);
                } else { // Spam
                    spam_sum += Math.pow((X[i][j] - spam_means[j]),2);
                }
            }
            spam_variances[j] = (spam_sum/spam_count) + var_smoothing;
            ham_variances[j] = (ham_sum/ham_count) + var_smoothing;
        }

        ham_probability = ham_count/(double)(spam_count+ham_count);
        spam_probability = 1.0 - ham_probability;
    }


    /**
     * Predicts the class labels for the given test data using the fitted Gaussian Naive Bayes classifier.
     * The method calculates the log probabilities for each class (ham and spam) based on the learned parameters,
     * and assigns the class label with the higher log probability to each test sample.
     *
     * @param X The input test data matrix with shape [num_samples, num_features].
     * @return An array of predicted class labels for the test data, where 0 represents the ham class and 1 represents the spam class.
     */
    @Override
    public int[] predict(double[][] X) {
        int[] prediction = new int[X.length];
        double ham_log_prob = Math.log(ham_probability);
        double spam_log_prob = Math.log(spam_probability);
        int r = 0;

        for (double[] x: X) {
            ham_log_prob = Math.log(ham_probability);
            spam_log_prob = Math.log(spam_probability);

            for (int i = 0; i < x.length; i++) {
                ham_log_prob += log_gaussianDistribution(x[i], ham_means[i], ham_variances[i]);;
                spam_log_prob += log_gaussianDistribution(x[i], spam_means[i], spam_variances[i]);
            }

            if (ham_log_prob > spam_log_prob) {
                prediction[r] = 0;
            } else {
                prediction[r] = 1;
            }
            r++;

        }
        return prediction;
    }


    /**
     * Calculates the logarithm of the Gaussian distribution probability density function for a given value,
     * mean, and variance.
     *
     * @param x        The value for which to calculate the probability.
     * @param mean     The mean of the Gaussian distribution.
     * @param variance The variance of the Gaussian distribution.
     * @return The logarithm of the Gaussian distribution probability density.
     */
    private static double log_gaussianDistribution(double x, double mean, double variance) {
        double exponent = -(Math.pow((x - mean), 2))/(2 * variance);
        return -0.5 * Math.log(variance * 2 * Math.PI) + exponent;
    }
}