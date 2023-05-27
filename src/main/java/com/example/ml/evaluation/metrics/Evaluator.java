/**
 * The com.example.ml.evaluation.metrics package provides a collection of evaluation metrics for assessing the performance of machine learning models.
 */
package com.example.ml.evaluation.metrics;

/**
 * The Evaluator class provides methods for evaluating the performance of a classification model.
 */
public class Evaluator {

    /**
     * Evaluates the performance of a classification model based on the true labels and predicted labels.
     *
     * @param true_y      The true labels.
     * @param prediction_y The predicted labels.
     */
    public static void evaluate(int[] true_y, int[] prediction_y ) {
        // # of TruePositives
        int TP = 0;
        // # of FalsePositives
        int FP = 0;
        // # of TrueNegatives
        int TN = 0;
        // # of FalseNegatives
        int FN = 0;
        // # of all predictions
        int total = true_y.length;


        for (int i = 0; i < total; i++) {
            if (prediction_y[i] == 0 && true_y[i] == 0) {
                TN++;
            } else if (prediction_y[i] == 0 && true_y[i] == 1) {
                FN++;
            } else if (prediction_y[i] == 1 && true_y[i] == 0) {
                FP++;
            } else {
                TP++;
            }
        }

        double accuracy = (TP + TN)/(double)total;
        double precision  = (double)TP / (TP+FP);
        double recall = (double)TP/ (TP+FN);
        double F1score = (2*precision*recall)/(precision+recall);

        System.out.println("------------------------------------------");
        System.out.print("Accuracy: ");
        System.out.println(accuracy);
        System.out.print("Precision: ");
        System.out.println(precision);
        System.out.print("Recall: ");
        System.out.println(recall);
        System.out.print("F1-score: ");
        System.out.println(F1score);
        System.out.println("------------------------------------------");

    }
}
