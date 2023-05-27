# SpamFilter

The SpamFilter project is an implementation of a Gaussian Naive Bayes classifier combined with a TF-IDF (Term Frequency-Inverse Document Frequency) transformer to create feature vectors for data. The goal of this project is to classify emails as spam or non-spam (ham) based on their content.

## Features

- Utilizes the Gaussian Naive Bayes algorithm for classification.
- Applies a TF-IDF transformer to convert email data into numerical feature vectors.
- Uses Enron data sets for training and testing the classifier.
- Provides accurate and efficient spam detection. 
- Measures accuracy and F1 score. 

## Usage

1. Run the `Main` class to execute the program.

   The program will perform the following steps:

    - Transform the training data using the TF-IDF (Term Frequency-Inverse Document Frequency) transformer to create feature vectors.
    - Train the Gaussian Naive Bayes classifier on the transformed training data.
    - Transform the testing data using the same TF-IDF transformer.
    - Predict the labels of the transformed testing data using the trained classifier.
    - Measure the performance of the classifier by calculating accuracy, precision, recall, and F1-score.
    - Print the performance metrics to the console.


2. Review the printed performance metrics:

    - Accuracy: Indicates the overall correctness of the classifier's predictions.
    - Precision: Measures the proportion of correctly classified positive instances out of all instances classified as positive.
    - Recall: Measures the proportion of correctly classified positive instances out of all actual positive instances.
    - F1-score: Represents the harmonic mean of precision and recall, providing a balanced measure between the two.

   Analyzing these metrics will give you insights into the performance of the SpamFilter program.







