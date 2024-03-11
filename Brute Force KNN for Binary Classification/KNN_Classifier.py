from typing import Callable

import numpy as np


class KNN_Classifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.k = None
        self.distance_metric = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int, distance_metric: str) -> None:
        """
        :brief: Fits the model to the training data.
        :param X_train: Training data.
        :param y_train: Training labels.
        :param k: Number of neighbors to consider.
        :param distance_metric: Distance metric to use.
        :return: None.

        This function stores the training data and labels, as well as the number of neighbors to consider and the
        distance metric to use.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.distance_metric = distance_metric

        return 0

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        :brief: Predicts the labels of the test data.
        :param X_test: Test data.
        :return: Predicted labels.

        This function works by calculating the distance between each test sample and all the training samples, and then
        selecting the k nearest neighbors. The predicted label for each test sample is the majority label among its
        k nearest neighbors.
        """
        return self.__k_nearest_neighbors__(self.X_train, self.y_train, X_test, self.k, self.distance_metric)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        :brief: Calculates the accuracy of predictions and return all related data for further analysis, precision,
        recall, f1-score, confusion matrix, roc curve and auc, and precision-recall curve.
        :param X_test:
        :param y_test:
        :return:
        """
        y_pred = self.predict(X_test)
        return self.__accuracy__(y_test, y_pred, self.__loss__), self.__precision__(y_test, y_pred), \
            self.__recall__(y_test, y_pred), self.__f1_score__(y_test, y_pred), self.__confusion_matrix__(y_test,
                                                                                                          y_pred), \
            self.__roc_curve__(y_test, y_pred), self.__auc__(y_test, y_pred), self.__precision_recall_curve__(y_test,
                                                                                                              y_pred), \
            self.__average_precision__(y_test, y_pred)

    def __euclidean_distance__(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        :brief: Calculates the Euclidean distance between two points.
        :param p1: A numpy array representing the first point in 2-dimensional space.
        :param p2: A numpy array representing the second point in 2-dimensional space.
        :return: The Euclidean distance between the two points.

        This function works by subtracting each dimension of p1 from the corresponding dimension of p2, squaring the
        result, summing all these squares together and then taking the square root of the sum.
        """
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def __manhattan_distance__(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        :brief: Calculates the Manhattan distance between two points.
        :param p1: A numpy array representing the first point in 2-dimensional space.
        :param p2: A numpy array representing the second point in 2-dimensional space.
        :return: The Manhattan distance between the two points.

        This function works by subtracting each dimension of p1 from the corresponding dimension of p2, taking the
        absolute value of the result, and then summing all these absolute values together.
        """
        return np.sum(np.abs(p1 - p2))

    def __accuracy__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        :brief: Calculates the accuracy of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: Accuracy.

        This function works by comparing each element of y_true with the corresponding element of y_pred. If they are
        equal, it's a correct prediction. The accuracy is then the number of correct predictions divided by the total
        number of predictions.
        """
        assert len(y_true) == len(y_pred), "Input vectors must have the same length"
        correct_predictions = np.sum(y_true == y_pred)
        total_samples = len(y_true)
        accuracy = correct_predictions / total_samples

        return accuracy

    def __precision__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        :brief: Calculates the precision of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: Precision.

        This function works by calculating the number of true positive predictions (TP), and the number of false
        positive predictions (FP). The precision is then the number of true positive predictions divided by the sum
        of true positive and false positive predictions.
        """
        assert len(y_true) == len(y_pred), "Input vectors must have the same length"
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))
        precision = true_positive / (true_positive + false_positive)

        return precision

    def __generalization_error__(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: Callable) -> float:
        """
        :brief: Calculates the generalization error of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :param loss_function: Loss function that takes (y_true, y_pred) as input.
        :return: Generalization error.

        This function works by applying the loss function to each pair of corresponding elements in y_true and y_pred,
        and then taking the average of all the resulting loss values.
        """
        assert len(y_true) == len(y_pred), "Input vectors must have the same length"
        loss_values = loss_function(y_true, y_pred)
        generalization_error = np.mean(loss_values)

        return generalization_error

    def __recall__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        :brief: Calculates the recall of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: Recall.

        This function works by calculating the number of true positive predictions (TP), and the number of false
        negative predictions (FN). The recall is then the number of true positive predictions divided by the sum of
        true positive and false negative predictions.
        """
        assert len(y_true) == len(y_pred), "Input vectors must have the same length"
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_negative = np.sum((y_true == 1) & (y_pred == 0))
        recall = true_positive / (true_positive + false_negative)

        return recall

    def __f1_score__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        :brief: Calculates the F1 score of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: F1 score.

        This function works by calculating the precision and recall of the predictions, and then using these values
        to calculate the F1 score.
        """
        precision = self.__precision__(y_true, y_pred)
        recall = self.__recall__(y_true, y_pred)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score

    def __confusion_matrix__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        :brief: Calculates the confusion matrix of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: Confusion matrix.

        This function works by calculating the number of true positive, false positive, true negative and false
        negative predictions, and then storing these values in a 2x2 numpy array.
        """
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))
        true_negative = np.sum((y_true == 0) & (y_pred == 0))
        false_negative = np.sum((y_true == 1) & (y_pred == 0))

        confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])

        return confusion_matrix

    def __roc_curve__(self, y_true: np.ndarray, y_pred: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        :brief: Calculates the ROC curve of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: FPR and TPR.

        This function works by calculating the true positive rate (TPR) and false positive rate (FPR) for different
        thresholds, and then returning these values as numpy arrays.
        """
        thresholds = np.unique(y_pred)
        tpr = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))

        for i, threshold in enumerate(thresholds):
            y_pred_thresholded = (y_pred >= threshold).astype(int)
            confusion_matrix = self.__confusion_matrix__(y_true, y_pred_thresholded)
            tpr[i] = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
            fpr[i] = confusion_matrix[0, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])

        return fpr, tpr

    def __auc__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        :brief: Calculates the AUC of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: AUC.

        This function works by calculating the ROC curve of the predictions, and then using the trapezoidal rule to
        calculate the area under the curve.
        """
        fpr, tpr = self.__roc_curve__(y_true, y_pred)
        auc = np.trapz(tpr, fpr)

        return auc

    def __precision_recall_curve__(self, y_true: np.ndarray, y_pred: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        :brief: Calculates the precision-recall curve of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: Precision and recall.

        This function works by calculating the precision and recall for different thresholds, and then returning
        these values as numpy arrays.
        """
        thresholds = np.unique(y_pred)
        precision = np.zeros(len(thresholds))
        recall = np.zeros(len(thresholds))

        for i, threshold in enumerate(thresholds):
            y_pred_thresholded = (y_pred >= threshold).astype(int)
            confusion_matrix = self.__confusion_matrix__(y_true, y_pred_thresholded)
            precision[i] = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
            recall[i] = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])

        return precision, recall

    def __average_precision__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        :brief: Calculates the average precision of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: Average precision.

        This function works by calculating the precision-recall curve of the predictions, and then using the
        trapezoidal rule to calculate the area under the curve.
        """
        precision, recall = self.__precision_recall_curve__(y_true, y_pred)
        average_precision = np.trapz(precision, recall)

        return average_precision

    def __loss__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        :brief: Calculates the loss of predictions.
        :param y_true: True labels (ground truth).
        :param y_pred: Predicted labels.
        :return: Loss.

        This function works by calculating the loss for each pair of corresponding elements in y_true and y_pred,
        and then returning these values as a numpy array.
        """
        loss = (y_true - y_pred) ** 2

        return loss

    def __k_nearest_neighbors__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int,
                                distance_metric: str) -> np.ndarray:
        """
        :brief: Finds the k nearest neighbors of each test sample.
        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_test: Test data.
        :param k: Number of neighbors to consider.
        :param distance_metric: Distance metric to use.
        :return: Predicted labels.

        This function works by calculating the distance between each test sample and all the training samples, and then
        selecting the k nearest neighbors. The predicted label for each test sample is the majority label among its
        k nearest neighbors.
        """
        y_pred = np.zeros(len(X_test))

        for i, x_test in enumerate(X_test):
            distances = np.zeros(len(X_train))

            for j, x_train in enumerate(X_train):
                if distance_metric == "euclidean":
                    distances[j] = self.__euclidean_distance__(x_test, x_train)
                elif distance_metric == "manhattan":
                    distances[j] = self.__manhattan_distance__(x_test, x_train)

            nearest_neighbors = np.argsort(distances)[:k]
            nearest_labels = y_train[nearest_neighbors]
            y_pred[i] = np.bincount(nearest_labels).argmax()

        return y_pred
