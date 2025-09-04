import numpy as np
import util

from linear_model import LinearModel
from p01b_logreg import LogisticRegression


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    # **It will add an intercept term**
    x_train1, y_train1 = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train GDA
    model1 = GDA()
    model1.fit(x_train1, y_train1)

    x_train2, y_train2 = util.load_dataset(train_path, add_intercept=True)

    model2 = LogisticRegression()
    model2.fit(x_train2, y_train2)

    # Plot data and decision boundary
    util.plot(x_train1, y_train1, theta_1=model1.theta, legend_1='GDA', theta_2=model2.theta, legend_2='LogisticRegression', save_path='output/p01f_{}.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model1.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Initialization
        m, n = x.shape
        self.theta = np.zeros(n+1)
        phi = np.sum(y) / m
        mu_0 = np.dot(x.T, 1-y) / np.sum(1-y)
        mu_1 = np.dot(x.T, y) / np.sum(y)
        
        # **How to calculate sigma**
        sigma = (np.dot((x[y == 0] - mu_0).T, x[y == 0] - mu_0) + np.dot((x[y == 1] - mu_1).T, x[y == 1] - mu_1)) / m
        sigma_inv = np.linalg.inv(sigma)
        
        # Calculate theta
        self.theta[0] = np.log(phi / (1-phi)) + ((mu_0.T @ sigma_inv @ mu_0) - (mu_1.T @ sigma_inv @ mu_1))/2
        self.theta[1:] = np.dot(sigma_inv, (mu_1 - mu_0))

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1/(1 + np.exp(-np.dot(x, self.theta)))
        # *** END CODE HERE
