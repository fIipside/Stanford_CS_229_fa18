import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train Logistic model
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # The following code is not by myself
    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01b_{}.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # Initialization
        m, n = x.shape
        self.theta = np.zeros(n)

        # Newton's Method
        while True:
            theta_old = np.copy(self.theta)

            h = 1/(1 + np.exp(-np.dot(x, self.theta)))
            gradient = -1/m * (np.dot(x.T, (y - h)))
            # **How to calculate H**
            # NumPy implicitly adds a new axis on the left, transforming its shape from (m,) to (1, m)
            H = 1/m * np.dot(x.T * (h * (1-h)), x)
            
            # Update theta
            self.theta -= np.dot(np.linalg.inv(H), gradient)

            # End training
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1/(1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***
