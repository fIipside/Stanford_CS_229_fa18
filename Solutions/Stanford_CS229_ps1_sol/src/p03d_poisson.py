import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    model = PoissonRegression(step_size=2e-7)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred)

    # Below is from the solution
    import matplotlib.pyplot as plt

    def plot(y_label, y_pred, title):
        plt.plot(y_label, 'go', label='label')
        plt.plot(y_pred, 'rx', label='prediction')
        plt.suptitle(title, fontsize=12)
        plt.legend(loc='upper left')
        plt.savefig('output/p03d.png')

    plot(y_eval, y_pred, 'Training Set')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # Initialization
        m, n = x.shape
        self.theta = np.zeros(n)

        # Batch gradient ascent
        while True:
            old_theta = np.copy(self.theta)
            h = np.exp(np.dot(x, self.theta))
            
            # m steps indicates learning rate is 1/m step size
            self.theta += self.step_size/m * np.dot(y - h, x)
            if np.linalg.norm(self.theta - old_theta, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(np.dot(x, self.theta))
        # *** END CODE HERE ***
