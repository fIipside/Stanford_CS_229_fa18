import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    mse = np.mean((y_eval - y_pred)**2)
    print(f'MSE={mse}')

    def plot(x, y_label, y_pred, savedir):
        plt.figure()
        plt.plot(x[:,-1], y_label, 'bx', label='label')
        plt.plot(x[:,-1], y_pred, 'ro', label='prediction')
        plt.legend(loc='upper left')
        plt.savefig('output/p05b_{}.png'.format(savedir))

    # Plot validation predictions on top of training set
    y_train_pred = model.predict(x_train)
    plot(x_train, y_train, y_train_pred, 'train')
 
    # No need to save predictions
    # Plot data
    plot(x_eval, y_eval, y_pred, 'eval')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        # Below codes are from solutions
        # Reshape the input x by adding an additional dimension so that it can broadcast. self.x Shape: (l, n)
        w_vector = np.exp(-(np.linalg.norm(self.x - np.reshape(x, (m, -1, n)), ord=2, axis=2))**2 / (2*self.tau**2))
        
        # Turn the weights into diagonal matrices, each corresponds to a single input. Shape (m, l, l)
        w = np.apply_along_axis(np.diag, axis=1, arr=w_vector)

        # Compute theta for each input x^(i). Shape (m, n)
        theta = np.linalg.inv(self.x.T @ w @ self.x) @ self.x.T @ w @ self.y
        
        # np.einsum('ij,ij->i', x, theta)
        return (x * theta).sum(axis=1)
        # *** END CODE HERE ***
