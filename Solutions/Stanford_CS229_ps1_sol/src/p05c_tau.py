import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    def plot(x, y_label, y_pred, savedir):
        plt.figure()
        plt.plot(x[:,-1], y_label, 'bx', label='label')
        plt.plot(x[:,-1], y_pred, 'ro', label='prediction')
        plt.legend(loc='upper left')
        plt.savefig('output/p05c_{}.png'.format(savedir))

    # Search tau_values for the best tau (lowest MSE on the validation set)
    best_tau = tau_values[0]
    best_fit = 1e10
    
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    for tau in tau_values:
        model_tau = LocallyWeightedLinearRegression(tau=tau)
        model_tau.fit(x_train, y_train)
        y_pred = model_tau.predict(x_valid)

        mse = np.mean((y_valid - y_pred)**2)

        if mse < best_fit:
            best_fit = mse
            best_tau = tau
        
        plot(x_valid, y_valid, y_pred, 'tau{}'.format(tau))

    print(f'Tau = {best_tau} achieves the lowest MSE = {best_fit} on the validation set.')
    
    # Fit a LWR model with the best tau value
    model_best = LocallyWeightedLinearRegression(tau=best_tau)
    model_best.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = model_best.predict(x_test)
    mse = np.mean((y_test - y_pred)**2)
    print(f'MSE = {mse}.')
    
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    
    # Plot data
    # The function is at the beginning
    # *** END CODE HERE ***
