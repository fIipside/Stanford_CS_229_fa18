import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    # Train logistic regression
    modelc = LogisticRegression(eps=1e-5)
    modelc.fit(x_train, t_train)

    # Plot data and decision boundary
    util.plot(x_train, t_train, modelc.theta, save_path='output/p02c_train.png')
    util.plot(x_test, t_test, modelc.theta, save_path='output/p02c_test.png')

    pred_c = modelc.predict(x_test)
    np.savetxt(pred_path_c, pred_c > 0.5, fmt='%d')
    # print("The accuracy on testing set is: ", np.mean(t_test == (pred_c > 0.5)))
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    # Train logistic regression
    modeld = LogisticRegression(eps=1e-5)
    modeld.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, modeld.theta, save_path='output/p02d_train.png')
    util.plot(x_test, y_test, modeld.theta, save_path='output/p02d_test.png')

    pred_d = modeld.predict(x_test)
    np.savetxt(pred_path_d, pred_d > 0.5, fmt='%d')
    # print("The accuracy on testing set is: ", np.mean(t_test == (pred_d > 0.5)))
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    
    # Calculate alpha **Our definition of predict is different from the solution**
    alpha = np.mean(modeld.predict(x_valid))

    # Calculate correction term
    cor = 1 + np.log(2 / alpha - 1) / modeld.theta[0]

    # Plot data and decision boundary
    util.plot(x_train, y_train, modeld.theta, save_path='output/p02e_train.png', correction=cor)
    util.plot(x_test, y_test, modeld.theta, save_path='output/p02e_test.png', correction=cor)

    pred_e = pred_d / alpha
    np.savetxt(pred_path_e, pred_e > 0.5, fmt='%d')
    print("The accuracy on testing set is: ", np.mean(t_test == (pred_e > 0.5)))
    # *** END CODER HERE
