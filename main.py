# Import libraries necessary for this project
import numpy as np
import pandas as pd
import visuals as vs  # Supplementary code
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score  # Import 'r2_score'
# Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """
    # Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    # Return the score
    return score


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': [1,2,3,4,5,6,7,8,9,10]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


if __name__ == '__main__':

    # Load the Boston housing dataset
    data = pd.read_csv('housing.csv')
    prices = data['MDEV']
    features = data.drop('MDEV', axis=1)
    data_sorted = data.sort_values('MDEV')
    print(data_sorted.head())
    print(data_sorted.tail())
    print("\nQ1 : \n")
    d1 = data_sorted.quantile(0.25)
    print(d1)
    print("\nQ2 : \n")
    print(data_sorted.quantile(0.5))
    print("\nQ3 : \n")
    print(data_sorted.quantile(0.75))

    # Success
    print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

    # Minimum price of the data
    minimum_price = min(prices)

    # Maximum price of the data
    maximum_price = max(prices)

    # Mean price of the data
    mean_price = np.mean(prices)

    # Median price of the data
    median_price = np.median(prices)

    # Standard deviation of prices of the data
    std_price = np.std(prices)

    # Show the calculated statistics
    print("Statistics for Boston housing dataset:\n")
    print("Minimum price: ${:,.2f}".format(minimum_price))
    print("Maximum price: ${:,.2f}".format(maximum_price))
    print("Mean price: ${:,.2f}".format(mean_price))
    print("Median price ${:,.2f}".format(median_price))
    print("Standard deviation of prices: ${:,.2f}".format(std_price))

    # Calculate the performance of this model
    score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
    print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))

    # Import 'train_test_split'
    from sklearn.cross_validation import train_test_split
    # Shuffle and split the data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)
    # Success
    print("Training and testing split was successful.")

    # Produce learning curves for varying training set sizes and maximum depths
    # vs.ModelLearning(features, prices)
    # vs.ModelComplexity(X_train, y_train)

    # Fit the training data to the model using grid search
    reg = fit_model(X_train, y_train)
    # Produce the value for 'max_depth'
    print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

    ################################################################################################
    # Produce a matrix for client data
    client_data = [[5, 34, 15],  # Client 1
                   [4, 55, 22],  # Client 2
                   [8, 7, 12]]  # Client 3
    # Show predictions
    for i, price in enumerate(reg.predict(client_data)):
        print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i + 1, price))

    from sklearn.neighbors import NearestNeighbors
    num_neighbors = 2

    def nearest_neighbor_price(x):
        """Return NearestNeighbors."""
        def find_nearest_neighbor_indexes(x, X):  # x is your vector and X is the data set.
            neigh = NearestNeighbors(num_neighbors)
            neigh.fit(X)
            distance, indexes = neigh.kneighbors(x)
            return indexes

        indexes = find_nearest_neighbor_indexes(x, features)
        sum_prices = []
        sum_features = []
        for i in indexes:
            sum_prices.append(prices[i])
            sum_features.append(data.iloc[i])
        neighbor_avg = np.mean(sum_prices)
        print(x)
        print("\nK-NN Neighbors :")
        print(sum_features[0])
        print("\nAvg K-NN :")
        print(sum_features[0].apply(np.mean))
        print("\nStd dev K-NN :")
        print(sum_features[0].apply(np.std))
        return neighbor_avg

    index = 0
    for i in client_data:
        val = nearest_neighbor_price(i)
        index += 1
        print("The predicted {} nearest neighbors price for home {} is: ${:,.2f}".format(num_neighbors, index, val))
