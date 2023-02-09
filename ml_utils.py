import numpy as np


# TODO: XGboost

methods =['RandomForest', 'LogisticRegression', 'DecisionTree','GaussianNB', 'SVM', 'MLP','GBC']
def train(train_features, train_labels, test_features, test_labels, method='GBC'):
    from sklearn.metrics import accuracy_score
    mdl = None
    if method == 'RandomForest':
        from sklearn.ensemble import RandomForestRegressor
        # Instantiate model with 1000 decision trees
        mdl = RandomForestRegressor(n_estimators=1000, random_state=42)
        # Train the model on training data
        mdl.fit(train_features, train_labels)

    elif method == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        mdl = LogisticRegression(random_state=42,solver='lbfgs',max_iter=1000).fit(train_features,train_labels)
    elif method == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        mdl = DecisionTreeClassifier(random_state=42).fit(train_features, train_labels)
    elif method == 'GaussianNB':
        from sklearn.naive_bayes import GaussianNB
        mdl = GaussianNB().fit(train_features, train_labels)
    elif method == 'SVM':
        from sklearn import svm
        mdl = svm.SVC(random_state=42).fit(train_features, train_labels)
    elif method == 'MLP':
        from sklearn.neural_network import MLPClassifier
        mdl = MLPClassifier(random_state=42).fit(train_features, train_labels)

    elif method == 'GBC':
        from sklearn.ensemble import GradientBoostingClassifier
        mdl = GradientBoostingClassifier(random_state=42).fit(train_features, train_labels)
    else:
        raise KeyError

    print('method is ' + method)

    # Use the forest's predict method on the test data
    predictions = mdl.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # threshold
    pred = []
    for val in predictions:
        if val > 0.5:
            pred.append(1)
        else:
            pred.append(0)

    # Calculate mean absolute percentage error (MAPE)
    # mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = accuracy_score(test_labels, pred)
    print('Accuracy:', round(accuracy, 2))

def train_all(train_features, train_labels, test_features, test_labels):
    for method in methods:
        train(train_features,train_labels,test_features,test_labels, method)

# random forest 0.65
# LogisticRegression 0.64
# DecisionTree 0.6
# Naive Bayes 0.63
# svm 0.63
# gradient boosting classifier 0.65
# MLP 0.65
