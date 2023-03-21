import pickle

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

methods = ['RandomForest', 'LogisticRegression', 'DecisionTree', 'GaussianNB', 'SVM', 'MLP', 'GBC', 'XGBoost']


def train(train_features, train_labels, test_features, test_labels, method='GBC', save_model=True):
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
        mdl = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000).fit(train_features, train_labels)
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
    elif method == 'XGBoost':
        import xgboost as xgb

        # Convert the training data to a DMatrix
        dtrain = xgb.DMatrix(train_features, label=train_labels)

        # Define the XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'eta': 0.1,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'seed': 42
        }

        # Train the XGBoost model
        mdl = xgb.train(params, dtrain)

        # Convert the test data to a DMatrix
        test_features = xgb.DMatrix(test_features)

    else:
        raise KeyError

    print('method is ' + method)
    if save_model:
        # Save the model to a file
        with open('./checkpoints/'+method+'model.pkl', 'wb') as f:
            pickle.dump(mdl, f)
    # Use the forest's predict method on the test data
    predictions = mdl.predict(test_features)
    # for i in range(test_features.shape[0]):
    #     row = test_features[i]
    #     if row[0] == 44 and row[-3] == 0.239:
    #         print('hit')
    #         print(method,mdl.predict(row))

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', np.mean(errors), 'degrees.')

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
    print('Accuracy:', accuracy)

    # assuming pred is the predicted probability of positive class (class 1)
    auc = roc_auc_score(test_labels, pred)
    print('AUC:', auc)


def train_all(train_features, train_labels, test_features, test_labels):
    for method in methods:
        train(train_features, train_labels, test_features, test_labels, method)

# random forest 0.65
# LogisticRegression 0.64
# DecisionTree 0.6
# Naive Bayes 0.63
# svm 0.63
# gradient boosting classifier 0.65
# MLP 0.65
