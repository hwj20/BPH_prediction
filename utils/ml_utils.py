import pickle
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

methods = ['RandomForest', 'LogisticRegression', 'DecisionTree', 'GaussianNB', 'SVM', 'MLP', 'GBC', 'XGBoost']

save_features = True


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
        with open('./checkpoints/' + method + 'model.pkl', 'wb') as f:
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

    # Compute f1score
    f1score = f1_score(test_labels, predictions.round())
    print('f1score:', f1score)
    # Compute precision, recall, and accuracy
    precision = metrics.precision_score(test_labels, predictions.round())
    recall = metrics.recall_score(test_labels, predictions.round())
    accuracy = metrics.accuracy_score(test_labels, predictions.round())
    # Compute specificity
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions.round()).ravel()
    specificity = tn / (tn + fp)
    print('specificity:', specificity)

    # Plot confusion matrix as a heat map
    # cm = confusion_matrix(test_labels, predictions.round())
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    # ax.yaxis.set_ticklabels(['Negative', 'Positive'])

    # Display the heat map
    plt.show()
    # threshold
    pred = []
    for val in predictions:
        if val > 0.5:
            pred.append(1)
        else:
            pred.append(0)

    # Calculate and display accuracy
    accuracy = accuracy_score(test_labels, pred)
    print('Accuracy:', accuracy)

    # assuming pred is the predicted probability of positive class (class 1)
    auc = roc_auc_score(test_labels, pred)
    print('AUC:', auc)

    # Add metrics to a dictionary
    metrics_dict = {'f1-score': f1score, 'precision': precision, 'recall': recall, 'accuracy': accuracy,
                    'specificity': specificity}
    # Return the metrics dictionary
    return metrics_dict


def train_all(train_features, train_labels, test_features, test_labels):
    # Create an empty dictionary to store the metrics for each model
    results_dict = {}
    # Loop over the methods and train each model
    for method in methods:
        metrics_dict = train(train_features, train_labels, test_features, test_labels, method)
        # Add the metrics dictionary to the results dictionary
        if method == 'GBC':
            method = 'DRALF'
        results_dict[method] = metrics_dict

    print(results_dict)
    # save_features(train_features, train_labels, test_features, test_labels)
    # Create a pandas DataFrame from the results dictionary
    import pandas as pd
    df = pd.DataFrame.from_dict(results_dict, orient='index')

    # Plot the metrics as a grouped bar chart using seaborn
    # sns.set_style("whitegrid")
    # ax = df.plot(kind='bar', rot=0, figsize=(10, 6))
    # ax.set_title('Comparison of Model Performance')
    # ax.set_xlabel('Model')
    # ax.set_ylabel('Score')
    # ax.set_ylim([0.5, 1.0])
    # ax.set_yticks(np.arange(0.5, 1.01, 0.25))  # set y-axis ticks from 0.5 to 1.0 with step 0.25
    #
    # # Move the legend to the top-right corner
    # # ax.legend(loc='upper right')
    # ax.legend(loc=(1.01, 1))
    # # ax.legend(loc=(1.05, 1), bbox_to_anchor=(1.1, 1))

    # Plot the metrics as a grouped bar chart using seaborn
    sns.set_style("whitegrid")
    colors = ['#5cb85c', '#5bc0de', '#d9534f', '#9b59b6', '#34495e']  # 自定义颜色
    ax = df.plot(kind='bar', rot=0, figsize=(10, 15), subplots=True, layout=(5, 1), sharex=True, color=colors)
    ax[0][0].set_title('Comparison of Model Performance', fontsize=18, fontweight='bold')
    ax[4][0].set_xlabel('Model', fontsize=14)
    ax[4][0].tick_params(labelsize=12)  # 设置x轴标签字体大小
    # ax.set_ylim([0.5, 1.0])
    # ax.set_yticks(np.arange(0.5, 1.01, 0.25))  # set y-axis ticks from 0.5 to 1.0 with step 0.25

    # 设置y轴刻度尺
    yticks = [0.5, 0.75, 1.0]
    for i in range(5):
        ax[i][0].set_ylim([0.5, 1])  # 设置每个子图的y轴范围
        ax[i][0].set_yticks(yticks)  # 设置y轴刻度尺
        ax[i][0].set_yticklabels(yticks, fontsize=12)  # 设置y轴刻度尺标签的字体大小
        ax[i][0].legend(loc=(1.01, 1))

    plt.tight_layout()  # 收紧图像布局

    plt.show()


def save_features(train_features, train_labels, test_features, test_labels):
    # 保存决策树特征
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(train_features, train_labels)
    train_dtc_features = dtc.predict_proba(train_features)
    test_dtc_features = dtc.predict_proba(test_features)

    # 保存svm特征
    svm = SVC(random_state=42, probability=True)
    svm.fit(train_features, train_labels)
    train_svm_features = svm.predict_proba(train_features)
    test_svm_features = svm.predict_proba(test_features)

    # 将新特征与原特征进行拼接
    train_new_features = np.hstack((train_features, train_dtc_features, train_svm_features))
    test_new_features = np.hstack((test_features, test_dtc_features, test_svm_features))

    if save_features:
        # Save the extracted features to disk
        with open('./checkpoints/train_features_dtc.pkl', 'wb') as f:
            pickle.dump(train_dtc_features, f)
        with open('./checkpoints/train_features_svm.pkl', 'wb') as f:
            pickle.dump(train_svm_features, f)
        with open('./checkpoints/test_features_dtc.pkl', 'wb') as f:
            pickle.dump(test_dtc_features, f)
        with open('./checkpoints/test_features_svm.pkl', 'wb') as f:
            pickle.dump(test_svm_features, f)
    return train_new_features, train_labels, test_new_features, test_labels
