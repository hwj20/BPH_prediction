import pickle
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

methods = ['RandomForest', 'LogisticRegression', 'GaussianNB', 'SVM', 'MLP', 'GBC', 'XGBoost', 'Ensemble']

save_features = True
draw_comparison = False
draw_AUC = True


def train(train_features, train_labels, test_features, test_labels, method='GBC', save_model=True):
    """
    :param train_features:
    :param train_labels:
    :param test_features:
    :param test_labels:
    :param method: selected method
    :param save_model: if True, save the save model into .pkl file
    :return:
    """
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
        mdl = MLPClassifier(random_state=42, max_iter=500).fit(train_features, train_labels)

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
    elif method == 'Ensemble':
        metric_dict, predictions = train_mean(train_features, train_labels, test_features, test_labels)
        return metric_dict, predictions
    else:
        raise KeyError

    print('=' * 10 + 'Method is ' + method)
    if save_model:
        # Save the model to a file
        with open('./checkpoints/' + method + 'model.pkl', 'wb') as f:
            pickle.dump(mdl, f)
    # Use the forest's predict method on the test data
    predictions = mdl.predict(test_features)

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

    # threshold
    pred = []
    for val in predictions:
        if val > 0.5:
            pred.append(1)
        else:
            pred.append(0)

    """
    # Plot confusion matrix as a heat map
    # cm = confusion_matrix(test_labels, pred)
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    # ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    # Display the heat map
    # plt.show()
    """

    # Calculate and display accuracy
    accuracy = accuracy_score(test_labels, pred)
    print('Accuracy:', accuracy)

    # assuming pred is the predicted probability of positive class (class 1)
    auc_score = roc_auc_score(test_labels, pred)
    print('AUC:', auc_score)

    # Add metrics to a dictionary
    metrics_dict = {'f1-score': f1score, 'precision': precision, 'recall': recall, 'accuracy': accuracy,
                    'specificity': specificity}
    # Return the metrics dictionary
    return metrics_dict, predictions


def train_mean(train_features, train_labels, test_features, test_labels):
    """
    test the method of mean result of different models
    :param train_features:
    :param train_labels:
    :param test_features:
    :param test_labels:
    :return:
    """
    # Create an empty dictionary to store the metrics for each model
    results_dict = np.zeros([len(test_labels)])
    count = 0
    # Loop over the methods and train each model
    for method in methods:
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

            mdl = MLPClassifier(random_state=42, max_iter=500).fit(train_features, train_labels)

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

        elif method == 'Ensemble':
            continue
        else:
            raise KeyError

        predictions = mdl.predict(test_features)
        results_dict += predictions
        count += 1

    print('=' * 10 + 'Method is Mean')
    predictions = results_dict / count
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

    # threshold
    pred = []
    for val in predictions:
        if val > 0.5:
            pred.append(1)
        else:
            pred.append(0)

    """
    # Plot confusion matrix as a heat map
    # cm = confusion_matrix(test_labels, pred)
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    # ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    # Display the heat map
    # plt.show()
    """

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
    print(metrics_dict)

    return metrics_dict, predictions


def train_all(train_features, train_labels, test_features, test_labels):
    # Create an empty dictionary to store the metrics for each model
    results_dict = {}
    prediction_dict = {}

    # if presented the mean method, the result is as follows
    # results_dict['Ensemble'] = {'f1-score': 0.9356223175965666, 'precision': 0.9478260869565217,
    #                             'recall': 0.923728813559322, 'accuracy': 0.9318181818181818,
    #                             'specificity': 0.9411764705882353}

    # Loop over the methods and train each model
    for method in methods:
        metrics_dict, prediction = train(train_features, train_labels, test_features, test_labels, method)
        results_dict[method] = metrics_dict
        prediction_dict[method] = prediction

    print(results_dict)
    if draw_comparison:
        # save_features(train_features, train_labels, test_features, test_labels)
        # Create a pandas DataFrame from the results dictionary
        import pandas as pd
        df = pd.DataFrame.from_dict(results_dict, orient='index')

        # Plot the metrics as a grouped bar chart using seaborn
        sns.set_style("whitegrid")
        ax = df.plot(kind='bar', rot=0, figsize=(10, 6))
        ax.set_title('Comparison of Model Performance')
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_ylim([0.5, 1.0])
        ax.set_yticks(np.arange(0.5, 1.01, 0.25))  # set y-axis ticks from 0.5 to 1.0 with step 0.25

        # Move the legend to the top-right corner
        # ax.legend(loc='upper right')
        ax.legend(loc=(1.01, 1))

        sns.set_style("whitegrid")
        colors = ['#5cb85c', '#5bc0de', '#d9534f', '#9b59b6', '#34495e']  # 自定义颜色
        ax = df.plot(kind='bar', rot=0, figsize=(10, 15), subplots=True, layout=(5, 1), sharex=True, color=colors)
        # ax[0][0].set_title('Comparison of Model Performance', fontsize=18, fontweight='bold')
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

    if draw_AUC:
        num_methods = len(prediction_dict)
        rows = (num_methods + 3) // 4  # Calculate the number of rows needed

        fig, axs = plt.subplots(rows, 4, figsize=(15, 5 * rows))  # Adjust figsize as needed

        for idx, (method_name, prediction) in enumerate(prediction_dict.items()):
            row = idx // 4
            col = idx % 4
            fpr, tpr, thresholds = roc_curve(test_labels, prediction)
            # roc_auc = auc(fpr, tpr)
            roc_auc = auc(fpr, tpr)

            ax = axs[row, col]
            ax.plot(fpr, tpr, lw=2, label=f'{method_name} (AUC = %0.2f)' % roc_auc)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC of {method_name}')
            ax.legend(loc="lower right")

        plt.tight_layout()  # Adjust layout spacing
        plt.show()


# UNUSED method to save the features from DT and SVM models
def save_features(train_features, train_labels, test_features, test_labels):
    # save features from DT
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(train_features, train_labels)
    train_dtc_features = dtc.predict_proba(train_features)
    test_dtc_features = dtc.predict_proba(test_features)

    # save features from SVM
    svm = SVC(random_state=42, probability=True)
    svm.fit(train_features, train_labels)
    train_svm_features = svm.predict_proba(train_features)
    test_svm_features = svm.predict_proba(test_features)

    # concat features
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
