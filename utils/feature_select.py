# encoding=utf-8
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

show_pearson = False
save_file = True
label_name = 'is_BPH'


def feature_analysis(feature_name, read_file):
    """
    analyse the statics value of a feature
    :param feature_name:
    :param read_file:
    :return:
    """
    # Select the feature and label columns
    feature = read_file[feature_name]
    label = read_file[label_name]

    # Calculate statistical features
    feature_stats = {
        'Mean': feature.mean(),
        'Standard Deviation': feature.std(),
        'Minimum': feature.min(),
        'Maximum': feature.max(),
        'Median': feature.median(),
        'Mode': feature.mode().values[0],
        'Count': feature.count(),
        'Missing Values': feature.isnull().sum()
    }

    # Calculate label statistics
    label_stats = {
        'Count of Positive Samples': label.sum(),
        'Count of Negative Samples': label.count() - label.sum()
    }

    # Print the feature statistics
    print("Feature:", feature_name)
    print("Label:", label_name)
    print("Feature Statistics:")
    for stat, value in feature_stats.items():
        print(f"{stat}: {value}")

    # Print the label statistics
    print("\nLabel Statistics:")
    for stat, value in label_stats.items():
        print(f"{stat}: {value}")

    # Example usage


def process_data(read_file):
    """
    drop some columns(missing too much) and fill missing value
    :param read_file:
    :return:
    """
    # Create a list of columns to be dropped
    cols_to_drop = ['GLU', 'ACR', '登记号', '病案号', '就诊ID', 'patient_unique_number', 'has_other_disease', 'PBG',
                    'FBG']

    # Drop the columns in one step
    read_file = read_file.drop(cols_to_drop, axis=1)

    if show_pearson:
        # Load the data into a pandas DataFrame
        df = read_file

        # Calculate the Pearson correlation matrix
        corr = df.corr(method='pearson')

        # Print the Pearson correlation matrix
        print(corr)
        # Plot the Pearson correlation matrix as a heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm')

        # Add the title to the plot
        plt.title("Pearson Correlation Heatmap")

        # Show the plot
        plt.show()
        input()

    # the following codes remove the columns with a high missing ratio
    read_file = read_file.astype({col: np.int8 for col in read_file.columns[read_file.dtypes == np.bool_]})
    # Define a threshold for the label ratio
    threshold = 5

    # Iterate over each column
    cols_to_drop = []
    for col in read_file.columns:
        if col == label_name:
            continue
        else:
            # Group the data by the target column
            grouped = read_file.groupby('is_BPH')

            # Calculate the ratio of each class in the column
            counts = grouped[col].count()

            # Check if the ratio of one class to the other is too high
            if counts[0] > threshold * counts[1] or counts[1] > threshold * counts[0]:
                # Drop the column if the ratio is too high
                cols_to_drop.append(col)
    print('Drop ', cols_to_drop)
    read_file = read_file.drop(cols_to_drop, axis=1)

    """
    # show the count of neg or pos of label
    # df = read_file
    # df = df.dropna()
    # print(df.count())
    # # Group the data by the target column
    # grouped = df.groupby('is_BPH')
    # print(grouped.count())
    """
    scaler = MinMaxScaler()
    num_cols = [col for col in read_file.columns if col != 'is_BPH' and np.issubdtype(read_file[col].dtype, np.number)]
    read_file[num_cols] = scaler.fit_transform(read_file[num_cols])

    read_file.fillna(read_file.mean(), inplace=True)
    if save_file:
        # Save the processed data to a new file
        read_file.to_csv("processed_data.csv", encoding='utf_8_sig', index=False)
    return read_file
    # print(read_file.describe())


def load_data(read_file):
    """
    drop the label of read_file
    :param read_file:
    :return:
    """
    features = read_file
    # Labels are the values we want to predict
    labels = np.array(features[label_name])
    # Remove the labels from the features
    features = features.drop(label_name, axis=1)
    return features, labels


def find_top_feature(read_file):
    # Load the data and split into features and labels
    X, y = load_data(read_file)

    # Calculate mutual information between features and labels
    mi_scores = mutual_info_classif(X, y)

    # Select top-k features based on mutual information
    k = 10
    top_k_features_mi = X.columns[mi_scores.argsort()[::-1][:k]]

    # Calculate F-values of features
    f_scores = f_classif(X, y)[0]

    # Select top-k features based on F-values
    top_k_features_f = X.columns[f_scores.argsort()[::-1][:k]]

    # Calculate Gini indices of features
    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    # Select top-k features based on Gini indices
    forest = ExtraTreesClassifier(n_estimators=250, random_state=42)
    forest.fit(X, y)
    gi_scores = forest.feature_importances_
    top_k_features_gi = X.columns[gi_scores.argsort()[::-1][:k]]

    print(top_k_features_gi)
    print(top_k_features_f)
    print(top_k_features_mi)

    res = []
    for label in top_k_features_f:
        if label in top_k_features_gi or label in top_k_features_mi:
            res.append(label)
    for label in top_k_features_gi:
        if label in top_k_features_mi and label not in res:
            res.append(label)
    print(res)

    return res
