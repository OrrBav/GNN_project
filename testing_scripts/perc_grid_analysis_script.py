import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


colon_path = "/home/dsi/orrbavly/GNN_project/embeddings/colon_percentiles/TRB/percentiles_results_cos_3_all.json"
EPOCHS = 500

def run_kmeans(data, labels, sample_names):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    # 3. Map clusters back to sample names for analysis
    clustered_data = {
        "Sample Name": sample_names,
        "Cluster": clusters,
        "Label": labels,  # Optional, if labels are meaningful for evaluation
    }
    # 4. Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # 5. Visualization of clusters
    plt.figure(figsize=(8, 6))
    for cluster_id in range(n_clusters):
        indices = clusters == cluster_id
        plt.scatter(
            data_pca[indices, 0],
            data_pca[indices, 1],
            label=f'Cluster {cluster_id}',
            alpha=0.7
        )

    plt.title('K-Means Clustering Visualization (2D PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid()
    plt.show()

def load_json_and_extract_features_as_lists(json_path, stack=False):
    """Load JSON file, extract statistical features, and return structured lists of lists for each group."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    group_0 = []
    group_1 = []
    
    for sample_name, percentiles_dict in data.items():
        # Convert the JSON data structure to a list of lists for each sample
        sample_data = [[int(k)] + v for k, v in percentiles_dict.items()]  # Convert each k-value dictionary entry to a list

        # Extract statistical features for this sample
        features_df = extract_statistical_features_from_percentiles([sample_data], list(percentiles_dict.keys()), stack)

        # Initialize lists for each feature across all k values
        iqr_values = ['iqr']
        lower_tail_range_values = ['lower_tail_range']
        upper_tail_range_values = ['upper_tail_range']
        lower_tail_ratio_values = ['lower_tail_ratio']
        upper_tail_ratio_values = ['upper_tail_ratio']
        percentile_90_10_diff_values = ['percentile_90_10_diff']
        percentile_95_5_diff_values = ['percentile_95_5_diff']
        
        # Populate each feature list with values from all k's in the original JSON order
        for col in features_df.columns[1:]:  # Skip 'Sample' column
            if 'iqr' in col:
                iqr_values.append(features_df[col].iloc[0])
            elif 'lower_tail_range' in col:
                lower_tail_range_values.append(features_df[col].iloc[0])
            elif 'upper_tail_range' in col:
                upper_tail_range_values.append(features_df[col].iloc[0])
            elif 'lower_tail_ratio' in col:
                lower_tail_ratio_values.append(features_df[col].iloc[0])
            elif 'upper_tail_ratio' in col:
                upper_tail_ratio_values.append(features_df[col].iloc[0])
            elif 'percentile_90_10_diff' in col:
                percentile_90_10_diff_values.append(features_df[col].iloc[0])
            elif 'percentile_95_5_diff' in col:
                percentile_95_5_diff_values.append(features_df[col].iloc[0])

        # Combine all feature lists for this sample
        sample_features = {
            "sample_name": sample_name,
            "features": {
                "iqr": iqr_values,
                "lower_tail_range": lower_tail_range_values,
                "upper_tail_range": upper_tail_range_values,
                "lower_tail_ratio": lower_tail_ratio_values,
                "upper_tail_ratio": upper_tail_ratio_values,
                "percentile_90_10_diff": percentile_90_10_diff_values,
                "percentile_95_5_diff": percentile_95_5_diff_values,
            }
        }

        # Append to the appropriate group based on sample_name
        if 'H' in sample_name or 'low' in sample_name:
            group_0.append(sample_features)
        elif 'OC' in sample_name or 'high' in sample_name:
            group_1.append(sample_features)
    
    return group_0, group_1

    
def extract_statistical_features_from_percentiles(samples, k_labels, stack=False):
    """Extract statistical metrics from specified percentiles for each k value in each sample.
    
    Parameters:
    - samples: List of samples, where each sample is a list of lists.
               Each inner list has the format [k_value, p5, p15, p25, p35, p50, p70, p80, p90, p95].
    - k_labels: List of labels for each k value (e.g., [5, 10, 15, 20, 'sqrt', 'sqrt/2', 'log']).
    - stack: if another percentile vector for the same k value was concatanated(for example: PCA values)
    Returns:
    - DataFrame of statistical features for each sample and k value.
    """
    feature_list = []

    for sample_idx, sample in enumerate(samples):
        sample_features = {}
        sample_features['Sample'] = sample_idx
        
        for k_index, k_label in enumerate(k_labels):
            # Extract the provided percentiles for this k value
            percentiles = sample[k_index][1:]  # Exclude the k_value itself, just get percentiles
            
            if stack:
                perc = []
                half_size = len(percentiles) // 2
                for i in range (half_size):
                    value = 0.7 * percentiles[i] + 0.3 * percentiles[half_size + i]
                    perc.append(value)
                p5, p15, p25, p35, p50, p70, p80, p90, p95 = perc
            else:
                # Map the percentiles to variables for clarity
                p5, p15, p25, p35, p50, p70, p80, p90, p95 = percentiles

            # Calculate key features
            iqr = p70 - p25  # Interquartile Range (estimated as 75-25 range)
            lower_tail_range = p25 - p5  # Approximate lower tail range as 25-5
            upper_tail_range = p95 - p70  # Approximate upper tail range as 95-75
            
            # Calculate ratios
            lower_tail_ratio = lower_tail_range / iqr if iqr != 0 else 0
            upper_tail_ratio = upper_tail_range / iqr if iqr != 0 else 0
            
            # Additional quantile differences
            percentile_90_10_diff = p90 - p15  # 90th - 10th percentile difference
            percentile_95_5_diff = p95 - p5    # 95th - 5th percentile difference

            # Store features in the dictionary with keys indicating k value and metric
            sample_features[f'k={k_label}_iqr'] = iqr
            sample_features[f'k={k_label}_lower_tail_range'] = lower_tail_range
            sample_features[f'k={k_label}_upper_tail_range'] = upper_tail_range
            sample_features[f'k={k_label}_lower_tail_ratio'] = lower_tail_ratio
            sample_features[f'k={k_label}_upper_tail_ratio'] = upper_tail_ratio
            sample_features[f'k={k_label}_percentile_90_10_diff'] = percentile_90_10_diff
            sample_features[f'k={k_label}_percentile_95_5_diff'] = percentile_95_5_diff
        
        # Add this sample's features to the list
        feature_list.append(sample_features)
    
    # Convert the list of feature dictionaries to a DataFrame
    feature_df = pd.DataFrame(feature_list)
    return feature_df


def run_grid_search(data, labels, sample_names, epochs=15):
    """
    Run Random Forest Grid Search with repeated train-test splits.

    Parameters:
    - data: Processed feature data.
    - labels: Corresponding labels.
    - sample_names: List of sample names.
    - epochs: Number of iterations for train-test splits.

    Returns:
    - stats_df: Summary DataFrame with performance metrics.
    """
    # Initialize dictionaries and lists for tracking statistics
    false_negative_counts = {}
    test_group_counts = {}
    balanced_accuracy_scores = {}
    f1_scores = {}
    stats_list = []

    for i in range(epochs):
        print(f"\n####### round {i + 1} #######")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test, train_names, test_names = train_test_split(
            data, labels, sample_names, test_size=0.2, stratify=labels
        )

        # Standardize the feature values
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and evaluate the model
        train_grid_xgboost(X_train, X_test, y_train, y_test, test_names, 
                              false_negative_counts, test_group_counts, 
                              balanced_accuracy_scores, f1_scores)

    # Create a summary DataFrame
    for sample_name in test_group_counts.keys():
        balanced_acc_variance = pd.Series(balanced_accuracy_scores.get(sample_name, [])).var() if sample_name in balanced_accuracy_scores else 0
        stats_list.append({
            "sample_name": sample_name,
            "false_negative_appearances": false_negative_counts.get(sample_name, 0),
            "total_test_appearances": test_group_counts[sample_name],
            "average_balanced_accuracy": sum(balanced_accuracy_scores.get(sample_name, [])) / len(balanced_accuracy_scores.get(sample_name, [])),
            "average_f1_score": sum(f1_scores.get(sample_name, [])) / len(f1_scores.get(sample_name, [])),
            "balanced_accuracy_variance": balanced_acc_variance
        })

    # Convert the list to a DataFrame
    stats_df = pd.DataFrame(stats_list)
    stats_df['appearance_rate'] = stats_df['false_negative_appearances'] / stats_df['total_test_appearances']

    print("Finished running RF")

    return stats_df


def train_grid_xgboost(X_train, X_test, y_train, y_test, test_names, 
                          false_negative_counts, test_group_counts, 
                          balanced_accuracy_scores, f1_scores):
    # Define the parameter grid, including scale_pos_weight
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'scale_pos_weight': [1.5, 1.7, 1.8, 2.0]  # Adjusted based on class imbalance for colon (90/49)
    }

    # Initialize the XGBClassifier
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Initialize GridSearchCV, focusing on recall for label 1
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='balanced_accuracy', cv=5, n_jobs=3, verbose=0)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Print the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best balanced_accuracy Score from Grid Search:", grid_search.best_score_)

    # Predict using the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate metrics
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Identify false negatives (True label = 1, Predicted label = 0)
    false_negative_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true == 1 and pred == 0]
    false_negative_names = [test_names[i] for i in false_negative_indices]

    # Update the false negative counts
    for name in false_negative_names:
        false_negative_counts[name] = false_negative_counts.get(name, 0) + 1

    # Update the test group counts and track metrics for "high" samples
    for name in test_names:
        if "high" in name:
            test_group_counts[name] = test_group_counts.get(name, 0) + 1
            # Track balanced accuracy
            if name not in balanced_accuracy_scores:
                balanced_accuracy_scores[name] = []
            balanced_accuracy_scores[name].append(balanced_acc)

            # Track F1 scores
            if name not in f1_scores:
                f1_scores[name] = []
            f1_scores[name].append(f1)



def train_grid_rf_samples(X_train, X_test, y_train, y_test, test_names, 
                          false_negative_counts, test_group_counts, 
                          balanced_accuracy_scores, f1_scores):
    """
    Train Random Forest with GridSearchCV and update tracking metrics.
    """
    # Create a pipeline with scaling and Random Forest
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('rf', RandomForestClassifier(class_weight='balanced'))  # Random Forest classifier
    ])

    # Define the parameter grid
    param_grid = {
        'rf__n_estimators': [20, 50, 100, 200, 500],
        'rf__max_depth': [5, 10, 15, 20],
        'rf__min_samples_split': [2, 5, 10, 15],
        'rf__min_samples_leaf': [1, 2, 4, 8],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__bootstrap': [True, False]
    }

    # Initialize GridSearchCV with the pipeline and parameter grid
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1, verbose=0)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Balanced Accuracy Score from Grid Search:", grid_search.best_score_)

    # Predict using the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate metrics
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Identify false negatives (True label = 1, Predicted label = 0)
    false_negative_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true == 1 and pred == 0]
    false_negative_names = [test_names[i] for i in false_negative_indices]

    # Update the false negative counts
    for name in false_negative_names:
        false_negative_counts[name] = false_negative_counts.get(name, 0) + 1

    # Update the test group counts and track metrics for "high" samples
    for name in test_names:
        if "high" in name:
            test_group_counts[name] = test_group_counts.get(name, 0) + 1
            # Track balanced accuracy
            if name not in balanced_accuracy_scores:
                balanced_accuracy_scores[name] = []
            balanced_accuracy_scores[name].append(balanced_acc)

            # Track F1 scores
            if name not in f1_scores:
                f1_scores[name] = []
            f1_scores[name].append(f1)

def prepare_og_perc(percentiles_data, labels_dict, vector_indices=None, average_vectors=False, epochs=15):
    """
    Prepare data for original percentile data (not stats) and run Random Forest Grid Search.

    Parameters:
    - percentiles_data: Dictionary containing percentile data for each sample.
    - labels_dict: Dictionary mapping sample names to labels.
    - vector_indices: Optional list of indices specifying which features to include.
    - average_vectors: If True, averages the vectors across k values instead of concatenating.
    - epochs: Number of train-test splits for the grid search.

    Returns:
    - stats_df: DataFrame summarizing the Random Forest results.
    """
    data = []
    labels = []
    sample_names = []  # To store sample names

    # Data preparation logic
    if average_vectors:
        for sample_name, percentiles_dict in percentiles_data.items():
            vectors = np.array(list(percentiles_dict.values()))
            avg_vector = np.mean(vectors, axis=0)
            data.append(avg_vector)
            labels.append(labels_dict[sample_name])
            sample_names.append(sample_name)  # Keep track of sample name
    else:
        max_length = max(len(np.concatenate(list(percentiles_dict.values()))) for percentiles_dict in percentiles_data.values())
        for sample_name, percentiles_dict in percentiles_data.items():
            vectors = list(percentiles_dict.values())
            if vector_indices is not None:
                selected_vectors = [vectors[i] for i in vector_indices if i < len(vectors)]
                flattened_percentiles = np.concatenate(selected_vectors)
            else:
                flattened_percentiles = np.concatenate(vectors)
            padded_percentiles = np.pad(flattened_percentiles, (0, max_length - len(flattened_percentiles)), 'constant')
            data.append(padded_percentiles)
            labels.append(labels_dict[sample_name])
            sample_names.append(sample_name)  # Keep track of sample name

    # Convert data and labels to arrays
    data = np.array(data)
    labels = np.array(labels)
    sample_names = np.array(sample_names)

    # Run Random Forest Grid Search
    stats_df = run_grid_search(data, labels, sample_names, epochs=epochs)
    print("Finished running RF with original percentiles data.")
    return stats_df



def prepare_data_from_groups(group_0, group_1, average_vectors=False, vector_indices=None):
    """
    Used for Stata data.
    Prepare data for machine learning algorithms using extracted features from two groups.

    Parameters:
    - group_0: List of lists for Group 0, where each inner list represents a patient.
    - group_1: List of lists for Group 1, where each inner list represents a patient.
    - average_vectors: If True, averages the vectors across k values instead of concatenating.
    - vector_indices: Optional list of indices specifying which features to include.

    Returns:
    - data: Processed feature data.
    - labels: Corresponding labels.
    - sample_names: List of sample names (indices for now).
    """
    data = []
    labels = []
    sample_names = []  # To store sample names

    # Define the maximum length for padding
    max_length = 0

    # Combine groups and assign labels
    combined_data = group_0 + group_1
    combined_labels = [0] * len(group_0) + [1] * len(group_1)


    # Find the maximum length needed for padding based on selected indices
    if not average_vectors:
        if vector_indices is not None:
            max_length = max(
                len(np.concatenate([patient_features["features"][i][1:] for i in vector_indices if i < len(patient_features["features"])]))
                for patient_features in combined_data
            )
        else:
            max_length = max(
                len(np.concatenate([feature[1:] for feature in patient_features["features"].values()]))
                for patient_features in combined_data
            )

    for patient_features, label in zip(combined_data, combined_labels):
        # Extract sample name and feature values
        sample_name = patient_features["sample_name"]
        vectors = [feature[1:] for feature in patient_features["features"].values()]

        if average_vectors:
            # Calculate the mean across all k values for each feature
            avg_vector = np.mean(np.array(vectors), axis=0)
            data.append(avg_vector)
        else:
            if vector_indices is not None:
                selected_vectors = [vectors[i] for i in vector_indices if i < len(vectors)]
                flattened_features = np.concatenate(selected_vectors)
            else:
                flattened_features = np.concatenate(vectors)

            # Pad the concatenated features to match the max length
            padded_features = np.pad(flattened_features, (0, max_length - len(flattened_features)), 'constant')
            data.append(padded_features)

        labels.append(label)
        sample_names.append(sample_name)  # Retain original sample name

    # Convert data and labels to arrays
    data = np.array(data)
    labels = np.array(labels)
    sample_names = np.array(sample_names)

    run_kmeans(data, labels, sample_names)
    
    # Run Random Forest Grid Search
    stats_df = run_grid_search(data, labels, sample_names, epochs=EPOCHS)

    print("Finished running Algorithm with original percentiles data.")
    return stats_df


def load_results(file_path, data_type):
    # load json file
    with open(file_path, 'r') as f:
        all_results = json.load(f)
    # create labels dictionary
    labels_dict = {}
    for sample_name, percentile_dict in all_results.items():
        if data_type == 'ovarian':
            if sample_name.endswith("_H"):
                labels_dict[sample_name] = 0
            elif sample_name.endswith("_OC"):
                labels_dict[sample_name] = 1
            else:
                raise Exception("Error - invalid sample type")
        elif data_type == 'colon':
            if sample_name.endswith("_low"):
                labels_dict[sample_name] = 0
            elif sample_name.endswith("_high"):
                labels_dict[sample_name] = 1
            else:
                raise Exception("Error - invalid sample type")
    return all_results, labels_dict

if __name__ == '__main__':
    start = time.time()
    all_results, labels_dict = load_results(colon_path, data_type='colon')
    ## OG data
    # stats_df = test_2(all_results, labels_dict)
     
    ## OUTLINES DATA
    samples_group_0, samples_group_1 = load_json_and_extract_features_as_lists(colon_path)
    stats_df = prepare_data_from_groups(samples_group_0, samples_group_1)
    print("Saving to file...")
    stats_df.to_csv("/home/dsi/orrbavly/GNN_project/outputs/colon_samples_stats_outlines_xgboost.csv")
    end=time.time()
    print(f"Runtime: {(end - start)/60}")