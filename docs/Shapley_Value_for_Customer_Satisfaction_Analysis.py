import pandas as pd
import numpy as np
import itertools
import math

def preprocess_data(filepath, overall_satisfaction_col, feature_cols, 
                    dissatisfaction_threshold, failure_threshold_map,
                    overall_score_higher_is_better=True, 
                    feature_score_higher_is_better=True):
    """
    Loads and preprocesses the data.

    Args:
        filepath (str): Path to the CSV file.
        overall_satisfaction_col (str): Name of the overall satisfaction column.
        feature_cols (list): List of feature column names.
        dissatisfaction_threshold (float): Threshold for overall satisfaction.
                                           Scores <= threshold are "Dissatisfied" if higher_is_better=True,
                                           Scores >= threshold are "Dissatisfied" if higher_is_better=False.
        failure_threshold_map (dict): A dictionary mapping feature_col_name to its failure threshold.
                                      Scores <= threshold are "Failed" if higher_is_better=True for that feature,
                                      Scores >= threshold are "Failed" if higher_is_better=False for that feature.
        overall_score_higher_is_better (bool): True if higher overall scores mean more satisfaction.
        feature_score_higher_is_better (bool): True if higher feature scores mean better performance.
                                               Can be overridden per feature in failure_threshold_map if needed
                                               by making threshold_map values tuples: (threshold, higher_is_better_for_feature)

    Returns:
        pd.DataFrame: Processed DataFrame with binarized columns.
        str: Name of the binarized overall satisfaction column.
        list: List of binarized feature column names.
    """
    df = pd.read_csv(filepath)

    # Binarize overall satisfaction
    binarized_overall_col = f"{overall_satisfaction_col}_Dissatisfied"
    if overall_score_higher_is_better:
        df[binarized_overall_col] = (df[overall_satisfaction_col] <= dissatisfaction_threshold).astype(int)
    else:
        df[binarized_overall_col] = (df[overall_satisfaction_col] >= dissatisfaction_threshold).astype(int)

    # Binarize feature columns
    binarized_feature_cols = []
    for col in feature_cols:
        bin_col_name = f"{col}_Failed"
        threshold_info = failure_threshold_map.get(col)
        
        current_feature_higher_is_better = feature_score_higher_is_better # Default
        current_failure_threshold = None

        if isinstance(threshold_info, tuple): # (threshold, specific_higher_is_better)
            current_failure_threshold = threshold_info[0]
            current_feature_higher_is_better = threshold_info[1]
        elif threshold_info is not None: # Just threshold, use global feature_score_higher_is_better
            current_failure_threshold = threshold_info
        else:
            raise ValueError(f"Failure threshold not defined for feature: {col}")

        if current_feature_higher_is_better:
            df[bin_col_name] = (df[col] <= current_failure_threshold).astype(int)
        else:
            df[bin_col_name] = (df[col] >= current_failure_threshold).astype(int)
        binarized_feature_cols.append(bin_col_name)
        
    return df, binarized_overall_col, binarized_feature_cols

def get_value_of_coalition(df, coalition_bin_feature_cols, binarized_overall_col):
    """
    Calculates the value v(M) = Reach_M - Noise_M for a given coalition M.
    M is represented by a list of binarized feature column names.
    """
    if not coalition_bin_feature_cols: # Empty coalition
        return 0.0

    # Mask for rows where at least one feature in the coalition has "Failed" (is 1)
    failed_on_any_in_coalition_mask = df[coalition_bin_feature_cols].any(axis=1)
    
    df_dissatisfied = df[df[binarized_overall_col] == 1]
    df_not_dissatisfied = df[df[binarized_overall_col] == 0]

    num_total_dissatisfied = len(df_dissatisfied)
    num_total_not_dissatisfied = len(df_not_dissatisfied)

    if num_total_dissatisfied == 0 and num_total_not_dissatisfied == 0:
        # This case should ideally not happen with reasonable data.
        # If it does, it means no data points, so value is undefined or 0.
        print("Warning: No data points found for calculating coalition value.")
        return 0.0
    
    # Reach_M = P(Failed_on_any_in_M | Dissatisfied)
    # = Count(Failed_on_any_in_M AND Dissatisfied) / Count(Dissatisfied)
    if num_total_dissatisfied > 0:
        num_failed_and_dissatisfied = df_dissatisfied[failed_on_any_in_coalition_mask[df_dissatisfied.index]].shape[0]
        reach_M = num_failed_and_dissatisfied / num_total_dissatisfied
    else:
        reach_M = 0.0 # No dissatisfied customers, so cannot be reached via failure.

    # Noise_M = P(Failed_on_any_in_M | Not Dissatisfied)
    # = Count(Failed_on_any_in_M AND Not Dissatisfied) / Count(Not Dissatisfied)
    if num_total_not_dissatisfied > 0:
        num_failed_and_not_dissatisfied = df_not_dissatisfied[failed_on_any_in_coalition_mask[df_not_dissatisfied.index]].shape[0]
        noise_M = num_failed_and_not_dissatisfied / num_total_not_dissatisfied
    else:
        noise_M = 0.0 # No not_dissatisfied customers.

    value_M = reach_M - noise_M
    return value_M

def calculate_shapley_values(df, binarized_feature_cols, binarized_overall_col):
    """
    Calculates Shapley values for each feature.
    """
    num_features = len(binarized_feature_cols)
    feature_indices = list(range(num_features))
    shapley_values = np.zeros(num_features)

    for i in feature_indices: # For each feature 'k' (represented by index i)
        feature_k_col_name = binarized_feature_cols[i]
        
        # Iterate over all possible coalitions M that DO NOT contain feature k
        remaining_feature_indices = [idx for idx in feature_indices if idx != i]
        
        for m_size in range(num_features): # m_size is the size of coalition M (from 0 to n-1)
            # gamma_n(M) = m! * (n - m - 1)! / n!
            # Here, n is num_features, m is m_size
            if num_features - m_size -1 < 0 : # Avoid factorial of negative
                 # This case happens when m_size = num_features, which means coalition M includes all other n-1 features.
                 # The marginal contribution is to the grand coalition.
                 # The loop for m_size should go up to num_features -1 if we consider M not containing k.
                 # If M is of size m, M U {k} is of size m+1.
                 # The sum is over coalitions M of size m, where m ranges from 0 to n-1.
                 # (n-m-1)! is (num_features - m_size -1)!
                 # This gamma is for a coalition M of size m_size.
                 pass # will be handled by itertools.combinations

            gamma_weight = (math.factorial(m_size) * math.factorial(num_features - m_size - 1)) / math.factorial(num_features)

            for M_indices_tuple in itertools.combinations(remaining_feature_indices, m_size):
                M_cols = [binarized_feature_cols[idx] for idx in M_indices_tuple]
                M_union_k_cols = M_cols + [feature_k_col_name]

                v_M_union_k = get_value_of_coalition(df, M_union_k_cols, binarized_overall_col)
                v_M = get_value_of_coalition(df, M_cols, binarized_overall_col)
                
                shapley_values[i] += gamma_weight * (v_M_union_k - v_M)
                
    return {binarized_feature_cols[i]: shapley_values[i] for i in range(num_features)}


def get_reach_noise_success_for_set(df, set_bin_feature_cols, binarized_overall_col):
    """Calculates Reach, Noise, and Success for a given set of (binarized) features."""
    if not set_bin_feature_cols:
        return 0.0, 0.0, 0.0

    failed_on_any_in_set_mask = df[set_bin_feature_cols].any(axis=1)
    
    df_dissatisfied = df[df[binarized_overall_col] == 1]
    df_not_dissatisfied = df[df[binarized_overall_col] == 0]

    num_total_dissatisfied = len(df_dissatisfied)
    num_total_not_dissatisfied = len(df_not_dissatisfied)

    if num_total_dissatisfied == 0 and num_total_not_dissatisfied == 0:
        return 0.0, 0.0, 0.0

    if num_total_dissatisfied > 0:
        num_failed_and_dissatisfied = df_dissatisfied[failed_on_any_in_set_mask[df_dissatisfied.index]].shape[0]
        reach = num_failed_and_dissatisfied / num_total_dissatisfied
    else:
        reach = 0.0

    if num_total_not_dissatisfied > 0:
        num_failed_and_not_dissatisfied = df_not_dissatisfied[failed_on_any_in_set_mask[df_not_dissatisfied.index]].shape[0]
        noise = num_failed_and_not_dissatisfied / num_total_not_dissatisfied
    else:
        noise = 0.0
        
    success = reach - noise
    return reach, noise, success


def determine_key_drivers(df, binarized_feature_cols, binarized_overall_col, shapley_values_dict):
    """
    Determines the key dissatisfiers based on Shapley values and the Success metric.
    """
    # Sort features by Shapley value in descending order
    sorted_features = sorted(shapley_values_dict.items(), key=lambda item: item[1], reverse=True)
    
    print("\n--- Determining Key Dissatisfiers (Cumulative Analysis) ---")
    print(f"{'Step':<5} {'Added Feature':<30} {'Cumulative Set Size':<20} {'Reach':<10} {'Noise':<10} {'Success':<10}")
    
    cumulative_set_cols = []
    optimal_set_cols = []
    max_success_achieved = -float('inf')
    
    results_log = []

    for i, (feature_col, sv) in enumerate(sorted_features):
        current_cumulative_cols_for_step = cumulative_set_cols + [feature_col]
        
        reach, noise, success = get_reach_noise_success_for_set(df, current_cumulative_cols_for_step, binarized_overall_col)
        
        results_log.append({
            'step': i + 1,
            'added_feature': feature_col.replace('_Failed', ''),
            'set_size': len(current_cumulative_cols_for_step),
            'reach': reach,
            'noise': noise,
            'success': success,
            'current_set_cols_for_step': list(current_cumulative_cols_for_step) # for debugging or internal use
        })
        
        print(f"{i+1:<5} {feature_col.replace('_Failed', ''):<30} {len(current_cumulative_cols_for_step):<20} {reach:<10.3f} {noise:<10.3f} {success:<10.3f}")

        if success >= max_success_achieved : # If current success is better or equal, update optimal set and continue
            max_success_achieved = success
            optimal_set_cols = list(current_cumulative_cols_for_step) # Keep this set as potentially optimal
            cumulative_set_cols.append(feature_col) # Add to the set for the next iteration
        else:
            # Success decreased, so the set from the *previous* step was optimal.
            # `optimal_set_cols` already holds the best set found so far that led to max_success_achieved.
            print(f"Success decreased. Optimal set identified before adding '{feature_col.replace('_Failed', '')}'.")
            break 
            
    # If the loop completed without success decreasing, the last cumulative set is optimal.
    # This is handled because optimal_set_cols is updated whenever success >= max_success_achieved.

    return [col.replace('_Failed', '') for col in optimal_set_cols], results_log


def main():
    """
    Main function to run the analysis.
    """
    # --- Configuration ---
    # Replace with your actual file path
    filepath = 'sample_customer_data.csv' 
    
    # Define overall satisfaction column and its dissatisfaction threshold
    overall_satisfaction_col = 'OverallSatisfaction' # Raw score column
    # Assuming 1-10 scale, higher is better. Scores <= 5 are "Dissatisfied".
    dissatisfaction_threshold = 5 
    overall_score_higher_is_better = True

    # Define feature columns and their failure thresholds
    # Format: { 'feature_col_name': threshold } or 
    #         { 'feature_col_name': (threshold, higher_is_better_for_this_feature) }
    feature_cols = ['FeatureA', 'FeatureB', 'FeatureC', 'FeatureD']
    failure_threshold_map = {
        'FeatureA': 3, # Assuming 1-5 scale, higher is better. Scores <=3 are "Failed".
        'FeatureB': 3,
        'FeatureC': (7, False), # Assuming 1-10 scale, but for this feature LOWER is better. Scores >=7 are "Failed".
        'FeatureD': 2
    }
    # Default assumption for feature scores (can be overridden in failure_threshold_map)
    feature_score_higher_is_better = True 

    # --- Create a dummy sample_customer_data.csv for testing ---
    data = {
        'OverallSatisfaction': [2, 8, 5, 10, 3, 6, 1, 9, 4, 7, 2, 5, 8, 3, 6, 10, 1, 4, 9, 7],
        'FeatureA':            [1, 5, 3,  4, 2, 5, 1, 4, 2, 3, 1, 3, 5, 2, 4, 5, 1, 2, 5, 3], # Scale 1-5, higher better
        'FeatureB':            [2, 4, 2,  5, 1, 3, 2, 5, 1, 4, 2, 2, 4, 1, 3, 5, 2, 1, 5, 4], # Scale 1-5, higher better
        'FeatureC':            [8, 3, 6,  2, 9, 4, 10,1, 7, 5, 8, 6, 2, 9, 4, 1, 10,7, 3, 5], # Scale 1-10, LOWER better
        'FeatureD':            [1, 3, 1,  3, 1, 2, 1, 3, 1, 2, 1, 1, 3, 1, 2, 3, 1, 1, 3, 2]  # Scale 1-3, higher better
    }
    df_sample = pd.DataFrame(data)
    df_sample.to_csv(filepath, index=False)
    print(f"Created dummy data at {filepath}")
    # --- End of dummy data creation ---

    # 1. Preprocess data
    df_processed, bin_overall_col, bin_feature_cols = preprocess_data(
        filepath,
        overall_satisfaction_col,
        feature_cols,
        dissatisfaction_threshold,
        failure_threshold_map,
        overall_score_higher_is_better,
        feature_score_higher_is_better
    )
    print("\n--- Processed Data Head ---")
    print(df_processed[[bin_overall_col] + bin_feature_cols].head())

    # Check if there are any dissatisfied or not dissatisfied customers
    if df_processed[bin_overall_col].nunique() < 2:
        print(f"\nWarning: The binarized overall satisfaction column '{bin_overall_col}' has only one unique value.")
        print("This will lead to Reach or Noise (or both) being undefined or zero for all coalitions.")
        print("Please check your dissatisfaction_threshold and data distribution.")
        # Depending on the case, you might want to exit or handle differently
        if df_processed[df_processed[bin_overall_col] == 1].empty:
            print("No customers are marked as 'Dissatisfied'.")
        if df_processed[df_processed[bin_overall_col] == 0].empty:
            print("No customers are marked as 'Not Dissatisfied'.")
        # return # Optionally exit if data is unsuitable

    # 2. Calculate Shapley values
    print("\nCalculating Shapley values... (this may take time for many features)")
    shapley_values = calculate_shapley_values(df_processed, bin_feature_cols, bin_overall_col)
    
    print("\n--- Shapley Values ---")
    for feature, sv in sorted(shapley_values.items(), key=lambda item: item[1], reverse=True):
        print(f"{feature.replace('_Failed', ''):<30}: {sv:.4f}")

    # 3. Determine Key Dissatisfiers
    key_dissatisfiers, full_log = determine_key_drivers(df_processed, bin_feature_cols, bin_overall_col, shapley_values)
    
    print("\n--- Final Set of Key Dissatisfiers ---")
    if key_dissatisfiers:
        for kd in key_dissatisfiers:
            print(f"- {kd}")
    else:
        print("No key dissatisfiers identified based on the criteria (or no features provided).")

if __name__ == '__main__':
    main()
