import pandas as pd
import itertools

def joint_distribution(df, variables=None):
    """
    This function computes the joint probability distribution of a set of binary variables.

    Args:
    - df: A pandas DataFrame containing binary variables.
    - variables: A list of binary variables to compute the joint distribution for. If None, all columns in the DataFrame are used.

    Returns:
    - joint_dist_df: A pandas DataFrame containing the joint probability distribution of the specified variables.
    """
    if variables is None:
        variables = df.columns.tolist()
    # Step 1: Filter the DataFrame to include only the specified variables
    filtered_df = df[variables]
    
    # Step 2: Create all possible combinations of the binary variables
    all_combinations = list(itertools.product([0, 1], repeat=len(variables)))
    all_combinations_df = pd.DataFrame(all_combinations, columns=variables)
    
    # Step 3: Group by the unique combinations of the specified variables and count occurrences
    joint_dist_df = filtered_df.groupby(variables).size().reset_index(name='count')
    
    # Step 4: Merge with all combinations to include zero probabilities
    joint_dist_df = pd.merge(all_combinations_df, joint_dist_df, on=variables, how='left')
    joint_dist_df['count'] = joint_dist_df['count'].fillna(0)
    
    # Step 5: Normalize the counts to get the joint probability distribution
    total_count = joint_dist_df['count'].sum()
    joint_dist_df['prob'] = joint_dist_df['count'] / total_count
    
    return joint_dist_df[variables + ['prob']]