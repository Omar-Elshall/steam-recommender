"""
Evaluation metrics for recommendation systems.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error
import pandas as pd

def precision_at_k(recommendations, relevant_items, k=10):
    """
    Calculate precision@k for a single user.
    
    Parameters:
    -----------
    recommendations : list
        List of recommended item IDs
    relevant_items : list
        List of relevant item IDs (ground truth)
    k : int
        Number of recommendations to consider
    
    Returns:
    --------
    precision : float
        Precision@k value
    """
    # Ensure we only consider the top k recommendations
    if len(recommendations) > k:
        recommendations = recommendations[:k]
    
    # Count number of relevant items in recommendations
    n_relevant = sum(1 for item in recommendations if item in relevant_items)
    
    # Return precision@k
    return n_relevant / len(recommendations) if recommendations else 0.0

def recall_at_k(recommendations, relevant_items, k=10):
    """
    Calculate recall@k for a single user.
    
    Parameters:
    -----------
    recommendations : list
        List of recommended item IDs
    relevant_items : list
        List of relevant item IDs (ground truth)
    k : int
        Number of recommendations to consider
    
    Returns:
    --------
    recall : float
        Recall@k value
    """
    # Ensure we only consider the top k recommendations
    if len(recommendations) > k:
        recommendations = recommendations[:k]
    
    # Count number of relevant items in recommendations
    n_relevant = sum(1 for item in recommendations if item in relevant_items)
    
    # Return recall@k
    return n_relevant / len(relevant_items) if relevant_items else 0.0

def average_precision(recommendations, relevant_items):
    """
    Calculate average precision for a single user.
    
    Parameters:
    -----------
    recommendations : list
        List of recommended item IDs
    relevant_items : list
        List of relevant item IDs (ground truth)
    
    Returns:
    --------
    ap : float
        Average precision value
    """
    hits = 0
    sum_prec = 0.0
    
    for i, item in enumerate(recommendations):
        if item in relevant_items:
            hits += 1
            sum_prec += hits / (i + 1)
    
    return sum_prec / len(relevant_items) if relevant_items else 0.0

def mean_average_precision(recommendations_dict, relevant_items_dict):
    """
    Calculate mean average precision (MAP) for all users.
    
    Parameters:
    -----------
    recommendations_dict : dict
        Dictionary mapping user_id to list of recommended item IDs
    relevant_items_dict : dict
        Dictionary mapping user_id to list of relevant item IDs
    
    Returns:
    --------
    map_score : float
        Mean average precision value
    """
    # Ensure we evaluate only users that have recommendations and ground truth
    common_users = set(recommendations_dict.keys()) & set(relevant_items_dict.keys())
    
    if not common_users:
        return 0.0
    
    # Calculate average precision for each user
    aps = [average_precision(recommendations_dict[user], relevant_items_dict[user]) 
           for user in common_users]
    
    # Return mean average precision
    return sum(aps) / len(aps)

def ndcg_at_k(recommendations, relevant_items, k=10, method="standard"):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Parameters:
    -----------
    recommendations : list
        List of recommended item IDs
    relevant_items : list
        List of relevant item IDs (ground truth)
    k : int
        Number of recommendations to consider
    method : str
        Method to compute relevance score ("standard" or "binary")
    
    Returns:
    --------
    ndcg : float
        NDCG@k value
    """
    # Ensure we only consider the top k recommendations
    if len(recommendations) > k:
        recommendations = recommendations[:k]
    
    # Initialize relevance scores list
    relevance = []
    
    # Compute relevance scores based on the method
    for item in recommendations:
        if method == "binary":
            # Binary relevance: 1 if item is relevant, 0 otherwise
            rel = 1 if item in relevant_items else 0
        else:
            # Standard relevance based on position in relevant items
            if item in relevant_items:
                # Higher relevance for items higher in the relevant list
                rel = 1.0 / (relevant_items.index(item) + 1)
            else:
                rel = 0
        
        relevance.append(rel)
    
    # Calculate DCG
    dcg = relevance[0] if relevance else 0
    for i in range(1, len(relevance)):
        # DCG formula: rel_i / log_2(i+1)
        dcg += relevance[i] / np.log2(i + 2)
    
    # Calculate ideal DCG
    if method == "binary":
        # For binary relevance, ideal ordering has all relevant items first
        ideal_relevance = [1] * min(len(relevant_items), k) + [0] * max(0, k - len(relevant_items))
    else:
        # For standard relevance, ideal ordering is by relevance score
        ideal_relevance = [1.0 / (i + 1) for i in range(min(len(relevant_items), k))]
    
    idcg = ideal_relevance[0] if ideal_relevance else 0
    for i in range(1, len(ideal_relevance)):
        idcg += ideal_relevance[i] / np.log2(i + 2)
    
    # Return NDCG
    return dcg / idcg if idcg > 0 else 0

def diversity(recommendations, item_features):
    """
    Calculate diversity of recommendations based on feature similarity.
    
    Parameters:
    -----------
    recommendations : list
        List of recommended item IDs
    item_features : dict
        Dictionary mapping item_id to feature vector
    
    Returns:
    --------
    diversity : float
        Diversity score (1 - average pairwise similarity)
    """
    # Get feature vectors for recommended items
    vectors = [item_features[item] for item in recommendations if item in item_features]
    
    if len(vectors) < 2:
        return 0.0
    
    # Calculate pairwise cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(vectors)
    
    # Remove self-similarities (diagonal)
    np.fill_diagonal(sim_matrix, 0)
    
    # Calculate average similarity
    avg_sim = sim_matrix.sum() / (len(vectors) * (len(vectors) - 1))
    
    # Diversity is 1 - average similarity
    return 1.0 - avg_sim

def coverage(recommendations_dict, all_items):
    """
    Calculate catalog coverage - percentage of items that are recommended to at least one user.
    
    Parameters:
    -----------
    recommendations_dict : dict
        Dictionary mapping user_id to list of recommended item IDs
    all_items : list
        List of all item IDs in the catalog
    
    Returns:
    --------
    coverage : float
        Coverage score (percentage of catalog covered)
    """
    # Get all unique items that have been recommended to at least one user
    recommended_items = set()
    for user, items in recommendations_dict.items():
        recommended_items.update(items)
    
    # Calculate coverage
    return len(recommended_items) / len(all_items) if all_items else 0.0

def novelty(recommendations, item_popularity):
    """
    Calculate novelty of recommendations based on inverse popularity.
    
    Parameters:
    -----------
    recommendations : list
        List of recommended item IDs
    item_popularity : dict
        Dictionary mapping item_id to popularity score
    
    Returns:
    --------
    novelty : float
        Novelty score (higher means less popular/more novel items)
    """
    if not recommendations:
        return 0.0
    
    # Calculate inverse popularity for each recommended item
    inv_pop_sum = sum(1.0 - item_popularity.get(item, 0) for item in recommendations)
    
    # Return average inverse popularity
    return inv_pop_sum / len(recommendations)

def serendipity(recommendations, expected_recommendations, relevant_items):
    """
    Calculate serendipity - the "unexpectedness" of relevant recommendations.
    
    Parameters:
    -----------
    recommendations : list
        List of actual recommended item IDs
    expected_recommendations : list
        List of expected/obvious recommended item IDs (e.g., from a non-personalized algorithm)
    relevant_items : list
        List of relevant item IDs (ground truth)
    
    Returns:
    --------
    serendipity : float
        Serendipity score
    """
    # Find unexpected relevant recommendations
    unexpected_relevant = [item for item in recommendations 
                          if item in relevant_items and item not in expected_recommendations]
    
    # Calculate precision of unexpected relevant items
    if not recommendations:
        return 0.0
    
    return len(unexpected_relevant) / len(recommendations)

def rmse(predictions, ground_truth):
    """
    Calculate Root Mean Squared Error for rating predictions.
    
    Parameters:
    -----------
    predictions : list or dict
        List of (user_id, item_id, predicted_rating) tuples or dictionary {(user_id, item_id): predicted_rating}
    ground_truth : list or dict
        List of (user_id, item_id, actual_rating) tuples or dictionary {(user_id, item_id): actual_rating}
    
    Returns:
    --------
    rmse : float
        Root Mean Squared Error
    """
    # Convert to dictionaries if needed
    if isinstance(predictions, list):
        pred_dict = {(u, i): r for u, i, r in predictions}
    else:
        pred_dict = predictions
    
    if isinstance(ground_truth, list):
        truth_dict = {(u, i): r for u, i, r in ground_truth}
    else:
        truth_dict = ground_truth
    
    # Find common (user, item) pairs
    common_pairs = set(pred_dict.keys()) & set(truth_dict.keys())
    
    if not common_pairs:
        return float('inf')
    
    # Calculate squared errors
    squared_errors = [(pred_dict[pair] - truth_dict[pair])**2 for pair in common_pairs]
    
    # Return RMSE
    return np.sqrt(sum(squared_errors) / len(common_pairs))

def evaluate_recommendations(recommendations_dict, test_data, all_items=None, item_features=None, 
                           item_popularity=None, non_personalized_recs=None):
    """
    Evaluate recommendations using multiple metrics.
    
    Parameters:
    -----------
    recommendations_dict : dict
        Dictionary mapping user_id to list of recommended item IDs
    test_data : pandas.DataFrame
        DataFrame containing ground truth data with columns 'user_id', 'item_id', and optionally 'rating'
    all_items : list, optional
        List of all item IDs for calculating coverage
    item_features : dict, optional
        Dictionary mapping item_id to feature vector for calculating diversity
    item_popularity : dict, optional
        Dictionary mapping item_id to popularity score for calculating novelty
    non_personalized_recs : dict, optional
        Dictionary mapping user_id to list of non-personalized recommendations for calculating serendipity
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # Group test data by user to get relevant items for each user
    user_relevant_items = {}
    for user_id, group in test_data.groupby('user_id'):
        user_relevant_items[user_id] = group['item_id'].tolist()
    
    # Calculate precision and recall at different k values
    precision_at_5 = np.mean([precision_at_k(recommendations_dict.get(user, []), 
                                           relevants, k=5) 
                            for user, relevants in user_relevant_items.items()])
    
    recall_at_5 = np.mean([recall_at_k(recommendations_dict.get(user, []), 
                                      relevants, k=5) 
                         for user, relevants in user_relevant_items.items()])
    
    precision_at_10 = np.mean([precision_at_k(recommendations_dict.get(user, []), 
                                            relevants, k=10) 
                             for user, relevants in user_relevant_items.items()])
    
    recall_at_10 = np.mean([recall_at_k(recommendations_dict.get(user, []), 
                                       relevants, k=10) 
                          for user, relevants in user_relevant_items.items()])
    
    # Calculate MAP
    map_score = mean_average_precision(recommendations_dict, user_relevant_items)
    
    # Calculate NDCG@10
    ndcg_score = np.mean([ndcg_at_k(recommendations_dict.get(user, []), 
                                   relevants, k=10) 
                        for user, relevants in user_relevant_items.items()])
    
    # Create metrics dictionary
    metrics = {
        'precision@5': precision_at_5,
        'recall@5': recall_at_5,
        'precision@10': precision_at_10,
        'recall@10': recall_at_10,
        'map': map_score,
        'ndcg@10': ndcg_score
    }
    
    # Calculate coverage if all_items is provided
    if all_items:
        metrics['coverage'] = coverage(recommendations_dict, all_items)
    
    # Calculate diversity if item_features is provided
    if item_features:
        avg_diversity = np.mean([diversity(recs, item_features) 
                               for recs in recommendations_dict.values()])
        metrics['diversity'] = avg_diversity
    
    # Calculate novelty if item_popularity is provided
    if item_popularity:
        avg_novelty = np.mean([novelty(recs, item_popularity) 
                             for recs in recommendations_dict.values()])
        metrics['novelty'] = avg_novelty
    
    # Calculate serendipity if non_personalized_recs is provided
    if non_personalized_recs:
        avg_serendipity = np.mean([serendipity(recommendations_dict.get(user, []), 
                                             non_personalized_recs.get(user, []), 
                                             user_relevant_items.get(user, [])) 
                                 for user in user_relevant_items.keys()])
        metrics['serendipity'] = avg_serendipity
    
    return metrics