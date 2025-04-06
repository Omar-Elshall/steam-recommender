"""
Similarity functions for recommendation algorithms.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

def cosine_sim(matrix, index=None):
    """
    Compute cosine similarity between all pairs of vectors in the matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray or scipy.sparse.csr_matrix
        Matrix where each row is a vector
    index : list, optional
        List of indices for the rows in the matrix
    
    Returns:
    --------
    similarity_matrix : numpy.ndarray
        Matrix of pairwise similarities
    """
    similarity_matrix = cosine_similarity(matrix)
    
    # Set diagonal to 0 to avoid recommending the same item
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

def pearson_sim(matrix, index=None):
    """
    Compute Pearson correlation between all pairs of vectors in the matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray or scipy.sparse.csr_matrix
        Matrix where each row is a vector
    index : list, optional
        List of indices for the rows in the matrix
    
    Returns:
    --------
    similarity_matrix : numpy.ndarray
        Matrix of pairwise similarities
    """
    # Convert to dense if sparse
    if isinstance(matrix, csr_matrix):
        matrix = matrix.toarray()
    
    # Center the data by subtracting the mean of each row
    centered_matrix = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    
    # Compute Pearson correlation (normalized cosine similarity)
    similarity_matrix = cosine_similarity(centered_matrix)
    
    # Set diagonal to 0 to avoid recommending the same item
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

def jaccard_sim(matrix, index=None):
    """
    Compute Jaccard similarity between all pairs of vectors in the matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray or scipy.sparse.csr_matrix
        Binary matrix where each row is a vector
    index : list, optional
        List of indices for the rows in the matrix
    
    Returns:
    --------
    similarity_matrix : numpy.ndarray
        Matrix of pairwise similarities
    """
    # Convert to binary matrix
    if isinstance(matrix, csr_matrix):
        binary_matrix = (matrix > 0).toarray().astype(int)
    else:
        binary_matrix = (matrix > 0).astype(int)
    
    # Compute Jaccard similarity
    similarity_matrix = 1 - pairwise_distances(binary_matrix, metric='jaccard')
    
    # Set diagonal to 0 to avoid recommending the same item
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

def adjusted_cosine_sim(matrix, index=None):
    """
    Compute adjusted cosine similarity - cosine similarity after centering by user means.
    
    Parameters:
    -----------
    matrix : numpy.ndarray or scipy.sparse.csr_matrix
        Matrix where each row is a user and each column is an item
    index : list, optional
        List of indices for the rows in the matrix
    
    Returns:
    --------
    similarity_matrix : numpy.ndarray
        Matrix of pairwise similarities between items
    """
    # Convert to dense if sparse
    if isinstance(matrix, csr_matrix):
        matrix = matrix.toarray()
    
    # Center by user means
    user_means = np.nanmean(matrix, axis=1, keepdims=True)
    centered_matrix = matrix - user_means
    
    # Replace NaN with 0 after centering
    centered_matrix = np.nan_to_num(centered_matrix)
    
    # Compute item-item similarity (transpose first)
    similarity_matrix = cosine_similarity(centered_matrix.T)
    
    # Set diagonal to 0 to avoid recommending the same item
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

def get_top_k_items(similarity_matrix, item_indices, item_id, k=10, exclude_items=None):
    """
    Get top-k most similar items to a given item.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Matrix of pairwise similarities
    item_indices : dict
        Mapping from item_id to matrix index
    item_id : str
        ID of the target item
    k : int
        Number of similar items to retrieve
    exclude_items : list, optional
        List of item IDs to exclude from recommendations
    
    Returns:
    --------
    similar_items : list
        List of (item_id, similarity_score) tuples
    """
    if item_id not in item_indices:
        return []
    
    item_idx = item_indices[item_id]
    item_similarities = similarity_matrix[item_idx]
    
    # Create reverse mapping from index to item_id
    idx_to_item = {idx: item for item, idx in item_indices.items()}
    
    # Create list of (item_id, similarity) tuples
    item_sim_pairs = [(idx_to_item[idx], sim) for idx, sim in enumerate(item_similarities)]
    
    # Filter out excluded items
    if exclude_items:
        item_sim_pairs = [(item, sim) for item, sim in item_sim_pairs if item not in exclude_items]
    
    # Sort by similarity (descending) and take top k
    item_sim_pairs.sort(key=lambda x: x[1], reverse=True)
    return item_sim_pairs[:k]

def get_top_k_users(similarity_matrix, user_indices, user_id, k=10):
    """
    Get top-k most similar users to a given user.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Matrix of pairwise similarities
    user_indices : dict
        Mapping from user_id to matrix index
    user_id : str
        ID of the target user
    k : int
        Number of similar users to retrieve
    
    Returns:
    --------
    similar_users : list
        List of (user_id, similarity_score) tuples
    """
    if user_id not in user_indices:
        return []
    
    user_idx = user_indices[user_id]
    user_similarities = similarity_matrix[user_idx]
    
    # Create reverse mapping from index to user_id
    idx_to_user = {idx: user for user, idx in user_indices.items()}
    
    # Create list of (user_id, similarity) tuples
    user_sim_pairs = [(idx_to_user[idx], sim) for idx, sim in enumerate(user_similarities)]
    
    # Sort by similarity (descending) and take top k
    user_sim_pairs.sort(key=lambda x: x[1], reverse=True)
    return user_sim_pairs[:k]