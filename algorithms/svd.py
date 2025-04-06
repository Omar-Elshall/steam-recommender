"""
Singular Value Decomposition (SVD) recommendation algorithm.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from utils.data_loader import create_user_item_matrix

class SVDRecommender:
    """
    Recommender system using Singular Value Decomposition (SVD).
    """
    
    def __init__(self, n_factors=20):
        """
        Initialize the recommender.
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors (singular values) to use
        """
        self.n_factors = n_factors
        self.user_item_matrix = None
        self.user_index = None
        self.item_index = None
        self.reverse_user_index = None
        self.reverse_item_index = None
        self.user_items_df = None
        self.U = None  # User features
        self.sigma = None  # Singular values
        self.Vt = None  # Item features
        self.reconstructed_matrix = None
    
    def fit(self, user_items_df, interaction_type='playtime', verbose=True):
        """
        Perform SVD on the user-item matrix.
        
        Parameters:
        -----------
        user_items_df : pandas.DataFrame
            DataFrame with user-item interactions
        interaction_type : str
            Type of interaction to use ('ownership' or 'playtime')
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        self : SVDRecommender
            The fitted recommender
        """
        self.user_items_df = user_items_df
        
        # Create user-item matrix
        self.user_item_matrix, self.user_index, self.item_index = create_user_item_matrix(
            user_items_df, interaction_type)
        
        # Create reverse mappings
        self.reverse_user_index = {idx: user for user, idx in self.user_index.items()}
        self.reverse_item_index = {idx: item for item, idx in self.item_index.items()}
        
        # Convert to dense matrix for SVD
        # For very large matrices, you might want to use a more efficient approach
        matrix_dense = self.user_item_matrix.toarray()
        
        # Center the matrix by subtracting row means
        self.row_means = np.mean(matrix_dense, axis=1).reshape(-1, 1)
        centered_matrix = matrix_dense - self.row_means
        
        # Fill NaN values with zeros (if any)
        centered_matrix = np.nan_to_num(centered_matrix)
        
        if verbose:
            print(f"Performing SVD with {self.n_factors} factors...")
        
        # Perform SVD
        n_factors = min(self.n_factors, min(matrix_dense.shape) - 1)
        self.U, self.sigma, self.Vt = svds(centered_matrix, k=n_factors)
        
        # Sort by singular values (descending)
        idx = np.argsort(self.sigma)[::-1]
        self.sigma = self.sigma[idx]
        self.U = self.U[:, idx]
        self.Vt = self.Vt[idx, :]
        
        # Reconstruct the matrix with low rank approximation
        self.reconstructed_matrix = self.U.dot(np.diag(self.sigma)).dot(self.Vt) + self.row_means
        
        if verbose:
            # Calculate reconstruction error
            mse = np.mean((matrix_dense - self.reconstructed_matrix) ** 2)
            print(f"SVD completed. Mean Squared Error: {mse:.4f}")
        
        return self
    
    def recommend(self, user_id, n=10, exclude_items=None):
        """
        Generate recommendations for a user.
        
        Parameters:
        -----------
        user_id : str
            ID of the target user
        n : int
            Number of recommendations to generate
        exclude_items : list, optional
            List of item IDs to exclude from recommendations
        
        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame with recommendations
        """
        if self.reconstructed_matrix is None:
            raise ValueError("Recommender not fitted yet")
            
        if user_id not in self.user_index:
            # For unknown users, we can return empty DataFrame
            return pd.DataFrame(columns=['item_id', 'score', 'item_name'])
            
        # Get user index
        user_idx = self.user_index[user_id]
        
        # Get predicted ratings from reconstructed matrix
        predicted_ratings = self.reconstructed_matrix[user_idx]
        
        # Get items already owned by the user
        owned_items = set()
        if self.user_item_matrix is not None:
            user_row = self.user_item_matrix[user_idx].toarray().flatten()
            owned_indices = np.where(user_row > 0)[0]
            owned_items = {self.reverse_item_index[idx] for idx in owned_indices}
        
        # Add additional excluded items
        if exclude_items:
            owned_items.update(exclude_items)
        
        # Create list of (item_index, predicted_rating) pairs
        item_rating_pairs = []
        for item_idx, rating in enumerate(predicted_ratings):
            item_id = self.reverse_item_index[item_idx]
            
            # Skip owned or excluded items
            if item_id in owned_items:
                continue
                
            item_rating_pairs.append((item_idx, rating))
        
        # Sort by predicted rating (descending) and take top-n
        item_rating_pairs.sort(key=lambda x: x[1], reverse=True)
        top_items = item_rating_pairs[:n]
        
        # Create recommendations DataFrame
        recommendations = []
        for item_idx, rating in top_items:
            item_id = self.reverse_item_index[item_idx]
            
            # Get item name
            item_name = self.user_items_df[self.user_items_df['item_id'] == item_id]['item_name'].iloc[0] \
                if item_id in self.user_items_df['item_id'].values else 'Unknown'
            
            recommendations.append({
                'item_id': item_id,
                'score': float(rating),  # Convert from numpy float to Python float
                'item_name': item_name
            })
        
        return pd.DataFrame(recommendations)
    
    def predict(self, user_id, item_id):
        """
        Predict rating for a specific user-item pair.
        
        Parameters:
        -----------
        user_id : str
            ID of the user
        item_id : str
            ID of the item
        
        Returns:
        --------
        predicted_rating : float
            Predicted rating
        """
        if self.reconstructed_matrix is None:
            raise ValueError("Recommender not fitted yet")
            
        if user_id not in self.user_index or item_id not in self.item_index:
            # Return global mean for unknown users or items
            return np.mean(self.row_means)
            
        user_idx = self.user_index[user_id]
        item_idx = self.item_index[item_id]
        
        # Return the predicted rating from the reconstructed matrix
        return self.reconstructed_matrix[user_idx, item_idx]
    
    def get_user_factors(self, user_id):
        """
        Get the latent factor values for a user.
        
        Parameters:
        -----------
        user_id : str
            ID of the user
        
        Returns:
        --------
        factors : numpy.ndarray
            User's latent factors
        """
        if self.U is None:
            raise ValueError("Recommender not fitted yet")
            
        if user_id not in self.user_index:
            # Return average factors for unknown users
            return np.mean(self.U, axis=0)
            
        user_idx = self.user_index[user_id]
        return self.U[user_idx]
    
    def get_item_factors(self, item_id):
        """
        Get the latent factor values for an item.
        
        Parameters:
        -----------
        item_id : str
            ID of the item
        
        Returns:
        --------
        factors : numpy.ndarray
            Item's latent factors
        """
        if self.Vt is None:
            raise ValueError("Recommender not fitted yet")
            
        if item_id not in self.item_index:
            # Return average factors for unknown items
            return np.mean(self.Vt, axis=1)
            
        item_idx = self.item_index[item_id]
        return self.Vt[:, item_idx]
    
    def recommend_for_group(self, user_ids, n=10, method='average'):
        """
        Generate recommendations for a group of users.
        
        Parameters:
        -----------
        user_ids : list
            List of user IDs in the group
        n : int
            Number of recommendations to generate
        method : str
            Method for combining individual predictions ('average', 'least_misery', or 'most_pleasure')
        
        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame with group recommendations
        """
        if self.reconstructed_matrix is None:
            raise ValueError("Recommender not fitted yet")
        
        # Filter valid user IDs
        valid_user_ids = [user_id for user_id in user_ids if user_id in self.user_index]
        
        if not valid_user_ids:
            return pd.DataFrame(columns=['item_id', 'score', 'item_name'])
        
        # Get all items owned by any user in the group
        all_owned_items = set()
        for user_id in valid_user_ids:
            user_idx = self.user_index[user_id]
            user_row = self.user_item_matrix[user_idx].toarray().flatten()
            owned_indices = np.where(user_row > 0)[0]
            owned_items = {self.reverse_item_index[idx] for idx in owned_indices}
            all_owned_items.update(owned_items)
        
        # For each candidate item, get predictions for all users
        n_items = self.reconstructed_matrix.shape[1]
        group_scores = {}
        
        for item_idx in range(n_items):
            item_id = self.reverse_item_index[item_idx]
            
            # Skip if item is owned by any user in the group
            if item_id in all_owned_items:
                continue
            
            # Get predictions for all users
            predictions = []
            for user_id in valid_user_ids:
                user_idx = self.user_index[user_id]
                pred = self.reconstructed_matrix[user_idx, item_idx]
                predictions.append(pred)
            
            # Combine predictions based on method
            if method == 'average':
                group_scores[item_id] = np.mean(predictions)
            elif method == 'least_misery':
                group_scores[item_id] = np.min(predictions)
            elif method == 'most_pleasure':
                group_scores[item_id] = np.max(predictions)
        
        # Sort by score (descending) and take top-n
        top_items = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Create recommendations DataFrame
        recommendations = []
        for item_id, score in top_items:
            # Get item name
            item_name = self.user_items_df[self.user_items_df['item_id'] == item_id]['item_name'].iloc[0] \
                if item_id in self.user_items_df['item_id'].values else 'Unknown'
            
            recommendations.append({
                'item_id': item_id,
                'score': float(score),
                'item_name': item_name
            })
        
        return pd.DataFrame(recommendations)