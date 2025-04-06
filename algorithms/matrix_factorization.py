"""
Matrix factorization recommendation algorithm.
"""

import numpy as np
import pandas as pd
import time
from scipy.sparse import csr_matrix

from utils.data_loader import create_user_item_matrix

class MatrixFactorizationRecommender:
    """
    Matrix factorization recommender system using SGD optimization.
    """
    
    def __init__(self, n_factors=20, n_epochs=20, learning_rate=0.005, regularization=0.02):
        """
        Initialize the recommender.
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors
        n_epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for gradient descent
        regularization : float
            Regularization parameter
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.user_index = None
        self.item_index = None
        self.reverse_user_index = None
        self.reverse_item_index = None
        self.user_items_df = None
        self.global_mean = 0.0
    
    def fit(self, user_items_df, interaction_type='playtime', verbose=True):
        """
        Train the matrix factorization model.
        
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
        self : MatrixFactorizationRecommender
            The fitted recommender
        """
        self.user_items_df = user_items_df
        
        # Create user-item matrix
        self.user_item_matrix, self.user_index, self.item_index = create_user_item_matrix(
            user_items_df, interaction_type)
        
        # Create reverse mappings
        self.reverse_user_index = {idx: user for user, idx in self.user_index.items()}
        self.reverse_item_index = {idx: item for item, idx in self.item_index.items()}
        
        # Number of users and items
        n_users, n_items = self.user_item_matrix.shape
        
        # Calculate global mean
        if self.user_item_matrix.nnz > 0:
            self.global_mean = self.user_item_matrix.sum() / self.user_item_matrix.nnz
        else:
            self.global_mean = 0.0
        
        # Initialize user and item factors
        self.user_factors = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(n_items, self.n_factors))
        
        # Convert to dense representation for training
        # For very large matrices, you might want to use a more efficient approach
        if isinstance(self.user_item_matrix, csr_matrix):
            matrix_dense = self.user_item_matrix.toarray()
        else:
            matrix_dense = self.user_item_matrix
        
        # Create mask of observed entries
        mask = matrix_dense > 0
        
        # Training loop
        start_time = time.time()
        best_rmse = float('inf')
        
        for epoch in range(self.n_epochs):
            # Compute all predictions
            predictions = np.dot(self.user_factors, self.item_factors.T)
            
            # Compute errors only for observed entries
            errors = np.zeros_like(matrix_dense)
            errors[mask] = matrix_dense[mask] - predictions[mask]
            
            # Update user and item factors using gradient descent
            for u in range(n_users):
                user_mask = mask[u]
                if np.any(user_mask):
                    user_error = errors[u, user_mask]
                    # Update user factors
                    user_grad = np.dot(user_error, self.item_factors[user_mask]) - self.regularization * self.user_factors[u]
                    self.user_factors[u] += self.learning_rate * user_grad
            
            for i in range(n_items):
                item_mask = mask[:, i]
                if np.any(item_mask):
                    item_error = errors[item_mask, i]
                    # Update item factors
                    item_grad = np.dot(item_error, self.user_factors[item_mask]) - self.regularization * self.item_factors[i]
                    self.item_factors[i] += self.learning_rate * item_grad
            
            # Compute RMSE for this epoch
            current_rmse = np.sqrt(np.mean(errors[mask]**2))
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, RMSE: {current_rmse:.4f}, Time: {time.time() - start_time:.2f}s")
            
            # Save best model
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_user_factors = self.user_factors.copy()
                best_item_factors = self.item_factors.copy()
        
        # Use best model
        self.user_factors = best_user_factors
        self.item_factors = best_item_factors
        
        if verbose:
            print(f"Training completed in {time.time() - start_time:.2f}s, Best RMSE: {best_rmse:.4f}")
        
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
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Recommender not fitted yet")
            
        if user_id not in self.user_index:
            # For unknown users, we can use average factors or return empty DataFrame
            return pd.DataFrame(columns=['item_id', 'score', 'item_name'])
            
        # Get user index
        user_idx = self.user_index[user_id]
        
        # Compute predicted ratings for all items
        user_vector = self.user_factors[user_idx]
        predicted_ratings = np.dot(user_vector, self.item_factors.T)
        
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
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Recommender not fitted yet")
            
        if user_id not in self.user_index or item_id not in self.item_index:
            # Return global mean for unknown users or items
            return self.global_mean
            
        user_idx = self.user_index[user_id]
        item_idx = self.item_index[item_id]
        
        # Compute dot product of user and item factors
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
    
    def get_user_profile(self, user_id):
        """
        Get the latent factor profile for a user.
        
        Parameters:
        -----------
        user_id : str
            ID of the user
            
        Returns:
        --------
        profile : numpy.ndarray
            User's latent factor profile
        """
        if self.user_factors is None:
            raise ValueError("Recommender not fitted yet")
            
        if user_id not in self.user_index:
            # Return average user factors for unknown users
            return np.mean(self.user_factors, axis=0)
            
        user_idx = self.user_index[user_id]
        return self.user_factors[user_idx]
    
    def get_item_profile(self, item_id):
        """
        Get the latent factor profile for an item.
        
        Parameters:
        -----------
        item_id : str
            ID of the item
            
        Returns:
        --------
        profile : numpy.ndarray
            Item's latent factor profile
        """
        if self.item_factors is None:
            raise ValueError("Recommender not fitted yet")
            
        if item_id not in self.item_index:
            # Return average item factors for unknown items
            return np.mean(self.item_factors, axis=0)
            
        item_idx = self.item_index[item_id]
        return self.item_factors[item_idx]
    
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
        if self.user_factors is None or self.item_factors is None:
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
        
        # Compute predictions for each user and item
        item_scores = {}
        
        for item_idx in range(self.item_factors.shape[0]):
            item_id = self.reverse_item_index[item_idx]
            
            # Skip if item is owned by any user in the group
            if item_id in all_owned_items:
                continue
            
            # Get predictions for all users
            predictions = []
            for user_id in valid_user_ids:
                user_idx = self.user_index[user_id]
                pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                predictions.append(pred)
            
            # Combine predictions based on method
            if method == 'average':
                item_scores[item_id] = np.mean(predictions)
            elif method == 'least_misery':
                item_scores[item_id] = np.min(predictions)
            elif method == 'most_pleasure':
                item_scores[item_id] = np.max(predictions)
        
        # Sort by score (descending) and take top-n
        top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Create recommendations DataFrame
        recommendations = []
        for item_id, score in top_items:
            # Get item name
            item_name = self.user_items_df[self.user_items_df['item_id'] == item_id]['item_name'].iloc[0] \
                if item_id in self.user_items_df['item_id'].values else 'Unknown'
            
            recommendations.append({
                'item_id': item_id,
                'score': float(score),  # Convert from numpy float to Python float
                'item_name': item_name
            })
        
        return pd.DataFrame(recommendations)