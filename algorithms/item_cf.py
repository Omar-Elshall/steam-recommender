"""
Item-based collaborative filtering recommendation algorithm.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict

from utils.data_loader import create_user_item_matrix
from utils.similarity import cosine_sim, adjusted_cosine_sim, get_top_k_items

class ItemBasedCF:
    """
    Item-based collaborative filtering recommender system.
    """
    
    def __init__(self, k=20, sim_method='cosine', interaction_type='ownership'):
        """
        Initialize the recommender.
        
        Parameters:
        -----------
        k : int
            Number of similar items to consider
        sim_method : str
            Similarity method ('cosine' or 'adjusted_cosine')
        interaction_type : str
            Type of interaction to use ('ownership' or 'playtime')
        """
        self.k = k
        self.sim_method = sim_method
        self.interaction_type = interaction_type
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_index = None
        self.item_index = None
        self.reverse_user_index = None
        self.reverse_item_index = None
        self.user_items_df = None
    
    def fit(self, user_items_df):
        """
        Create user-item matrix and compute item similarity.
        
        Parameters:
        -----------
        user_items_df : pandas.DataFrame
            DataFrame with user-item interactions
        
        Returns:
        --------
        self : ItemBasedCF
            The fitted recommender
        """
        self.user_items_df = user_items_df
        
        # Create user-item matrix
        self.user_item_matrix, self.user_index, self.item_index = create_user_item_matrix(
            user_items_df, self.interaction_type)
        
        # Create reverse mappings
        self.reverse_user_index = {idx: user for user, idx in self.user_index.items()}
        self.reverse_item_index = {idx: item for item, idx in self.item_index.items()}
        
        # Compute item similarity based on the chosen method
        if self.sim_method == 'cosine':
            # Compute cosine similarity between items (columns)
            self.item_similarity = cosine_sim(self.user_item_matrix.T)
        elif self.sim_method == 'adjusted_cosine':
            # Compute adjusted cosine similarity (accounting for user rating bias)
            self.item_similarity = adjusted_cosine_sim(self.user_item_matrix)
        else:
            raise ValueError(f"Unknown similarity method: {self.sim_method}")
        
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
        if self.user_item_matrix is None or self.item_similarity is None:
            raise ValueError("Recommender not fitted yet")
            
        if user_id not in self.user_index:
            # Return empty DataFrame for unknown users
            return pd.DataFrame(columns=['item_id', 'score', 'item_name'])
            
        # Get user index
        user_idx = self.user_index[user_id]
        
        # Get items owned by the user
        user_items = []
        for item_idx in range(self.user_item_matrix.shape[1]):
            if self.user_item_matrix[user_idx, item_idx] > 0:
                item_id = self.reverse_item_index[item_idx]
                user_items.append((item_id, self.user_item_matrix[user_idx, item_idx]))
        
        if not user_items:
            return pd.DataFrame(columns=['item_id', 'score', 'item_name'])
        
        # Create set of excluded items (user's owned items + additional excluded items)
        exclude_item_ids = set(item_id for item_id, _ in user_items)
        if exclude_items:
            exclude_item_ids.update(exclude_items)
        
        # Compute item scores based on similar items
        item_scores = defaultdict(float)
        total_sim = defaultdict(float)
        
        for owned_item_id, owned_item_value in user_items:
            # Skip if item is not in item index (should not happen)
            if owned_item_id not in self.item_index:
                continue
            
            owned_item_idx = self.item_index[owned_item_id]
            
            # Get similar items
            for candidate_idx in range(len(self.reverse_item_index)):
                if candidate_idx == owned_item_idx:
                    continue  # Skip the same item
                
                sim = self.item_similarity[owned_item_idx, candidate_idx]
                if sim <= 0:
                    continue
                    
                candidate_id = self.reverse_item_index[candidate_idx]
                
                # Skip if candidate item is already owned or excluded
                if candidate_id in exclude_item_ids:
                    continue
                
                # Update item score and similarity sum
                item_scores[candidate_id] += sim * owned_item_value
                total_sim[candidate_id] += sim
        
        # Create list of (item_id, score) tuples
        item_score_pairs = []
        for item_id, score in item_scores.items():
            # Normalize by total similarity
            if total_sim[item_id] > 0:
                normalized_score = score / total_sim[item_id]
                item_score_pairs.append((item_id, normalized_score))
        
        # Sort by score (descending) and take top-n
        item_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_items = item_score_pairs[:n]
        
        # Create recommendations DataFrame
        recommendations = []
        for item_id, score in top_items:
            # Get item name
            item_name = self.user_items_df[self.user_items_df['item_id'] == item_id]['item_name'].iloc[0] \
                if item_id in self.user_items_df['item_id'].values else 'Unknown'
            
            recommendations.append({
                'item_id': item_id,
                'score': score,
                'item_name': item_name
            })
        
        return pd.DataFrame(recommendations)
    
    def get_similar_items(self, item_id, n=10):
        """
        Find items similar to a given item.
        
        Parameters:
        -----------
        item_id : str
            ID of the target item
        n : int
            Number of similar items to find
        
        Returns:
        --------
        similar_items : pandas.DataFrame
            DataFrame with similar items
        """
        if self.item_similarity is None:
            raise ValueError("Recommender not fitted yet")
            
        if item_id not in self.item_index:
            return pd.DataFrame(columns=['item_id', 'similarity', 'item_name'])
            
        # Get similar items
        similar_items = get_top_k_items(self.item_similarity, self.item_index, item_id, k=n)
        
        # Create DataFrame with item names
        result = []
        for similar_id, similarity in similar_items:
            # Get item name
            item_name = self.user_items_df[self.user_items_df['item_id'] == similar_id]['item_name'].iloc[0] \
                if similar_id in self.user_items_df['item_id'].values else 'Unknown'
            
            result.append({
                'item_id': similar_id,
                'similarity': similarity,
                'item_name': item_name
            })
        
        return pd.DataFrame(result)
    
    def recommend_similar_to_items(self, item_ids, n=10):
        """
        Recommend items similar to a set of items.
        
        Parameters:
        -----------
        item_ids : list
            List of item IDs to find similar items for
        n : int
            Number of recommendations to generate
        
        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame with recommendations
        """
        if self.item_similarity is None:
            raise ValueError("Recommender not fitted yet")
            
        # Filter valid item IDs
        valid_item_ids = [item_id for item_id in item_ids if item_id in self.item_index]
        
        if not valid_item_ids:
            return pd.DataFrame(columns=['item_id', 'score', 'item_name'])
        
        # Compute item scores based on similarity to provided items
        item_scores = defaultdict(float)
        
        for base_item_id in valid_item_ids:
            base_item_idx = self.item_index[base_item_id]
            
            for candidate_idx in range(len(self.reverse_item_index)):
                if candidate_idx == base_item_idx:
                    continue  # Skip the same item
                
                candidate_id = self.reverse_item_index[candidate_idx]
                
                # Skip if candidate item is in the input list
                if candidate_id in item_ids:
                    continue
                
                # Update item score with similarity
                sim = self.item_similarity[base_item_idx, candidate_idx]
                if sim > 0:
                    item_scores[candidate_id] += sim
        
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
                'score': score,
                'item_name': item_name
            })
        
        return pd.DataFrame(recommendations)