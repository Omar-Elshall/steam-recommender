"""
User-based collaborative filtering recommendation algorithm.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict
import heapq

from utils.data_loader import create_user_item_matrix
from utils.similarity import cosine_sim, pearson_sim, jaccard_sim, get_top_k_users

class UserBasedCF:
    """
    User-based collaborative filtering recommender system.
    """
    
    def __init__(self, k=20, sim_method='cosine', interaction_type='ownership'):
        """
        Initialize the recommender.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors to consider
        sim_method : str
            Similarity method ('cosine', 'pearson', or 'jaccard')
        interaction_type : str
            Type of interaction to use ('ownership' or 'playtime')
        """
        self.k = k
        self.sim_method = sim_method
        self.interaction_type = interaction_type
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_index = None
        self.item_index = None
        self.reverse_user_index = None
        self.reverse_item_index = None
        self.user_items_df = None
    
    def fit(self, user_items_df):
        """
        Create user-item matrix and compute user similarity.
        
        Parameters:
        -----------
        user_items_df : pandas.DataFrame
            DataFrame with user-item interactions
        
        Returns:
        --------
        self : UserBasedCF
            The fitted recommender
        """
        self.user_items_df = user_items_df
        
        # Create user-item matrix
        self.user_item_matrix, self.user_index, self.item_index = create_user_item_matrix(
            user_items_df, self.interaction_type)
        
        # Create reverse mappings
        self.reverse_user_index = {idx: user for user, idx in self.user_index.items()}
        self.reverse_item_index = {idx: item for item, idx in self.item_index.items()}
        
        # Compute user similarity based on the chosen method
        if self.sim_method == 'cosine':
            self.user_similarity = cosine_sim(self.user_item_matrix)
        elif self.sim_method == 'pearson':
            self.user_similarity = pearson_sim(self.user_item_matrix)
        elif self.sim_method == 'jaccard':
            self.user_similarity = jaccard_sim(self.user_item_matrix)
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
        if self.user_item_matrix is None:
            raise ValueError("Recommender not fitted yet")
            
        if user_id not in self.user_index:
            # Return empty DataFrame for unknown users
            return pd.DataFrame(columns=['item_id', 'score', 'item_name'])
            
        # Get user index
        user_idx = self.user_index[user_id]
        
        # Find similar users
        similar_users = []
        for idx, sim in enumerate(self.user_similarity[user_idx]):
            if idx != user_idx and sim > 0:
                similar_users.append((idx, sim))
        
        # Sort by similarity (descending) and take top-k
        similar_users.sort(key=lambda x: x[1], reverse=True)
        similar_users = similar_users[:self.k]
        
        # Get items owned by the user
        if exclude_items is None:
            user_items = self.user_items_df[self.user_items_df['user_id'] == user_id]['item_id'].values
        else:
            user_items = exclude_items
        
        # Compute item scores based on similar users
        item_scores = defaultdict(float)
        total_sim = defaultdict(float)
        
        for neighbor_idx, sim in similar_users:
            neighbor_id = self.reverse_user_index[neighbor_idx]
            
            # Get items owned by the neighbor
            neighbor_items = self.user_items_df[self.user_items_df['user_id'] == neighbor_id]
            
            for _, row in neighbor_items.iterrows():
                item_id = row['item_id']
                
                # Skip if the user already owns this item
                if item_id in user_items:
                    continue
                
                # Use playtime as weight if available, otherwise use 1
                if self.interaction_type == 'playtime' and 'playtime_forever' in row:
                    weight = row['playtime_forever']
                    # Normalize playtime (0-1 scale)
                    max_playtime = self.user_items_df['playtime_forever'].max()
                    if max_playtime > 0:
                        weight = weight / max_playtime
                else:
                    weight = 1.0
                
                # Update item score and similarity sum
                item_scores[item_id] += sim * weight
                total_sim[item_id] += sim
                
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
    
    def get_similar_users(self, user_id, n=10):
        """
        Find users similar to a given user.
        
        Parameters:
        -----------
        user_id : str
            ID of the target user
        n : int
            Number of similar users to find
        
        Returns:
        --------
        similar_users : pandas.DataFrame
            DataFrame with similar users
        """
        if self.user_similarity is None:
            raise ValueError("Recommender not fitted yet")
            
        if user_id not in self.user_index:
            return pd.DataFrame(columns=['user_id', 'similarity'])
            
        # Get similar users
        similar_users = get_top_k_users(self.user_similarity, self.user_index, user_id, k=n)
        
        # Create DataFrame
        return pd.DataFrame(similar_users, columns=['user_id', 'similarity'])
    
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
            Method for combining individual recommendations ('average', 'least_misery', or 'most_pleasure')
        
        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame with group recommendations
        """
        if self.user_item_matrix is None:
            raise ValueError("Recommender not fitted yet")
            
        # Filter valid user IDs
        valid_user_ids = [user_id for user_id in user_ids if user_id in self.user_index]
        
        if not valid_user_ids:
            return pd.DataFrame(columns=['item_id', 'score', 'item_name'])
            
        # Get individual recommendations for each user (with high limit)
        individual_recs = {}
        for user_id in valid_user_ids:
            # Get items already owned by this user
            user_items = self.user_items_df[self.user_items_df['user_id'] == user_id]['item_id'].values
            
            # Get individual recommendations (more than needed for group)
            user_recs = self.recommend(user_id, n=100, exclude_items=user_items)
            
            # Convert to dictionary for easier access
            individual_recs[user_id] = {row['item_id']: row['score'] for _, row in user_recs.iterrows()}
        
        # Combine individual recommendations based on method
        combined_scores = defaultdict(float)
        
        if method == 'average':
            # Average score across all users
            for user_id, recs in individual_recs.items():
                for item_id, score in recs.items():
                    combined_scores[item_id] += score / len(valid_user_ids)
                    
        elif method == 'least_misery':
            # Minimum score among all users
            for item_id in set().union(*[set(recs.keys()) for recs in individual_recs.values()]):
                scores = [recs.get(item_id, 0) for recs in individual_recs.values()]
                combined_scores[item_id] = min(scores)
                
        elif method == 'most_pleasure':
            # Maximum score among all users
            for item_id in set().union(*[set(recs.keys()) for recs in individual_recs.values()]):
                scores = [recs.get(item_id, 0) for recs in individual_recs.values()]
                combined_scores[item_id] = max(scores)
        
        # Sort by combined score (descending) and take top-n
        top_items = heapq.nlargest(n, combined_scores.items(), key=lambda x: x[1])
        
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