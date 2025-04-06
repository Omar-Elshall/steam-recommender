"""
Content-based recommendation algorithm.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_loader import create_item_features

class ContentBasedRecommender:
    """
    Content-based recommender system using item features.
    """
    
    def __init__(self, tags_weight=0.6, specs_weight=0.1, genres_weight=0.3):
        """
        Initialize the recommender.
        
        Parameters:
        -----------
        tags_weight : float
            Weight for game tags in similarity calculation
        specs_weight : float
            Weight for game specs in similarity calculation
        genres_weight : float
            Weight for game genres in similarity calculation
        """
        self.tags_weight = tags_weight
        self.specs_weight = specs_weight
        self.genres_weight = genres_weight
        self.item_features = None
        self.feature_names = None
        self.feature_weights = None
        self.item_profiles = None
        self.item_similarity = None
        self.games_df = None
        self.user_items_df = None
    
    def fit(self, games_df, user_items_df=None, bundle_df=None):
        """
        Create item features and compute item similarity.
        
        Parameters:
        -----------
        games_df : pandas.DataFrame
            DataFrame with game information
        user_items_df : pandas.DataFrame, optional
            DataFrame with user-item interactions
        bundle_df : pandas.DataFrame, optional
            DataFrame with bundle information
        
        Returns:
        --------
        self : ContentBasedRecommender
            The fitted recommender
        """
        self.games_df = games_df
        self.user_items_df = user_items_df
        
        # Create item features
        self.item_features, self.feature_names = create_item_features(games_df, bundle_df)
        
        # Create feature weights based on feature type
        self.feature_weights = np.ones(len(self.feature_names))
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name.startswith('tag_'):
                self.feature_weights[i] = self.tags_weight
            elif feature_name.startswith('spec_'):
                self.feature_weights[i] = self.specs_weight
            elif feature_name.startswith('genre_'):
                self.feature_weights[i] = self.genres_weight
        
        # Normalize feature weights
        total_weight = np.sum(self.feature_weights)
        if total_weight > 0:
            self.feature_weights = self.feature_weights / total_weight
        
        # Create item profiles with weighted features
        self.item_profiles = {}
        for item_id, features in self.item_features.items():
            self.item_profiles[item_id] = features * self.feature_weights
        
        # Compute item similarity matrix if needed
        # Note: For performance, we'll compute similarities on-demand rather than precomputing
        
        return self
    
    def recommend(self, user_id=None, n=10, exclude_items=None, items_to_use=None):
        """
        Generate recommendations for a user based on content similarity.
        
        Parameters:
        -----------
        user_id : str, optional
            ID of the target user
        n : int
            Number of recommendations to generate
        exclude_items : list, optional
            List of item IDs to exclude from recommendations
        items_to_use : list, optional
            List of item IDs to base recommendations on (if not using user_id)
        
        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame with recommendations
        """
        if self.item_profiles is None:
            raise ValueError("Recommender not fitted yet")
        
        # Determine which items to use as the basis for recommendations
        base_items = []
        
        if items_to_use:
            # Use specified items
            base_items = [item_id for item_id in items_to_use if item_id in self.item_profiles]
        elif user_id and self.user_items_df is not None:
            # Use items owned by the user
            user_items = self.user_items_df[self.user_items_df['user_id'] == user_id]
            base_items = user_items['item_id'].tolist()
        
        if not base_items:
            # No base items, return empty DataFrame
            return pd.DataFrame(columns=['item_id', 'score', 'item_name'])
        
        # Create set of excluded items
        exclude_set = set(base_items)  # Exclude base items by default
        if exclude_items:
            exclude_set.update(exclude_items)
        
        # Compute average user profile (if using user's items)
        user_profile = np.zeros(len(self.feature_names))
        for item_id in base_items:
            if item_id in self.item_profiles:
                user_profile += self.item_profiles[item_id]
        
        if len(base_items) > 0:
            user_profile = user_profile / len(base_items)
        
        # Compute similarity between user profile and all items
        item_scores = {}
        for item_id, item_profile in self.item_profiles.items():
            # Skip excluded items
            if item_id in exclude_set:
                continue
            
            # Compute cosine similarity
            sim = cosine_similarity([user_profile], [item_profile])[0][0]
            
            # Store similarity score
            item_scores[item_id] = sim
        
        # Sort by score (descending) and take top-n
        top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Create recommendations DataFrame
        recommendations = []
        for item_id, score in top_items:
            # Get item name
            item_name = "Unknown"
            if self.games_df is not None and item_id in self.games_df['item_id'].values:
                item_name = self.games_df[self.games_df['item_id'] == item_id]['item_name'].iloc[0]
            elif self.user_items_df is not None and item_id in self.user_items_df['item_id'].values:
                item_name = self.user_items_df[self.user_items_df['item_id'] == item_id]['item_name'].iloc[0]
            
            recommendations.append({
                'item_id': item_id,
                'score': score,
                'item_name': item_name
            })
        
        return pd.DataFrame(recommendations)
    
    def get_similar_items(self, item_id, n=10, exclude_items=None):
        """
        Find items similar to a given item based on content features.
        
        Parameters:
        -----------
        item_id : str
            ID of the target item
        n : int
            Number of similar items to find
        exclude_items : list, optional
            List of item IDs to exclude from recommendations
        
        Returns:
        --------
        similar_items : pandas.DataFrame
            DataFrame with similar items
        """
        if self.item_profiles is None:
            raise ValueError("Recommender not fitted yet")
            
        if item_id not in self.item_profiles:
            return pd.DataFrame(columns=['item_id', 'similarity', 'item_name'])
        
        # Create set of excluded items
        exclude_set = {item_id}  # Exclude the item itself
        if exclude_items:
            exclude_set.update(exclude_items)
        
        # Get item profile
        item_profile = self.item_profiles[item_id]
        
        # Compute similarity between item and all other items
        item_similarities = {}
        for other_id, other_profile in self.item_profiles.items():
            # Skip excluded items
            if other_id in exclude_set:
                continue
                
            # Compute cosine similarity
            sim = cosine_similarity([item_profile], [other_profile])[0][0]
            
            # Store similarity score
            item_similarities[other_id] = sim
        
        # Sort by similarity (descending) and take top-n
        similar_items = sorted(item_similarities.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Create DataFrame
        result = []
        for similar_id, similarity in similar_items:
            # Get item name
            item_name = "Unknown"
            if self.games_df is not None and similar_id in self.games_df['item_id'].values:
                item_name = self.games_df[self.games_df['item_id'] == similar_id]['item_name'].iloc[0]
            elif self.user_items_df is not None and similar_id in self.user_items_df['item_id'].values:
                item_name = self.user_items_df[self.user_items_df['item_id'] == similar_id]['item_name'].iloc[0]
            
            result.append({
                'item_id': similar_id,
                'similarity': similarity,
                'item_name': item_name
            })
        
        return pd.DataFrame(result)
    
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
            Method for combining profiles ('average', 'least_misery', or 'most_pleasure')
        
        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame with group recommendations
        """
        if self.item_profiles is None or self.user_items_df is None:
            raise ValueError("Recommender not fitted yet or missing user-item data")
        
        # Get items owned by each user
        user_items = {}
        all_owned_items = set()
        
        for user_id in user_ids:
            items = self.user_items_df[self.user_items_df['user_id'] == user_id]['item_id'].tolist()
            user_items[user_id] = items
            all_owned_items.update(items)
        
        # Generate individual recommendations for each user
        individual_recs = {}
        for user_id, items in user_items.items():
            if items:  # Only if user has some items
                # Generate more recommendations than needed to have enough for aggregation
                user_recs = self.recommend(user_id=None, n=100, exclude_items=all_owned_items, items_to_use=items)
                individual_recs[user_id] = {row['item_id']: row['score'] for _, row in user_recs.iterrows()}
        
        # Combine individual recommendations based on method
        if method == 'average':
            # Compute average score across all users
            combined_scores = defaultdict(float)
            for user_id, recs in individual_recs.items():
                for item_id, score in recs.items():
                    combined_scores[item_id] += score / len(user_ids)
                    
        elif method == 'least_misery':
            # Take the minimum score among all users
            all_items = set().union(*[set(recs.keys()) for recs in individual_recs.values()])
            combined_scores = {}
            
            for item_id in all_items:
                # Get scores for this item from all users (default to 0 if not recommended)
                scores = [recs.get(item_id, 0) for recs in individual_recs.values()]
                # Use minimum score
                combined_scores[item_id] = min(scores)
                
        elif method == 'most_pleasure':
            # Take the maximum score among all users
            all_items = set().union(*[set(recs.keys()) for recs in individual_recs.values()])
            combined_scores = {}
            
            for item_id in all_items:
                # Get scores for this item from all users (default to 0 if not recommended)
                scores = [recs.get(item_id, 0) for recs in individual_recs.values()]
                # Use maximum score
                combined_scores[item_id] = max(scores)
        
        # Sort by score and take top-n
        top_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Create recommendations DataFrame
        recommendations = []
        for item_id, score in top_items:
            # Get item name
            item_name = "Unknown"
            if self.games_df is not None and item_id in self.games_df['item_id'].values:
                item_name = self.games_df[self.games_df['item_id'] == item_id]['item_name'].iloc[0]
            elif self.user_items_df is not None and item_id in self.user_items_df['item_id'].values:
                item_name = self.user_items_df[self.user_items_df['item_id'] == item_id]['item_name'].iloc[0]
            
            recommendations.append({
                'item_id': item_id,
                'score': score,
                'item_name': item_name
            })
        
        return pd.DataFrame(recommendations)