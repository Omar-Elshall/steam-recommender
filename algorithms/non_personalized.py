"""
Non-personalized recommendation algorithms.
"""

import pandas as pd
import numpy as np
from collections import Counter

class NonPersonalizedRecommender:
    """
    Implements non-personalized recommendation techniques:
    - Most popular items (based on ownership or playtime)
    - Highest rated items
    - Recently played games
    - Random recommendations
    """
    
    def __init__(self, method='popularity'):
        """
        Initialize the recommender.
        
        Parameters:
        -----------
        method : str
            Recommendation method ('popularity', 'rating', 'recent', 'random')
        """
        self.method = method
        self.popular_items = None
        self.item_info = None
    
    def fit(self, user_items_df, user_reviews_df=None):
        """
        Compute popular items or ratings statistics.
        
        Parameters:
        -----------
        user_items_df : pandas.DataFrame
            DataFrame containing user-item interactions
        user_reviews_df : pandas.DataFrame, optional
            DataFrame containing user reviews
        
        Returns:
        --------
        self : NonPersonalizedRecommender
            The fitted recommender
        """
        # Store a reference to the original dataframes
        self.user_items_df = user_items_df
        
        # Process based on method
        if self.method == 'popularity':
            # Most popular items based on number of owners
            popularity = user_items_df.groupby('item_id').agg({
                'user_id': 'count',
                'item_name': 'first',
                'playtime_forever': 'sum'
            }).reset_index()
            
            popularity.columns = ['item_id', 'owners_count', 'item_name', 'total_playtime']
            
            # Sort by number of owners (descending)
            self.popular_items = popularity.sort_values('owners_count', ascending=False)
            
        elif self.method == 'playtime':
            # Most popular items based on total playtime
            playtime_popularity = user_items_df.groupby('item_id').agg({
                'user_id': 'count',
                'item_name': 'first',
                'playtime_forever': 'sum'
            }).reset_index()
            
            playtime_popularity.columns = ['item_id', 'owners_count', 'item_name', 'total_playtime']
            
            # Sort by total playtime (descending)
            self.popular_items = playtime_popularity.sort_values('total_playtime', ascending=False)
            
        elif self.method == 'rating':
            # Check if review data is available
            if user_reviews_df is None:
                raise ValueError("user_reviews_df is required for rating-based recommendations")
            
            # Compute average ratings
            ratings = user_reviews_df.groupby('item_id').agg({
                'recommend': ['mean', 'count']
            }).reset_index()
            
            # Flatten column names
            ratings.columns = ['item_id', 'recommend_rate', 'reviews_count']
            
            # Add item names from user_items_df
            item_names = user_items_df[['item_id', 'item_name']].drop_duplicates()
            ratings = ratings.merge(item_names, on='item_id', how='left')
            
            # Sort by average rating (descending) and minimum of 5 reviews
            self.popular_items = ratings[ratings['reviews_count'] >= 5].sort_values(
                'recommend_rate', ascending=False)
            
        elif self.method == 'recent':
            # Most recently played games (based on playtime_2weeks)
            recent = user_items_df[user_items_df['playtime_2weeks'] > 0].groupby('item_id').agg({
                'user_id': 'count',
                'item_name': 'first',
                'playtime_2weeks': 'sum'
            }).reset_index()
            
            recent.columns = ['item_id', 'recent_players', 'item_name', 'recent_playtime']
            
            # Sort by number of recent players (descending)
            self.popular_items = recent.sort_values('recent_players', ascending=False)
            
        elif self.method == 'random':
            # Get all unique items for random recommendations
            all_items = user_items_df[['item_id', 'item_name']].drop_duplicates()
            self.item_info = all_items
        
        return self
    
    def recommend(self, user_id=None, n=10, exclude_items=None):
        """
        Generate recommendations.
        
        Parameters:
        -----------
        user_id : str, optional
            User ID (not used for non-personalized recommendations but included for API consistency)
        n : int
            Number of recommendations to generate
        exclude_items : list, optional
            List of item IDs to exclude from recommendations
        
        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame with recommendations
        """
        if self.method == 'random':
            # Random recommendations
            recommendations = self.item_info.sample(n=min(n, len(self.item_info)))
            
        else:
            # Recommendations based on popularity/rating/recency
            if self.popular_items is None:
                raise ValueError("Recommender not fitted yet")
            
            # Filter out excluded items if needed
            if exclude_items:
                recommendations = self.popular_items[~self.popular_items['item_id'].isin(exclude_items)]
            else:
                recommendations = self.popular_items
            
            # Take top-n recommendations
            recommendations = recommendations.head(n)
        
        return recommendations[['item_id', 'item_name']]
    
    def get_common_games(self, user_ids, min_playtime=0):
        """
        Find common games owned by all specified users.
        
        Parameters:
        -----------
        user_ids : list
            List of user IDs
        min_playtime : int, optional
            Minimum playtime threshold (in minutes)
        
        Returns:
        --------
        common_games : pandas.DataFrame
            DataFrame with common games
        """
        # Filter by user IDs
        user_items = self.user_items_df[self.user_items_df['user_id'].isin(user_ids)]
        
        # Filter by minimum playtime
        if min_playtime > 0:
            user_items = user_items[user_items['playtime_forever'] >= min_playtime]
        
        # Count how many of the specified users own each game
        game_ownership = user_items.groupby('item_id').agg({
            'user_id': 'nunique',
            'item_name': 'first',
            'playtime_forever': 'mean'
        }).reset_index()
        
        # Filter games owned by all specified users
        common_games = game_ownership[game_ownership['user_id'] == len(user_ids)]
        
        # Sort by average playtime
        return common_games.sort_values('playtime_forever', ascending=False)
    
    def get_game_details(self, item_ids):
        """
        Get details for specific games.
        
        Parameters:
        -----------
        item_ids : list
            List of item IDs
        
        Returns:
        --------
        game_details : pandas.DataFrame
            DataFrame with game details
        """
        # Filter by item IDs
        return self.user_items_df[self.user_items_df['item_id'].isin(item_ids)][
            ['item_id', 'item_name']].drop_duplicates()