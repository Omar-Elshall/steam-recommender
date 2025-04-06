"""
Functions for loading and preprocessing Steam recommendation datasets.
"""

import json
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import config

def load_json_data(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_user_reviews():
    """Load and preprocess user reviews data."""
    reviews_data = load_json_data(config.USER_REVIEWS_FILE)
    
    # Flatten the reviews data
    flattened_reviews = []
    for user_entry in reviews_data:
        user_id = user_entry['user_id']
        for review in user_entry.get('reviews', []):
            flattened_reviews.append({
                'user_id': user_id,
                'item_id': review.get('item_id', ''),
                'recommend': review.get('recommend', False),
                'review': review.get('review', '')
            })
    
    return pd.DataFrame(flattened_reviews)

def load_user_items():
    """Load and preprocess user items data."""
    items_data = load_json_data(config.USER_ITEMS_FILE)
    
    # Flatten the items data
    flattened_items = []
    for user_entry in items_data:
        user_id = user_entry['user_id']
        for item in user_entry.get('items', []):
            flattened_items.append({
                'user_id': user_id,
                'item_id': item.get('item_id', ''),
                'item_name': item.get('item_name', ''),
                'playtime_forever': item.get('playtime_forever', 0),
                'playtime_2weeks': item.get('playtime_2weeks', 0)
            })
    
    return pd.DataFrame(flattened_items)

def load_steam_games():
    """Load and preprocess steam games data."""
    games_data = load_json_data(config.STEAM_GAMES_FILE)
    
    # Convert to DataFrame
    games_df = pd.DataFrame(games_data)
    
    # Ensure id column is renamed to item_id for consistency
    if 'id' in games_df.columns:
        games_df = games_df.rename(columns={'id': 'item_id', 'app_name': 'item_name'})
    
    return games_df

def load_bundle_data():
    """Load and preprocess bundle data."""
    bundle_data = load_json_data(config.BUNDLE_DATA_FILE)
    
    # Flatten bundle items
    flattened_bundles = []
    for bundle in bundle_data:
        bundle_id = bundle.get('bundle_id', '')
        bundle_name = bundle.get('bundle_name', '')
        for item in bundle.get('items', []):
            flattened_bundles.append({
                'bundle_id': bundle_id,
                'bundle_name': bundle_name,
                'item_id': item.get('item_id', ''),
                'item_name': item.get('item_name', ''),
                'genre': item.get('genre', '')
            })
    
    return pd.DataFrame(flattened_bundles)

def create_user_item_matrix(user_items_df, interaction_type='ownership'):
    """
    Create a user-item interaction matrix.
    
    Parameters:
    -----------
    user_items_df : pandas.DataFrame
        DataFrame containing user-item interactions
    interaction_type : str
        Type of interaction to use ('ownership' or 'playtime')
    
    Returns:
    --------
    user_item_matrix : scipy.sparse.csr_matrix
        Sparse matrix of user-item interactions
    user_index : dict
        Mapping from user_id to matrix row index
    item_index : dict
        Mapping from item_id to matrix column index
    """
    # Get unique users and items
    users = user_items_df['user_id'].unique()
    items = user_items_df['item_id'].unique()
    
    # Create mapping dictionaries
    user_index = {user: i for i, user in enumerate(users)}
    item_index = {item: i for i, item in enumerate(items)}
    
    # Create interaction values based on the type
    if interaction_type == 'ownership':
        # Binary matrix for ownership (1 if owned, 0 if not)
        values = np.ones(len(user_items_df))
    elif interaction_type == 'playtime':
        # Use normalized playtime as interaction strength
        max_playtime = user_items_df['playtime_forever'].max()
        if max_playtime > 0:
            values = user_items_df['playtime_forever'].values / max_playtime
        else:
            values = user_items_df['playtime_forever'].values
    else:
        raise ValueError("interaction_type must be 'ownership' or 'playtime'")
    
    # Create row and column indices
    row_indices = [user_index[user] for user in user_items_df['user_id']]
    col_indices = [item_index[item] for item in user_items_df['item_id']]
    
    # Create sparse matrix
    user_item_matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(users), len(items))
    )
    
    return user_item_matrix, user_index, item_index

def create_item_features(games_df, bundle_df=None):
    """
    Create item features matrix for content-based filtering.
    
    Parameters:
    -----------
    games_df : pandas.DataFrame
        DataFrame containing game information
    bundle_df : pandas.DataFrame, optional
        DataFrame containing bundle information
    
    Returns:
    --------
    item_features : dict
        Dictionary mapping item_id to feature vector
    feature_names : list
        List of feature names
    """
    item_features = {}
    all_tags = set()
    all_specs = set()
    all_genres = set()
    
    # Extract all possible tags and specs from games
    for _, game in games_df.iterrows():
        if 'tags' in game and isinstance(game['tags'], list):
            all_tags.update(game['tags'])
        if 'specs' in game and isinstance(game['specs'], list):
            all_specs.update(game['specs'])
    
    # Extract all possible genres from bundles
    if bundle_df is not None:
        all_genres.update(bundle_df['genre'].dropna().unique())
    
    # Convert to sorted lists for consistent feature vector creation
    feature_names = (
        ['tag_' + tag for tag in sorted(all_tags)] +
        ['spec_' + spec for spec in sorted(all_specs)] +
        ['genre_' + genre for genre in sorted(all_genres)]
    )
    
    # Create feature vector for each game
    for _, game in games_df.iterrows():
        item_id = game['item_id']
        features = np.zeros(len(feature_names))
        
        # Set tag features
        if 'tags' in game and isinstance(game['tags'], list):
            for tag in game['tags']:
                feature_idx = feature_names.index('tag_' + tag)
                features[feature_idx] = 1
        
        # Set spec features
        if 'specs' in game and isinstance(game['specs'], list):
            for spec in game['specs']:
                feature_idx = feature_names.index('spec_' + spec)
                features[feature_idx] = 1
        
        # Set genre features from bundle data
        if bundle_df is not None:
            game_bundles = bundle_df[bundle_df['item_id'] == item_id]
            for _, bundle_game in game_bundles.iterrows():
                if 'genre' in bundle_game and bundle_game['genre']:
                    feature_idx = feature_names.index('genre_' + bundle_game['genre'])
                    features[feature_idx] = 1
        
        item_features[item_id] = features
    
    return item_features, feature_names

def get_common_games(user_items_df, user_ids):
    """
    Find games that all specified users own.
    
    Parameters:
    -----------
    user_items_df : pandas.DataFrame
        DataFrame containing user-item interactions
    user_ids : list
        List of user IDs to find common games for
    
    Returns:
    --------
    common_games : pandas.DataFrame
        DataFrame of games owned by all specified users
    """
    # Filter the dataframe to include only the specified users
    filtered_df = user_items_df[user_items_df['user_id'].isin(user_ids)]
    
    # Group by item_id and count how many of the specified users own each game
    game_counts = filtered_df.groupby('item_id').agg({
        'user_id': 'nunique',
        'item_name': 'first',
        'playtime_forever': 'mean'
    }).reset_index()
    
    # Filter to games owned by all specified users
    common_games = game_counts[game_counts['user_id'] == len(user_ids)]
    
    return common_games.sort_values('playtime_forever', ascending=False)