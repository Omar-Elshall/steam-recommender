"""
Configuration settings for the recommendation system.
"""

import os

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
USER_REVIEWS_FILE = os.path.join(DATA_DIR, 'australian_user_reviews.json')
USER_ITEMS_FILE = os.path.join(DATA_DIR, 'australian_user_items.json')
BUNDLE_DATA_FILE = os.path.join(DATA_DIR, 'bundle_data.json')
STEAM_GAMES_FILE = os.path.join(DATA_DIR, 'steam_games.json')

# Algorithm parameters
# Non-personalized recommendations
TOP_N_POPULAR = 20  # Number of top popular games to recommend

# Collaborative filtering parameters
USER_CF_MIN_COMMON_ITEMS = 5  # Minimum number of common games for user similarity
USER_CF_K_NEIGHBORS = 50  # Number of neighbors for user-based CF
ITEM_CF_MIN_COMMON_USERS = 5  # Minimum number of common users for item similarity
ITEM_CF_K_NEIGHBORS = 50  # Number of neighbors for item-based CF

# Content-based filtering parameters
CONTENT_TAGS_WEIGHT = 0.6  # Weight for game tags in content-based filtering
CONTENT_GENRES_WEIGHT = 0.3  # Weight for game genres in content-based filtering
CONTENT_SPECS_WEIGHT = 0.1  # Weight for game specs in content-based filtering

# Matrix factorization parameters
MF_N_FACTORS = 20  # Number of latent factors
MF_N_EPOCHS = 20  # Number of training epochs
MF_LEARNING_RATE = 0.005  # Learning rate
MF_REGULARIZATION = 0.02  # Regularization parameter

# SVD parameters
SVD_N_FACTORS = 20  # Number of singular values to use in SVD

# Recommendation parameters
REC_N_ITEMS = 10  # Number of items to recommend by default