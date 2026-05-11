import numpy as np
import pandas as pd
import pickle
import math
import time
import os
import cupy as cp

def load_data():
    """Load data and create dictionaries"""
    user_item_matrix = pd.read_pickle('processing/processedData/user_item_matrix.pkl')
    
    # Generate dictionaries
    user2game = {}
    game2user = {}
    
    for user in user_item_matrix.index:
        owned_games = user_item_matrix.columns[user_item_matrix.loc[user] == True].tolist()
        user2game[user] = owned_games
        
        for game in owned_games:
            if game not in game2user:
                game2user[game] = []
            game2user[game].append(user)
    
    # Save dictionaries
    with open('processing/processedData/user2game_dict.pkl', 'wb') as f:
        pickle.dump(user2game, f)
    with open('processing/processedData/game2user_dict.pkl', 'wb') as f:
        pickle.dump(game2user, f)
    
    return user_item_matrix, user2game, game2user

def compute_user_similarity(target_user, user2game, min_common=5):
    """Calculate similarity between target user and all other users"""
    target_games = set(user2game[target_user])
    
    similarities = {}
    for other_user, other_games in user2game.items():
        if other_user != target_user:
            other_games_set = set(other_games)
            common_games = target_games.intersection(other_games_set)
            
            if len(common_games) >= min_common:
                # Jaccard similarity
                union = len(target_games.union(other_games_set))
                if union > 0:
                    similarity = len(common_games) / union
                    similarities[other_user] = similarity
    
    return similarities

def item_based_cf(target_user, user2game, game2user, similarity_matrix, top_n=10):
    """Generate recommendations using item-based collaborative filtering"""
    # Get owned games and candidate games
    owned_games = set(user2game[target_user])
    all_games = set(game2user.keys())
    candidate_games = all_games - owned_games
    
    # Calculate game popularity as baseline
    total_users = len(user2game)
    game_popularity = {game: len(users)/total_users for game, users in game2user.items()}
    
    # Make predictions
    predictions = {}
    for game in candidate_games:
        similarities = []
        for owned_game in owned_games:
            if game in similarity_matrix and owned_game in similarity_matrix[game]:
                similarity = similarity_matrix[game][owned_game]
                similarities.append((owned_game, similarity))
        
        # Sort and take top k neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = similarities[:25]
        
        # Calculate weighted similarity score
        numerator = 0
        denominator = 0
        for _, similarity in top_neighbors:
            numerator += similarity
            denominator += 1  # Count each neighbor equally
        
        # Convert to ownership probability
        if denominator > 0:
            # Base probability is weighted average of similarities
            raw_score = numerator / denominator
            
            # Apply sigmoid-like function to get probability
            # This scales similarity scores to reasonable probabilities
            prob = (1 + math.tanh(3 * raw_score - 1.5)) / 2
            
            # Blend with popularity for games with weak signals
            if raw_score < 0.2:
                pop_weight = 0.7 - raw_score * 2  # More weight to popularity for low scores
                prob = (1 - pop_weight) * prob + pop_weight * game_popularity.get(game, 0)
                
            predictions[game] = prob
        else:
            predictions[game] = game_popularity.get(game, 0)
    
    # Sort and return top recommendations
    recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommendations

def user_based_cf(target_user, user2game, game2user, top_n=10):
    """Generate recommendations using user-based collaborative filtering"""
    # Get owned games and candidate games
    owned_games = set(user2game[target_user])
    all_games = set(game2user.keys())
    candidate_games = all_games - owned_games
    
    # Calculate similarities to other users
    similarities = compute_user_similarity(target_user, user2game)
    
    # Calculate game popularity as baseline
    total_users = len(user2game)
    game_popularity = {game: len(users)/total_users for game, users in game2user.items()}
    
    # Make predictions
    predictions = {}
    for game in candidate_games:
        if game in game2user:
            owners = game2user[game]
            
            # Find similar users who own this game
            similar_owners = [(user, similarities[user]) for user in owners if user in similarities]
            
            # If we have similar users who own this game
            if similar_owners:
                # Calculate strength of ownership signal
                total_similarity = sum(sim for _, sim in similar_owners)
                ownership_signal = total_similarity / len(similar_owners)
                
                # Convert signal to probability using sigmoid-like function
                # Scale and shift parameters can be tuned
                prob = (1 + math.tanh(4 * ownership_signal - 1.5)) / 2
                
                # For low signals, blend with game popularity
                if ownership_signal < 0.2:
                    pop_weight = 0.8 - ownership_signal * 2
                    prob = (1 - pop_weight) * prob + pop_weight * game_popularity.get(game, 0)
                
                predictions[game] = prob
            else:
                predictions[game] = 0.1 * game_popularity.get(game, 0)  # Low probability for no signal
        else:
            predictions[game] = 0.05 * game_popularity.get(game, 0)  # Very low probability for games with no owners
    
    # Sort and return top recommendations
    recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommendations