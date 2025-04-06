"""
Main script for the Steam game recommendation system.
"""

import argparse
import pandas as pd
import os
import json
import time
from tabulate import tabulate

# Import utility functions
from utils.data_loader import (
    load_user_reviews, load_user_items, load_steam_games,
    load_bundle_data, get_common_games
)
from utils.evaluation import evaluate_recommendations

# Import recommendation algorithms
from algorithms.non_personalized import NonPersonalizedRecommender
from algorithms.user_cf import UserBasedCF
from algorithms.item_cf import ItemBasedCF
from algorithms.content_based import ContentBasedRecommender
from algorithms.matrix_factorization import MatrixFactorizationRecommender
from algorithms.svd import SVDRecommender

# Import configuration
import config

def load_data():
    """Load and preprocess the data."""
    print("Loading data...")
    
    # Load user reviews
    user_reviews_df = load_user_reviews()
    print(f"Loaded {len(user_reviews_df)} user reviews")
    
    # Load user items
    user_items_df = load_user_items()
    print(f"Loaded {len(user_items_df)} user-item interactions")
    
    # Load steam games
    steam_games_df = load_steam_games()
    print(f"Loaded {len(steam_games_df)} steam games")
    
    # Load bundle data
    bundle_df = load_bundle_data()
    print(f"Loaded {len(bundle_df)} bundle items")
    
    return user_reviews_df, user_items_df, steam_games_df, bundle_df

def train_recommenders(user_reviews_df, user_items_df, steam_games_df, bundle_df):
    """Train all recommendation models."""
    recommenders = {}
    
    print("\nTraining recommendation models...")
    
    # Non-personalized recommender (popularity-based)
    print("Training non-personalized recommender...")
    start_time = time.time()
    non_personalized = NonPersonalizedRecommender(method='popularity')
    non_personalized.fit(user_items_df, user_reviews_df)
    print(f"Done in {time.time() - start_time:.2f}s")
    recommenders['non_personalized'] = non_personalized
    
    # User-based collaborative filtering
    print("Training user-based collaborative filtering...")
    start_time = time.time()
    user_cf = UserBasedCF(k=config.USER_CF_K_NEIGHBORS, sim_method='cosine', interaction_type='ownership')
    user_cf.fit(user_items_df)
    print(f"Done in {time.time() - start_time:.2f}s")
    recommenders['user_cf'] = user_cf
    
    # Item-based collaborative filtering
    print("Training item-based collaborative filtering...")
    start_time = time.time()
    item_cf = ItemBasedCF(k=config.ITEM_CF_K_NEIGHBORS, sim_method='cosine', interaction_type='ownership')
    item_cf.fit(user_items_df)
    print(f"Done in {time.time() - start_time:.2f}s")
    recommenders['item_cf'] = item_cf
    
    # Content-based recommender
    print("Training content-based recommender...")
    start_time = time.time()
    content_based = ContentBasedRecommender(
        tags_weight=config.CONTENT_TAGS_WEIGHT,
        specs_weight=config.CONTENT_SPECS_WEIGHT,
        genres_weight=config.CONTENT_GENRES_WEIGHT
    )
    content_based.fit(steam_games_df, user_items_df, bundle_df)
    print(f"Done in {time.time() - start_time:.2f}s")
    recommenders['content_based'] = content_based
    
    # Matrix factorization
    print("Training matrix factorization recommender...")
    start_time = time.time()
    matrix_fact = MatrixFactorizationRecommender(
        n_factors=config.MF_N_FACTORS,
        n_epochs=config.MF_N_EPOCHS,
        learning_rate=config.MF_LEARNING_RATE,
        regularization=config.MF_REGULARIZATION
    )
    matrix_fact.fit(user_items_df, interaction_type='playtime', verbose=False)
    print(f"Done in {time.time() - start_time:.2f}s")
    recommenders['matrix_factorization'] = matrix_fact
    
    # SVD
    print("Training SVD recommender...")
    start_time = time.time()
    svd = SVDRecommender(n_factors=config.SVD_N_FACTORS)
    svd.fit(user_items_df, interaction_type='playtime', verbose=False)
    print(f"Done in {time.time() - start_time:.2f}s")
    recommenders['svd'] = svd
    
    return recommenders

def get_recommendations(recommenders, user_id, n=10):
    """
    Get recommendations from all algorithms for a specific user.
    
    Parameters:
    -----------
    recommenders : dict
        Dictionary of recommender objects
    user_id : str
        ID of the target user
    n : int
        Number of recommendations to generate
    
    Returns:
    --------
    recommendations : dict
        Dictionary of recommendation DataFrames for each algorithm
    """
    results = {}
    
    for name, recommender in recommenders.items():
        try:
            recs = recommender.recommend(user_id, n=n)
            results[name] = recs
        except Exception as e:
            print(f"Error getting recommendations from {name}: {e}")
            results[name] = pd.DataFrame(columns=['item_id', 'score', 'item_name'])
    
    return results

def get_group_recommendations(recommenders, user_ids, n=10):
    """
    Get group recommendations from algorithms that support it.
    
    Parameters:
    -----------
    recommenders : dict
        Dictionary of recommender objects
    user_ids : list
        List of user IDs in the group
    n : int
        Number of recommendations to generate
    
    Returns:
    --------
    recommendations : dict
        Dictionary of recommendation DataFrames for each algorithm
    """
    results = {}
    
    # Get common games owned by all users
    common_games = None
    if 'non_personalized' in recommenders:
        common_games = recommenders['non_personalized'].get_common_games(user_ids)
        results['common_games'] = common_games
    
    # Get recommendations from algorithms that support group recommendations
    group_algos = ['user_cf', 'content_based', 'matrix_factorization', 'svd']
    
    for name in group_algos:
        if name in recommenders:
            try:
                recs = recommenders[name].recommend_for_group(user_ids, n=n)
                results[name] = recs
            except Exception as e:
                print(f"Error getting group recommendations from {name}: {e}")
                results[name] = pd.DataFrame(columns=['item_id', 'score', 'item_name'])
    
    return results

def evaluate_models(recommenders, user_items_df, test_ratio=0.2, n_users=100):
    """
    Evaluate all recommendation models.
    
    Parameters:
    -----------
    recommenders : dict
        Dictionary of recommender objects
    user_items_df : pandas.DataFrame
        DataFrame with user-item interactions
    test_ratio : float
        Proportion of data to use for testing
    n_users : int
        Number of users to evaluate on
    
    Returns:
    --------
    results : dict
        Dictionary of evaluation metrics for each algorithm
    """
    print("\nEvaluating recommendation models...")
    
    # Get list of users with enough items
    user_counts = user_items_df.groupby('user_id').size()
    test_users = user_counts[user_counts >= 10].index.tolist()
    
    # Limit to n_users
    if len(test_users) > n_users:
        test_users = test_users[:n_users]
    
    print(f"Evaluating on {len(test_users)} users")
    
    # Split data into train and test sets
    train_data = {}
    test_data = {}
    
    for user_id in test_users:
        # Get user's items
        user_items = user_items_df[user_items_df['user_id'] == user_id]
        
        # Randomly split
        train = user_items.sample(frac=1-test_ratio)
        test = user_items.drop(train.index)
        
        train_data[user_id] = train
        test_data[user_id] = test
    
    # Combine into DataFrames
    train_df = pd.concat(train_data.values())
    test_df = pd.concat(test_data.values())
    
    # Generate recommendations for each user
    all_recommendations = {}
    
    for name, recommender in recommenders.items():
        user_recommendations = {}
        
        # Skip SVD for evaluation (too slow for this demo)
        if name in ['svd']:
            continue
        
        print(f"Generating recommendations for {name}...")
        start_time = time.time()
        
        for user_id in test_users:
            # Get items from training set for this user
            train_items = train_data[user_id]['item_id'].tolist()
            
            try:
                # For content-based, adapt the API
                if name == 'content_based':
                    recs = recommender.recommend(None, n=10, exclude_items=train_items, items_to_use=train_items)
                elif name == 'non_personalized':
                    recs = recommender.recommend(n=10, exclude_items=train_items)
                else:
                    recs = recommender.recommend(user_id, n=10, exclude_items=train_items)
                
                user_recommendations[user_id] = recs['item_id'].tolist()
            except Exception as e:
                print(f"Error getting recommendations for user {user_id} from {name}: {e}")
                user_recommendations[user_id] = []
        
        all_recommendations[name] = user_recommendations
        print(f"Done in {time.time() - start_time:.2f}s")
    
    # Evaluate recommendations
    eval_results = {}
    
    for name, recommendations in all_recommendations.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_recommendations(recommendations, test_df)
        eval_results[name] = metrics
    
    return eval_results

def display_recommendations(recommendations, title):
    """Display recommendations in a nice format."""
    print(f"\n{title}")
    
    if isinstance(recommendations, dict):
        for algo_name, recs in recommendations.items():
            if algo_name == 'common_games':
                print(f"\nCommon Games:")
                if len(recs) > 0:
                    print(tabulate(recs[['item_name', 'playtime_forever']], 
                                  headers=['Game', 'Avg. Playtime'], tablefmt='pretty'))
                else:
                    print("No common games found.")
            else:
                print(f"\n{algo_name.replace('_', ' ').title()} Recommendations:")
                if len(recs) > 0:
                    # Format the score column
                    recs_display = recs.copy()
                    if 'score' in recs_display.columns:
                        recs_display['score'] = recs_display['score'].apply(lambda x: f"{x:.3f}")
                    
                    print(tabulate(recs_display[['item_name', 'score'] if 'score' in recs_display.columns else ['item_name']], 
                                  headers=['Game', 'Score'] if 'score' in recs_display.columns else ['Game'], 
                                  tablefmt='pretty'))
                else:
                    print("No recommendations found.")
    else:
        if len(recommendations) > 0:
            print(tabulate(recommendations[['item_name', 'score'] if 'score' in recommendations.columns else ['item_name']], 
                          headers=['Game', 'Score'] if 'score' in recommendations.columns else ['Game'], 
                          tablefmt='pretty'))
        else:
            print("No recommendations found.")

def display_evaluation_results(results):
    """Display evaluation results in a nice format."""
    print("\nEvaluation Results:")
    
    # Convert to DataFrame for easier display
    metrics = []
    for algo_name, algo_metrics in results.items():
        algo_metrics['Algorithm'] = algo_name.replace('_', ' ').title()
        metrics.append(algo_metrics)
    
    metrics_df = pd.DataFrame(metrics)
    
    # Reorder columns
    col_order = ['Algorithm', 'precision@5', 'recall@5', 'precision@10', 'recall@10', 'map', 'ndcg@10']
    metrics_df = metrics_df[[col for col in col_order if col in metrics_df.columns]]
    
    # Format to 3 decimal places
    for col in metrics_df.columns:
        if col != 'Algorithm':
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.3f}")
    
    print(tabulate(metrics_df, headers='keys', tablefmt='pretty'))

def recommend_games_for_friends(recommenders, user_ids, n=10):
    """
    Generate recommendations for a group of friends to play together.
    
    Parameters:
    -----------
    recommenders : dict
        Dictionary of recommender objects
    user_ids : list
        List of user IDs in the group
    n : int
        Number of recommendations to generate
    
    Returns:
    --------
    recommendations : dict
        Dictionary of recommendation DataFrames for each algorithm
    """
    print(f"\nGenerating recommendations for {len(user_ids)} friends...")
    
    # Get group recommendations
    group_recs = get_group_recommendations(recommenders, user_ids, n=n)
    
    return group_recs

def find_similar_games(recommenders, item_id, n=10):
    """
    Find games similar to a specific game.
    
    Parameters:
    -----------
    recommenders : dict
        Dictionary of recommender objects
    item_id : str
        ID of the target game
    n : int
        Number of similar games to find
    
    Returns:
    --------
    similar_games : dict
        Dictionary of similar games DataFrames for each algorithm
    """
    results = {}
    
    # Get similar games from item-based CF
    if 'item_cf' in recommenders:
        try:
            similar_items = recommenders['item_cf'].get_similar_items(item_id, n=n)
            results['item_cf'] = similar_items
        except Exception as e:
            print(f"Error getting similar games from item-based CF: {e}")
    
    # Get similar games from content-based
    if 'content_based' in recommenders:
        try:
            similar_items = recommenders['content_based'].get_similar_items(item_id, n=n)
            results['content_based'] = similar_items
        except Exception as e:
            print(f"Error getting similar games from content-based: {e}")
    
    return results

def interactive_mode(recommenders, user_items_df):
    """Run the recommendation system in interactive mode."""
    print("\n===== Steam Game Recommendation System =====")
    print("What would you like to do?")
    
    while True:
        print("\nOptions:")
        print("1. Get recommendations for a single user")
        print("2. Get recommendations for a group of friends")
        print("3. Find games similar to a specific game")
        print("4. Compare recommendation algorithms")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            # Get recommendations for a single user
            user_id = input("Enter user ID: ")
            
            # Check if user exists
            if user_id not in user_items_df['user_id'].values:
                print(f"User {user_id} not found in the dataset.")
                continue
            
            # Get recommendations
            user_recs = get_recommendations(recommenders, user_id, n=10)
            display_recommendations(user_recs, f"Recommendations for user {user_id}")
            
        elif choice == '2':
            # Get recommendations for a group of friends
            user_ids_input = input("Enter user IDs separated by commas: ")
            user_ids = [uid.strip() for uid in user_ids_input.split(',')]
            
            # Check if users exist
            valid_user_ids = [uid for uid in user_ids if uid in user_items_df['user_id'].values]
            if not valid_user_ids:
                print("None of the specified users found in the dataset.")
                continue
            
            if len(valid_user_ids) < len(user_ids):
                print(f"Warning: Only {len(valid_user_ids)} out of {len(user_ids)} users found.")
            
            # Get group recommendations
            group_recs = recommend_games_for_friends(recommenders, valid_user_ids, n=10)
            display_recommendations(group_recs, f"Recommendations for {len(valid_user_ids)} friends")
            
        elif choice == '3':
            # Find games similar to a specific game
            game_name = input("Enter game name (or part of it): ")
            
            # Find matching games
            matching_games = user_items_df[user_items_df['item_name'].str.contains(game_name, case=False, na=False)]
            matching_games = matching_games[['item_id', 'item_name']].drop_duplicates()
            
            if len(matching_games) == 0:
                print(f"No games found matching '{game_name}'.")
                continue
            
            # Display matching games
            print("\nMatching games:")
            for i, (_, row) in enumerate(matching_games.iterrows(), 1):
                print(f"{i}. {row['item_name']} (ID: {row['item_id']})")
            
            # Let user select a game
            game_choice = input("\nEnter the number of the game you want to find similar games for: ")
            try:
                game_idx = int(game_choice) - 1
                if game_idx < 0 or game_idx >= len(matching_games):
                    print("Invalid choice.")
                    continue
                
                selected_game = matching_games.iloc[game_idx]
                item_id = selected_game['item_id']
                item_name = selected_game['item_name']
                
                # Get similar games
                similar_games = find_similar_games(recommenders, item_id, n=10)
                display_recommendations(similar_games, f"Games similar to {item_name}")
                
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
            
        elif choice == '4':
            # Compare recommendation algorithms
            print("\nThis will evaluate the recommendation algorithms on a sample of users.")
            print("This may take some time. Do you want to continue?")
            confirm = input("Enter 'y' to continue: ")
            
            if confirm.lower() == 'y':
                # Evaluate algorithms
                eval_results = evaluate_models(recommenders, user_items_df, test_ratio=0.2, n_users=50)
                display_evaluation_results(eval_results)
            
        elif choice == '5':
            # Exit
            print("\nThank you for using the Steam Game Recommendation System!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Steam Game Recommendation System")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate recommendation algorithms")
    parser.add_argument("--user", type=str, help="Get recommendations for a specific user")
    parser.add_argument("--group", type=str, help="Get recommendations for a group of users (comma-separated)")
    parser.add_argument("--game", type=str, help="Find games similar to a specific game ID")
    
    args = parser.parse_args()
    
    # Load data
    user_reviews_df, user_items_df, steam_games_df, bundle_df = load_data()
    
    # Train recommenders
    recommenders = train_recommenders(user_reviews_df, user_items_df, steam_games_df, bundle_df)
    
    # Handle different modes
    if args.interactive:
        interactive_mode(recommenders, user_items_df)
    elif args.evaluate:
        eval_results = evaluate_models(recommenders, user_items_df)
        display_evaluation_results(eval_results)
    elif args.user:
        user_recs = get_recommendations(recommenders, args.user)
        display_recommendations(user_recs, f"Recommendations for user {args.user}")
    elif args.group:
        user_ids = [uid.strip() for uid in args.group.split(',')]
        group_recs = recommend_games_for_friends(recommenders, user_ids)
        display_recommendations(group_recs, f"Recommendations for {len(user_ids)} friends")
    elif args.game:
        similar_games = find_similar_games(recommenders, args.game)
        game_name = user_items_df[user_items_df['item_id'] == args.game]['item_name'].iloc[0] \
            if args.game in user_items_df['item_id'].values else args.game
        display_recommendations(similar_games, f"Games similar to {game_name}")
    else:
        # Default to interactive mode
        interactive_mode(recommenders, user_items_df)

if __name__ == "__main__":
    main()