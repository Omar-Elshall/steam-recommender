import numpy as np
import pandas as pd
import cupy as cp
import time
from config import game_tags_data

def compute_full_similarity_matrix_gpu():
    """
    Computes the full Jaccard similarity matrix for all games with optimized GPU processing.
    """
    start_time = time.time()
    
    # Get all game IDs
    all_game_ids = game_tags_data['id'].values
    num_games = len(all_game_ids)
    print(f"Starting optimized computation for {num_games} games...")
    
    # Get all game tags as a feature matrix
    all_game_tags = game_tags_data.drop('id', axis=1).values
    
    # Transfer to GPU once
    all_game_tags_gpu = cp.asarray(all_game_tags)
    
    # Initialize matrix in smaller blocks to avoid memory issues
    block_size = 1000  # Process in 1000×1000 game blocks
    num_blocks = (num_games + block_size - 1) // block_size
    
    # Empty matrix to store results
    similarity_matrix = np.zeros((num_games, num_games))
    
    # Process blocks of the matrix
    for block_i in range(num_blocks):
        i_start = block_i * block_size
        i_end = min((block_i + 1) * block_size, num_games)
        i_size = i_end - i_start
        
        # Create GPU arrays for intersection and union calculations
        games_i = all_game_tags_gpu[i_start:i_end]
        
        print(f"Processing block {block_i+1}/{num_blocks} (games {i_start}-{i_end-1})...")
        block_start = time.time()
        
        for block_j in range(block_i, num_blocks):
            j_start = block_j * block_size
            j_end = min((block_j + 1) * block_size, num_games)
            j_size = j_end - j_start
            
            games_j = all_game_tags_gpu[j_start:j_end]
            
            # VECTORIZED COMPUTATION - this is the key optimization
            # Reshape for broadcasting: (i_size, 1, tags) and (1, j_size, tags)
            games_i_reshaped = games_i.reshape(i_size, 1, -1)
            games_j_reshaped = games_j.reshape(1, j_size, -1)
            
            # Calculate intersection and union for all pairs at once
            intersection = cp.sum(cp.logical_and(games_i_reshaped, games_j_reshaped), axis=2)
            union = cp.sum(cp.logical_or(games_i_reshaped, games_j_reshaped), axis=2)
            
            # Calculate Jaccard similarity (avoiding division by zero)
            jaccard = cp.zeros_like(intersection, dtype=cp.float32)
            valid_idx = union > 0
            jaccard[valid_idx] = intersection[valid_idx] / union[valid_idx]
            
            # Transfer results to CPU
            jaccard_np = cp.asnumpy(jaccard)
            
            # Fill the similarity matrix (both upper and lower triangles)
            similarity_matrix[i_start:i_end, j_start:j_end] = jaccard_np
            if i_start != j_start:  # Don't duplicate the diagonal blocks
                similarity_matrix[j_start:j_end, i_start:i_end] = jaccard_np.T
            
            # Free memory
            del jaccard
            cp.get_default_memory_pool().free_all_blocks()
            
        block_end = time.time()
        print(f"  Block completed in {block_end - block_start:.2f} seconds")
    
    # Convert to DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=all_game_ids, columns=all_game_ids)
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds. Matrix shape: {similarity_df.shape}")
    
    return similarity_df

def get_top_k_similar_games_from_matrix(similarity_matrix, game_id, k):
    """
    Returns the top k most similar games to the specified game.
    """
    if game_id not in similarity_matrix.index:
        raise ValueError(f"Game with ID {game_id} not found in the similarity matrix")
    
    # Get similarities for the specified game
    similarities = similarity_matrix.loc[game_id].reset_index()
    similarities.columns = ['game_id', 'similarity']
    
    # Remove self-similarity
    similarities = similarities[similarities['game_id'] != game_id]
    
    # Sort by similarity and return top k
    similarities = similarities.sort_values('similarity', ascending=False)
    return similarities.head(k)

# Example usage:
similarity_matrix = compute_full_similarity_matrix_gpu()

# Get recommendations
top_similar = get_top_k_similar_games_from_matrix(similarity_matrix, 371660, 10)
print("\nTop 10 similar games to game 371660:")
print(top_similar)