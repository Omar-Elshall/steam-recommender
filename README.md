# Steam Game Recommendation System

A recommendation system that helps gamers find the best games to play with friends based on their Steam game libraries. This project implements various recommendation algorithms to suggest games that multiple friends own and enjoy.

## Project Overview

This recommendation system implements the following algorithms:

1. **Non-personalized recommendations**: Based on game popularity and playtime statistics
2. **User-based collaborative filtering**: Recommends games based on similar users' preferences
3. **Item-based collaborative filtering**: Recommends games similar to ones users already own and play
4. **Content-based recommendations**: Recommends games based on game attributes (tags, genres, specs)
5. **Matrix factorization**: Uses latent factor analysis to discover patterns in user-game interactions
6. **Singular Value Decomposition (SVD)**: Reduces dimensionality to capture latent factors

The system focuses on helping friends find games they can play together, taking into account the games they already own and their preferences.

## Dataset

The project uses the Steam Video Game and Bundle Data from Kaggle, which includes:

- User game ownership data
- Game metadata (tags, genres, etc.)
- User reviews
- Bundle information

## Project Structure

```
recommendation_system/
├── data/                       # For your datasets
├── algorithms/                 # Different recommendation algorithms
│   ├── __init__.py
│   ├── non_personalized.py     # Non-personalized recommendations
│   ├── item_cf.py              # Item-based collaborative filtering
│   ├── user_cf.py              # User-based collaborative filtering
│   ├── content_based.py        # Content-based recommendations
│   ├── matrix_factorization.py # Matrix factorization
│   └── svd.py                  # Singular Value Decomposition
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── data_loader.py          # Functions to load and preprocess data
│   ├── similarity.py           # Similarity functions (cosine, Pearson, etc.)
│   └── evaluation.py           # Metrics and evaluation functions
├── config.py                   # Configuration parameters
├── main.py                     # Main script to run the algorithms
└── requirements.txt            # Dependencies
```

## Features

1. **Single-user recommendations**: Get personalized game recommendations for individual users
2. **Group recommendations**: Find games that a group of friends might enjoy playing together
3. **Common games**: Identify games that multiple friends already own
4. **Similar games**: Find games similar to a specific game using different similarity metrics
5. **Algorithm comparison**: Evaluate and compare the performance of different recommendation algorithms

## Installation and Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd recommendation_system
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Place the dataset files in the `data/` directory:

- `australian_user_reviews.json`
- `australian_user_items.json`
- `bundle_data.json`
- `steam_games.json`

## Usage

### Interactive Mode

Run the system in interactive mode:

```bash
python main.py --interactive
```

This allows you to:

- Get recommendations for a single user
- Get recommendations for a group of friends
- Find games similar to a specific game
- Compare recommendation algorithms

### Command Line Options

Get recommendations for a specific user:

```bash
python main.py --user <user_id>
```

Get recommendations for a group of friends:

```bash
python main.py --group <user_id1>,<user_id2>,<user_id3>
```

Find games similar to a specific game:

```bash
python main.py --game <game_id>
```

Evaluate and compare recommendation algorithms:

```bash
python main.py --evaluate
```

## Evaluation Metrics

The system evaluates recommendations using several metrics:

- Precision@k and Recall@k
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Coverage
- Diversity
- Novelty

## Contributors

- Omar Elshall - B00096779
- Abdullah Alshetiwi - B00096197
- Ezzeddine Diab - B00094815

## Acknowledgements

This project uses the Steam Video Game and Bundle Data from Kaggle, provided by Pypi Ahmad.
