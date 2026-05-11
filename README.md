# Steam Game Recommender

> A multi-algorithm recommendation system benchmarking 8 classical and deep-learning approaches on Steam user-game ownership data, with a Streamlit demo UI.

Built as the course project for **CMP 49412 — Intelligent Recommendation Systems** at the American University of Sharjah. Extended post-course with PySpark distributed ALS, two TensorFlow deep-learning content-based models, and an interactive Streamlit interface.

---

## Algorithms

| # | Family | Algorithm | Module |
|---|---|---|---|
| 1 | Non-personalized | Popularity ranking + Bayesian scoring | `processing/non_personalized_trending.py` |
| 2 | Memory-based CF | User-based collaborative filtering (cosine) | `processing/user_item_based_cf.py` |
| 3 | Memory-based CF | Item-based collaborative filtering (cosine) | `processing/user_item_based_cf.py` |
| 4 | Content-based | TF-IDF + genre/tag multi-hot | `processing/content_based_filtering.py`, `processing/text_based_filtering.py` |
| 5 | Model-based CF | ALS matrix factorization — in-house NumPy/CuPy | `processing/ALS_matrix_factorization.py` |
| 6 | Model-based CF | ALS matrix factorization — PySpark, distributed | `processing/pyspark_als.py` |
| 7 | Deep learning | Autoencoder content-based filtering (TensorFlow) | `processing/tf_autoencoder.py` |
| 8 | Deep learning | Two-tower neural retrieval (TensorFlow) | `processing/tf_two_tower.py` |

The two TensorFlow architectures (autoencoder, two-tower) follow the patterns taught in CMP 49412 Lesson 11 (Deep Learning for Content-Based Filtering).

## Demo UI

```bash
streamlit run app.py
```

The Streamlit app lets you pick a user, an algorithm, and the number of recommendations to surface. It also renders per-algorithm explanations and (when run after `--evaluate`) a comparison table of Precision@k, Recall@k, F1, and RMSE across all 8 models.

## Dataset

Steam Australian-user ownership and review dataset from Kaggle (Pypi Ahmad):

- `australian_users_items.json` — user → owned-games (the implicit-feedback signal)
- `australian_user_reviews.json` — user reviews + sentiment
- `steam_games.json` — game metadata (tags, genres, prices, release dates)
- `bundle_data.json` — game bundle metadata

Place these in `data/` before running anything.

## Stack

- **Python 3.10+**
- **NumPy**, **Pandas**, **scikit-learn** — classical ML and data prep
- **CuPy** — GPU-accelerated matrix ops for the in-house ALS (CUDA 12)
- **PySpark 3.5** — distributed ALS via `pyspark.ml.recommendation.ALS`
- **TensorFlow 2.18 / Keras** — autoencoder and two-tower models
- **Streamlit** — interactive UI

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For CuPy: requires CUDA 12 toolkit. If on CPU-only, comment out the CuPy line in `requirements.txt` and either disable CuPy paths or run with `CUDA_VISIBLE_DEVICES=""` to fall back to NumPy.

## Usage

### Interactive demo

```bash
streamlit run app.py
```

### Train and run from CLI

```bash
# Train in-house ALS
python main.py

# Train PySpark ALS
python -m processing.pyspark_als

# Train TensorFlow autoencoder
python -m processing.tf_autoencoder

# Train two-tower model
python -m processing.tf_two_tower

# Evaluate all algorithms
python main.py --evaluate

# Get recommendations for a user
python main.py --user <user_id>

# Group recommendations for friends
python main.py --group <user_id1>,<user_id2>,<user_id3>
```

## Evaluation

A 90/10 train-test split, stratified by user. For each test user, hold out 10% of their owned games and measure:

- **Precision@k** — fraction of top-k recommendations that are in the held-out set
- **Recall@k** — fraction of held-out games that appear in top-k
- **F1@k** — harmonic mean of precision and recall
- **RMSE** — root mean squared error of predicted ownership probability vs. ground-truth labels (applied to score-producing models: in-house ALS, PySpark ALS, two-tower)

Run with:

```bash
python main.py --evaluate
```

## Repository layout

```
.
├── app.py                              # Streamlit UI
├── main.py                             # CLI entry point
├── requirements.txt
├── data/                               # Place Kaggle datasets here (gitignored)
└── processing/
    ├── config.py                       # Shared config (data paths, hyperparams)
    ├── utils.py                        # Recommendation helpers used across algorithms
    ├── non_personalized_trending.py    # (1) Popularity
    ├── user_item_based_cf.py           # (2,3) User/item-based CF
    ├── content_based_filtering.py      # (4) Content-based
    ├── text_based_filtering.py         # (4) TF-IDF text
    ├── ALS_matrix_factorization.py     # (5) In-house ALS
    ├── pyspark_als.py                  # (6) PySpark ALS
    ├── tf_autoencoder.py               # (7) Autoencoder
    ├── tf_two_tower.py                 # (8) Two-tower
    └── evaluate_models.py              # Metrics: Precision@k, Recall@k, F1, RMSE
```

## License

MIT.

## Acknowledgements

Course materials and the two reference notebooks (autoencoder, two-tower content-based filtering) are from CMP 49412 at the American University of Sharjah. Kaggle dataset by Pypi Ahmad.
