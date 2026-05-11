"""
One-shot training orchestrator for the 4 missing demo artifacts.

Run with: python scripts/train_all.py

Trains in this order (fastest first so the UI gains capabilities quickly):
  1. Item similarity matrix (top-50 per game) — derived from V_matrix already in ALS pkl
  2. TF autoencoder — content-based latent space from game_tags
  3. PySpark ALS — distributed implicit-feedback factorization
  4. TF two-tower — neural retrieval

Logs to /tmp/steam-train.log as we go.
"""

import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "processing"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/steam-train.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("train_all")

PROCESSED = ROOT / "processing" / "processedData"


def step(name):
    log.info("=" * 70)
    log.info("STEP: %s", name)
    log.info("=" * 70)


# ----------------------------------------------------------------------
# 1. Item similarity matrix from V_matrix (ALS item embeddings)
# ----------------------------------------------------------------------
def train_item_similarity():
    step("1/4 Item similarity matrix (top-50 per game from ALS V_matrix)")
    t0 = time.time()

    with open(PROCESSED / "V_matrix.pkl", "rb") as f:
        V = pickle.load(f)  # (n_games, k)
    with open(PROCESSED / "game_idx.pkl", "rb") as f:
        game_idx = pickle.load(f)  # game_name -> int index

    log.info("V_matrix: %s, %d games indexed", V.shape, len(game_idx))

    # L2 normalize once so dot product == cosine similarity
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V_norm = V / norms

    idx_to_game = {i: g for g, i in game_idx.items()}
    n_games = V_norm.shape[0]

    similarity_matrix = {}
    top_k = 50
    batch = 500

    for start in range(0, n_games, batch):
        end = min(start + batch, n_games)
        sims = V_norm[start:end] @ V_norm.T  # (batch, n_games)
        for local_i in range(end - start):
            global_i = start + local_i
            sims[local_i, global_i] = -np.inf
            top_indices = np.argpartition(-sims[local_i], top_k)[:top_k]
            top_indices = top_indices[np.argsort(-sims[local_i, top_indices])]
            game_name = idx_to_game[global_i]
            similarity_matrix[game_name] = {
                idx_to_game[j]: float(sims[local_i, j]) for j in top_indices
            }
        if start % 5000 == 0:
            log.info("  progress: %d / %d games", end, n_games)

    out_path = PROCESSED / "game_similarity_matrix.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(similarity_matrix, f)
    log.info("Saved %s (%.1f MB) in %.1fs", out_path, out_path.stat().st_size / 1e6, time.time() - t0)


# ----------------------------------------------------------------------
# 2. TF autoencoder on game_tags matrix
# ----------------------------------------------------------------------
def train_autoencoder():
    step("2/4 TF autoencoder on game_tags (32k games × 339 tags)")
    t0 = time.time()

    with open(PROCESSED / "game_tags.pkl", "rb") as f:
        df = pickle.load(f)

    log.info("game_tags: %s", df.shape)
    X = df.values.astype("float32")

    from processing.tf_autoencoder import train_autoencoder as tf_train

    autoencoder, encoder, hist = tf_train(
        X, latent_dim=64, hidden=256, epochs=10, batch_size=512, save=True
    )
    log.info("Final loss: %.4f", hist["loss"][-1])
    log.info("Done in %.1fs", time.time() - t0)


# ----------------------------------------------------------------------
# 3. PySpark ALS on user2game (implicit feedback)
# ----------------------------------------------------------------------
def train_pyspark_als():
    step("3/4 PySpark ALS (distributed implicit-feedback factorization)")
    t0 = time.time()

    with open(PROCESSED / "user2game_dict.pkl", "rb") as f:
        user2game = pickle.load(f)
    log.info("user2game: %d users", len(user2game))

    from processing.pyspark_als import train_pyspark_als as ps_train

    ps_train(user2game, rank=32, reg_param=0.1, max_iter=10, alpha=40.0, save=True)
    log.info("Done in %.1fs", time.time() - t0)


# ----------------------------------------------------------------------
# 4. TF two-tower (item features = ALS V_matrix; user features = avg of owned)
#
# Why V_matrix instead of game_tags: V_matrix is keyed by game NAME (same
# keyspace as user2game), while game_tags is keyed by Steam app ID (float).
# Using V_matrix avoids a name<->id translation step.
# ----------------------------------------------------------------------
def train_two_tower():
    step("4/4 TF two-tower (user centroid x ALS V_matrix item embeddings)")
    t0 = time.time()

    with open(PROCESSED / "V_matrix.pkl", "rb") as f:
        V = pickle.load(f).astype("float32")  # (n_games, 100)
    with open(PROCESSED / "game_idx.pkl", "rb") as f:
        game_idx = pickle.load(f)  # game_name -> int index
    with open(PROCESSED / "user2game_dict.pkl", "rb") as f:
        user2game = pickle.load(f)

    log.info("V_matrix item embeddings: %s, n_users: %d", V.shape, len(user2game))

    users = sorted(user2game.keys())[:5000]
    user_to_idx = {u: i for i, u in enumerate(users)}

    n_features = V.shape[1]
    user_features = np.zeros((len(users), n_features), dtype="float32")
    positive_pairs = []
    for u in users:
        u_int = user_to_idx[u]
        game_indices = [game_idx[g] for g in user2game[u] if g in game_idx]
        if not game_indices:
            continue
        user_features[u_int] = V[game_indices].mean(axis=0)
        for g_int in game_indices:
            positive_pairs.append((u_int, g_int))

    item_features = V
    log.info("Positives: %d, users with games: %d", len(positive_pairs), len(users))

    from processing.tf_two_tower import train_two_tower as tt_train

    tt_train(
        user_features,
        item_features,
        positive_pairs,
        latent_dim=64,
        n_negatives_per_positive=4,
        epochs=5,
        batch_size=1024,
        save=True,
    )
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    overall = time.time()
    for fn in (train_item_similarity, train_autoencoder, train_pyspark_als, train_two_tower):
        try:
            fn()
        except Exception as e:
            log.exception("%s failed: %s", fn.__name__, e)
    log.info("ALL DONE in %.1fs", time.time() - overall)
