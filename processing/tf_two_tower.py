"""
TensorFlow two-tower neural network for content-based filtering on Steam games.

Architecture:
  user_tower: user_features -> Dense layers -> user_embedding (latent_dim)
  item_tower: item_features -> Dense layers -> item_embedding (latent_dim)
  score = dot(user_embedding, item_embedding) -> sigmoid -> p(own)

Trained on positive interactions (user owns game) with random negative sampling.
This is the canonical two-tower retrieval architecture used at scale at
Google (YouTube recommender), Netflix, etc.

Adapted from the CMP 49412 (Intelligent Recommendation Systems) course
notebook on two-tower content-based filtering.
"""

import os
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


PROCESSED_DIR = "processing/processedData"
MODEL_PATH = os.path.join(PROCESSED_DIR, "tf_two_tower.keras")


def build_tower(input_dim: int, latent_dim: int, name: str) -> keras.Model:
    """Single tower: input -> 256 -> 128 -> latent_dim, ReLU + dropout."""
    inp = keras.Input(shape=(input_dim,), name=f"{name}_input")
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    embedding = layers.Dense(latent_dim, activation=None, name=f"{name}_embedding")(x)
    # L2 normalize for cosine-similarity style training
    embedding = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(embedding)
    return keras.Model(inp, embedding, name=f"{name}_tower")


def build_two_tower(user_dim: int, item_dim: int, latent_dim: int = 64) -> Tuple[keras.Model, keras.Model, keras.Model]:
    """Returns (full_model, user_tower, item_tower)."""
    user_tower = build_tower(user_dim, latent_dim, "user")
    item_tower = build_tower(item_dim, latent_dim, "item")

    u_in = keras.Input(shape=(user_dim,), name="user_features")
    i_in = keras.Input(shape=(item_dim,), name="item_features")
    u_emb = user_tower(u_in)
    i_emb = item_tower(i_in)

    # Dot product → similarity logit
    score = layers.Dot(axes=1)([u_emb, i_emb])
    prob = layers.Activation("sigmoid", name="ownership_probability")(score)

    full = keras.Model([u_in, i_in], prob, name="two_tower")
    full.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return full, user_tower, item_tower


def sample_negatives(
    user_features: np.ndarray,
    item_features: np.ndarray,
    positive_pairs: List[Tuple[int, int]],
    n_negatives_per_positive: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build training tensors with random negative sampling.

    For each positive (u, i_owned), draw `n_negatives_per_positive` random items
    that the user does NOT own.
    """
    rng = np.random.default_rng(seed)
    n_users, _ = user_features.shape
    n_items, _ = item_features.shape

    pos_set = {(u, i) for u, i in positive_pairs}

    pos_u, pos_i, labels = [], [], []
    for u, i in positive_pairs:
        pos_u.append(u)
        pos_i.append(i)
        labels.append(1.0)

        attempts = 0
        added = 0
        while added < n_negatives_per_positive and attempts < n_negatives_per_positive * 10:
            neg_i = int(rng.integers(0, n_items))
            attempts += 1
            if (u, neg_i) in pos_set:
                continue
            pos_u.append(u)
            pos_i.append(neg_i)
            labels.append(0.0)
            added += 1

    pos_u = np.array(pos_u, dtype=np.int64)
    pos_i = np.array(pos_i, dtype=np.int64)
    labels = np.array(labels, dtype=np.float32)

    return user_features[pos_u], item_features[pos_i], labels


def train_two_tower(
    user_features: np.ndarray,
    item_features: np.ndarray,
    positive_pairs: List[Tuple[int, int]],
    latent_dim: int = 64,
    n_negatives_per_positive: int = 4,
    epochs: int = 10,
    batch_size: int = 512,
    save: bool = True,
):
    """Train the two-tower model end to end."""
    print(f"Two-tower training: {user_features.shape[0]} users, {item_features.shape[0]} items, "
          f"{len(positive_pairs)} positives × {n_negatives_per_positive} negs")

    X_user, X_item, y = sample_negatives(
        user_features, item_features, positive_pairs, n_negatives_per_positive
    )
    print(f"Training tensor: {X_user.shape[0]} samples (positives + sampled negatives)")

    full, user_tower, item_tower = build_two_tower(
        user_features.shape[1], item_features.shape[1], latent_dim
    )

    history = full.fit(
        [X_user, X_item],
        y,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        verbose=2,
    )

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        full.save(MODEL_PATH)
        user_tower.save(MODEL_PATH.replace(".keras", "_user.keras"))
        item_tower.save(MODEL_PATH.replace(".keras", "_item.keras"))
        print(f"Saved two-tower model to {MODEL_PATH}")

    return full, user_tower, item_tower, history.history


def recommend_for_user(
    user_tower: keras.Model,
    item_tower: keras.Model,
    user_feature_vector: np.ndarray,
    item_features: np.ndarray,
    game_ids: List[str],
    owned_game_ids: List[str],
    n: int = 10,
) -> List[Tuple[str, float]]:
    """Score all items for a user and return top-N excluding already-owned games."""
    u_emb = user_tower.predict(user_feature_vector.reshape(1, -1), verbose=0)[0]
    i_emb = item_tower.predict(item_features, batch_size=512, verbose=0)
    scores = i_emb @ u_emb  # both L2-normalized → cosine sim

    game_to_idx = {g: i for i, g in enumerate(game_ids)}
    owned_set = {game_to_idx[g] for g in owned_game_ids if g in game_to_idx}
    candidates = [
        (game_ids[i], float(scores[i]))
        for i in np.argsort(-scores)
        if i not in owned_set
    ]
    return candidates[:n]


if __name__ == "__main__":
    # Smoke test on random tensors
    rng = np.random.default_rng(42)
    n_users, n_items, u_dim, i_dim = 200, 500, 32, 80
    user_features = rng.standard_normal((n_users, u_dim)).astype(np.float32)
    item_features = rng.binomial(1, 0.1, size=(n_items, i_dim)).astype(np.float32)
    positives = [(int(rng.integers(0, n_users)), int(rng.integers(0, n_items))) for _ in range(1000)]
    train_two_tower(user_features, item_features, positives, epochs=3)
