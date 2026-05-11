"""
TensorFlow autoencoder for content-based filtering on Steam games.

Encodes each game's content-feature vector (tags + genres + specs, multi-hot)
into a compressed latent space. Similarity in latent space is used for
recommendations — games close to a user's owned-game centroid are surfaced.

Adapted from the CMP 49412 (Intelligent Recommendation Systems) course
notebook on autoencoder-based content filtering.
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


PROCESSED_DIR = "processing/processedData"
MODEL_PATH = os.path.join(PROCESSED_DIR, "tf_autoencoder.keras")


def build_autoencoder(input_dim: int, latent_dim: int = 64, hidden: int = 256) -> keras.Model:
    """Symmetric autoencoder: input -> hidden -> latent -> hidden -> input.

    Bottleneck `latent_dim` controls how much content compression happens.
    Larger latent = retains more nuance but less denoising; smaller = more
    aggressive feature deduplication.
    """
    inputs = keras.Input(shape=(input_dim,), name="content_vector")
    x = layers.Dense(hidden, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = layers.Dense(hidden, activation="relu")(latent)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

    autoencoder = keras.Model(inputs, outputs, name="content_autoencoder")
    encoder = keras.Model(inputs, latent, name="content_encoder")

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["mse"],
    )
    return autoencoder, encoder


def train_autoencoder(
    content_matrix: np.ndarray,
    latent_dim: int = 64,
    hidden: int = 256,
    epochs: int = 30,
    batch_size: int = 256,
    save: bool = True,
) -> Tuple[keras.Model, keras.Model, dict]:
    """Train autoencoder on multi-hot content feature vectors.

    Args:
        content_matrix: (n_games, n_features) binary or normalized feature matrix
        latent_dim: bottleneck size
    """
    n_games, n_features = content_matrix.shape
    print(f"Training autoencoder: {n_games} games × {n_features} features → {latent_dim}-d latent")

    autoencoder, encoder = build_autoencoder(n_features, latent_dim, hidden)

    history = autoencoder.fit(
        content_matrix,
        content_matrix,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        verbose=2,
    )

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        autoencoder.save(MODEL_PATH)
        encoder.save(MODEL_PATH.replace(".keras", "_encoder.keras"))
        print(f"Saved autoencoder to {MODEL_PATH}")

    return autoencoder, encoder, history.history


def get_latent_embeddings(encoder: keras.Model, content_matrix: np.ndarray) -> np.ndarray:
    """Project all games into latent space."""
    return encoder.predict(content_matrix, batch_size=512, verbose=0)


def recommend_for_user(
    encoder: keras.Model,
    content_matrix: np.ndarray,
    game_ids: List[str],
    owned_game_ids: List[str],
    n: int = 10,
) -> List[Tuple[str, float]]:
    """Recommend games similar to the centroid of a user's owned games in latent space."""
    game_to_idx = {g: i for i, g in enumerate(game_ids)}
    owned_indices = [game_to_idx[g] for g in owned_game_ids if g in game_to_idx]
    if not owned_indices:
        return []

    embeddings = get_latent_embeddings(encoder, content_matrix)
    user_centroid = embeddings[owned_indices].mean(axis=0)

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    centroid_norm = user_centroid / (np.linalg.norm(user_centroid) or 1.0)
    scores = normalized @ centroid_norm

    # Exclude already-owned games
    owned_set = set(owned_indices)
    candidates = [
        (game_ids[i], float(scores[i]))
        for i in np.argsort(-scores)
        if i not in owned_set
    ]
    return candidates[:n]


if __name__ == "__main__":
    # Smoke test on random data
    rng = np.random.default_rng(42)
    fake_content = rng.binomial(1, 0.1, size=(500, 80)).astype(np.float32)
    _, encoder, hist = train_autoencoder(fake_content, latent_dim=16, epochs=3)
    print(f"Final loss: {hist['loss'][-1]:.4f}")
