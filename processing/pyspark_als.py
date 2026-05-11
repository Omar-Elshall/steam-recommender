"""
PySpark ALS Matrix Factorization for Steam game recommendations.

Implements collaborative filtering on implicit feedback (game ownership)
using Spark's distributed ALS implementation. This module is the
production-grade counterpart to the in-house NumPy/CuPy ALS in
ALS_matrix_factorization.py — same math, distributed execution.

Why both? The NumPy/CuPy version is the educational implementation
showing the algorithm from first principles. PySpark gives us the
production path that scales horizontally and handles the full
~3 million user-item interaction matrix without OOM.
"""

import os
import pickle
from typing import List, Tuple

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator


PROCESSED_DIR = "processing/processedData"
MODEL_DIR = os.path.join(PROCESSED_DIR, "pyspark_als_model")


def get_spark(app_name: str = "steam-recommender-als") -> SparkSession:
    """Create or get a Spark session tuned for local recommender workloads."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def build_interactions_df(spark: SparkSession, user2game: dict) -> "pyspark.sql.DataFrame":
    """Convert user->[games] dict to a Spark DataFrame of (user_id, item_id, rating=1.0).

    Implicit feedback: ownership signal == 1.0, no negative samples.
    Spark ALS with `implicitPrefs=True` interprets the rating as a confidence weight.
    """
    rows: List[Tuple[str, str, float]] = []
    for user, games in user2game.items():
        for g in games:
            rows.append((str(user), str(g), 1.0))
    df = spark.createDataFrame(rows, ["user", "item", "rating"])
    return df


def index_users_items(df):
    """Map string ids to integer ids (ALS requires integer user/item columns)."""
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline

    user_idx = StringIndexer(inputCol="user", outputCol="user_id", handleInvalid="keep")
    item_idx = StringIndexer(inputCol="item", outputCol="item_id", handleInvalid="keep")
    pipeline = Pipeline(stages=[user_idx, item_idx])
    model = pipeline.fit(df)
    indexed = model.transform(df).select(
        col("user_id").cast("int"),
        col("item_id").cast("int"),
        col("rating"),
    )
    return indexed, model


def train_pyspark_als(
    user2game: dict,
    rank: int = 64,
    reg_param: float = 0.1,
    max_iter: int = 15,
    alpha: float = 40.0,
    save: bool = True,
):
    """Train PySpark ALS with implicit feedback on Steam ownership data.

    Args:
        user2game: dict mapping user_id -> list of owned game_ids
        rank: latent factor dimension
        reg_param: L2 regularization
        max_iter: ALS iterations
        alpha: implicit confidence multiplier (40 is the canonical Hu, Koren, Volinsky value)
    """
    spark = get_spark()
    df = build_interactions_df(spark, user2game)
    indexed, indexer_pipeline = index_users_items(df)

    train, test = indexed.randomSplit([0.9, 0.1], seed=42)

    als = ALS(
        rank=rank,
        regParam=reg_param,
        maxIter=max_iter,
        alpha=alpha,
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        implicitPrefs=True,
        coldStartStrategy="drop",
        seed=42,
    )

    print(f"Training PySpark ALS: rank={rank}, reg={reg_param}, iters={max_iter}, alpha={alpha}")
    model = als.fit(train)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"PySpark ALS test RMSE: {rmse:.4f}")

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        model.write().overwrite().save(MODEL_DIR)
        indexer_pipeline.write().overwrite().save(os.path.join(PROCESSED_DIR, "pyspark_indexer"))
        print(f"Saved PySpark ALS model to {MODEL_DIR}")

    return model, indexer_pipeline, rmse


def recommend_for_user(model: ALSModel, indexer_pipeline, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
    """Get top-N recommendations for a single user using the trained PySpark ALS model."""
    spark = get_spark()
    user_df = spark.createDataFrame([(user_id, "_dummy", 1.0)], ["user", "item", "rating"])
    indexed = indexer_pipeline.transform(user_df)
    user_int = indexed.select("user_id").first()[0]

    user_subset = spark.createDataFrame([(int(user_int),)], ["user_id"])
    recs = model.recommendForUserSubset(user_subset, n).first()
    if recs is None:
        return []

    item_labels = indexer_pipeline.stages[1].labels
    return [
        (item_labels[r.item_id], float(r.rating))
        for r in recs.recommendations
    ]


if __name__ == "__main__":
    # Smoke test: load processed user2game and train
    with open(os.path.join(PROCESSED_DIR, "user2game_dict.pkl"), "rb") as f:
        user2game = pickle.load(f)
    train_pyspark_als(user2game, max_iter=5)
