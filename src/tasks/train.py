import os
from typing import overload

import numpy as np
from comet_ml import Experiment
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from src.utils.load import load_model

from ..data.scan import OUT_VOCAB_SIZE, POS_VOCAB_FILE, POS_VOCAB_SIZE, get_final_map_function
from ..models.lstm import Seq2SeqAttentionLSTM, Seq2SeqLSTM
from ..utils.constants import ACTION_OUTPUT_NAME, POS_OUTPUT_NAME, get_default_params
from ..utils.loss import get_masked_categorical_crossentropy, MaskedCategoricalAccuracy


def train(
    train_ds,
    test_ds,
    pad_idx,
    start_idx,
    end_idx,
    experiment: Experiment,
    params: dict = None,
    load: bool = False,
    in_vectorizer=None,
    pos_vectorizer=None,
):
    params = {**get_default_params(), **params} if params else get_default_params()
    experiment.set_name(params["name"])
    experiment.log_parameters(params)

    # Map dataset acording to task (pos tag as aux, input or none)
    final_map_fn = get_final_map_function(
        include_pos_tag=params["include_pos_tag"],
        teacher_forcing=params["teacher_forcing"],
    )
    train_ds = train_ds.map(final_map_fn)
    test_ds = test_ds.map(final_map_fn)

    model_params = {
        "pad_idx": pad_idx,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "hidden_size": params["hidden_size"],
        "dropout": params["dropout"],
        "hidden_layers": params["hidden_layers"],
        "teacher_forcing": params["teacher_forcing"],
        "include_pos_tag": params["include_pos_tag"],
    }

    # Create model
    if params["use_attention"]:
        model = Seq2SeqAttentionLSTM(**model_params)
    else:
        model = Seq2SeqLSTM(**model_params)

    if load:
        model = load_model(model, in_vectorizer, pos_vectorizer, params["name"])

    # Create metrics and losses according to task
    mask_value: tf.Tensor = tf.one_hot(pad_idx, OUT_VOCAB_SIZE)
    masked_categorical_crossentropy = get_masked_categorical_crossentropy(mask_value)
    losses = {ACTION_OUTPUT_NAME: masked_categorical_crossentropy}
    metrics = {
        ACTION_OUTPUT_NAME: MaskedCategoricalAccuracy(
            mask_value, name="accuracy" if params["include_pos_tag"] == "aux" else f"{ACTION_OUTPUT_NAME}_accuracy"
        )
    }

    if params["include_pos_tag"] == "aux":
        pos_mask_value: tf.Tensor = tf.one_hot(pad_idx, POS_VOCAB_SIZE)
        losses[POS_OUTPUT_NAME] = get_masked_categorical_crossentropy(pos_mask_value)
        metrics[POS_OUTPUT_NAME] = MaskedCategoricalAccuracy(pos_mask_value, name="accuracy")

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=5.0)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics, run_eagerly=True)

    # Checkpoint callback
    checkpoint_path = os.path.join("snap", params["name"])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, "best_action_accuracy"),
        monitor=f"{ACTION_OUTPUT_NAME}_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )

    val_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, "best_val_action_accuracy"),
        monitor=f"val_{ACTION_OUTPUT_NAME}_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )

    # Train model
    with experiment.train():
        history = model.fit(
            train_ds.shuffle(1000, reshuffle_each_iteration=True).batch(params["batch_size"]),
            validation_data=test_ds.batch(params["batch_size"]),
            epochs=params["epochs"],
            callbacks=[checkpoint_callback, val_checkpoint_callback],
        )

    model.save_weights(os.path.join(checkpoint_path, "final_weights"), overwrite=True)

    with experiment.test():
        results = model.evaluate(test_ds.batch(params["batch_size"]))

        results = {x: y for x, y in zip(model.metrics_names, results)}

        if ACTION_OUTPUT_NAME not in results and "accuracy" in results:
            results[ACTION_OUTPUT_NAME] = results["accuracy"]

        experiment.log_metrics(results)

    return model, history


if __name__ == "__main__":
    train()
