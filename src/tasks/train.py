import os

from comet_ml import Experiment

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from src.data.scan import get_final_map_function

from ..models.lstm import Seq2SeqAttentionLSTM, Seq2SeqLSTM
from ..utils.constants import ACTION_OUTPUT_NAME, POS_OUTPUT_NAME, get_default_params


def train(
    train_ds,
    test_ds,
    pad_idx,
    start_idx,
    end_idx,
    experiment: Experiment,
    params: dict = None,
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

    # Create metrics and losses according to task
    losses = {ACTION_OUTPUT_NAME: "categorical_crossentropy"}
    metrics = {ACTION_OUTPUT_NAME: keras.metrics.CategoricalAccuracy(name=ACTION_OUTPUT_NAME)}
    loss_weights = {ACTION_OUTPUT_NAME: 1.0}

    if params["include_pos_tag"] == "aux":
        losses[POS_OUTPUT_NAME] = "categorical_crossentropy"
        metrics[POS_OUTPUT_NAME] = keras.metrics.CategoricalAccuracy(name=POS_OUTPUT_NAME)
        loss_weights[POS_OUTPUT_NAME] = 0.2

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=5.0)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics, loss_weights=loss_weights, run_eagerly=True)

    # Checkpoint callback
    checkpoint_path = os.path.join("snap", params["name"])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, "best_action_accuracy"),
        monitor=ACTION_OUTPUT_NAME,
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
            callbacks=[checkpoint_callback],
        )

    return model, history


if __name__ == "__main__":
    train()
