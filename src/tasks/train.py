from tensorflow import keras

from src.data.scan import get_final_map_function

from ..models.lstm import Seq2SeqAttentionLSTM, Seq2SeqLSTM
from ..utils.constants import ACTION_OUTPUT_NAME, POS_OUTPUT_NAME
from ..utils.args import args

def train(train_ds, test_ds, pad_idx, start_idx, end_idx, hp_callback=None):

    final_map_fn = get_final_map_function()

    train_ds = train_ds.map(final_map_fn)
    test_ds = test_ds.map(final_map_fn)

    if args.use_attention:
        model = Seq2SeqAttentionLSTM(pad_idx=pad_idx, start_idx=start_idx, end_idx=end_idx)
    else:
        model = Seq2SeqLSTM(pad_idx=pad_idx, start_idx=start_idx, end_idx=end_idx)

    losses = {ACTION_OUTPUT_NAME: 'categorical_crossentropy'}
    metrics = {ACTION_OUTPUT_NAME: 'accuracy'}
    loss_weights = {ACTION_OUTPUT_NAME: 1.0}

    if args.include_pos_tag == 'aux':
        losses[POS_OUTPUT_NAME] = 'categorical_crossentropy'
        metrics[POS_OUTPUT_NAME] = 'accuracy'
        loss_weights[POS_OUTPUT_NAME] = 0.2

    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=5.0)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
        loss_weights=loss_weights,
        run_eagerly=True
    )

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'snap/{args.name}', histogram_freq=1)

    callbacks = [tensorboard_callback]
    if hp_callback:
        callbacks.append(hp_callback)

    model.fit(
        train_ds.shuffle(1000, reshuffle_each_iteration=True).batch(args.batch_size),
        epochs=args.epochs,
        callbacks=callbacks,
    )
    
    model.summary()

    result = model.evaluate(test_ds.batch(512))
    print(result)

    return result

if __name__ == '__main__':
    train()