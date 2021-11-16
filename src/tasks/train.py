from tensorflow import keras
from tensorflow.python.keras.backend import clip

from ..data.scan import get_dataset
from ..models.lstm import Seq2SeqAttentionLSTM
from ..utils.constants import ACTION_OUTPUT_NAME, POS_OUTPUT_NAME
from ..utils.args import args

def train():
    train_ds, test_ds, (in_vec, _, _) = get_dataset('simple')

    pad_idx = in_vec.get_vocabulary().index('')
    start_idx = in_vec.get_vocabulary().index('<sos>')
    end_idx = in_vec.get_vocabulary().index('<eos>')

    model = Seq2SeqAttentionLSTM(pad_idx=pad_idx, start_idx=start_idx, end_idx=end_idx)

    losses = {ACTION_OUTPUT_NAME: 'categorical_crossentropy'}
    metrics = {ACTION_OUTPUT_NAME: 'accuracy'}
    loss_weights = {ACTION_OUTPUT_NAME: 1.0}

    if args.include_pos_tag == 'aux':
        losses[POS_OUTPUT_NAME] = 'categorical_crossentropy'
        metrics[POS_OUTPUT_NAME] = 'accuracy'
        loss_weights[POS_OUTPUT_NAME] = 0.2

    optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=5.0)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
        loss_weights=loss_weights,
        run_eagerly=True
    )

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'snap/{args.name}', histogram_freq=1)

    model.fit(
        train_ds.shuffle(1000, reshuffle_each_iteration=True).batch(args.batch_size),
        epochs=100,
        callbacks=[tensorboard_callback],
    )
    
    model.summary()

    result = model.evaluate(test_ds.batch(512))
    print(result)

if __name__ == '__main__':
    train()