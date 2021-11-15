from tensorflow import keras

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
        loss_weights[ACTION_OUTPUT_NAME] = 0.2

    model.compile(loss=losses, optimizer='adam', metrics=metrics, run_eagerly=True)

    history = model.fit(train_ds.shuffle(1000, reshuffle_each_iteration=True).batch(512, drop_remainder=True), epochs=20)
    
    model.summary()
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    loss, acc = model.evaluate(test_ds.batch(32, drop_remainder=True))
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {acc:.4f}')

if __name__ == '__main__':
    train()