from ..data.scan import get_dataset
from ..models.lstm import Seq2SeqAttentionLSTM

def train():
    train_ds, test_ds, (in_vec, out_vec) = get_dataset('simple')

    pad_idx = in_vec.get_vocabulary().index('')
    start_idx = in_vec.get_vocabulary().index('<SOS>')
    end_idx = in_vec.get_vocabulary().index('<EOS>')

    model = Seq2SeqAttentionLSTM(pad_idx=pad_idx, start_idx=start_idx, end_idx=end_idx)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
    history = model.fit(train_ds.shuffle(1000, reshuffle_each_iteration=True).batch(512, drop_remainder=True), epochs=3)

    model.summary()
    loss, acc = model.evaluate(test_ds.batch(32, drop_remainder=True))
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {acc:.4f}')

if __name__ == '__main__':
    train()