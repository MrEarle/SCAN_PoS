import src.utils.args as args

if __name__ == '__main__':
    args.parse_args()
    print('\n')
    print(args.args, end='\n\n')


    from src.tasks import train
    from src.data.scan import get_dataset
    
    train_ds, test_ds, (in_vec, _, _) = get_dataset('simple')
    pad_idx = in_vec.get_vocabulary().index('')
    start_idx = in_vec.get_vocabulary().index('<sos>')
    end_idx = in_vec.get_vocabulary().index('<eos>')
    
    train.train(train_ds, test_ds, pad_idx, start_idx, end_idx)
