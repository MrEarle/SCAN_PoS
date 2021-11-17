from argparse import ArgumentParser
import os

BASE_LOG_PATH = 'snap'

arg_parser = ArgumentParser()
arg_parser.add_argument('--include_pos_tag', default='', choices=['input', 'aux', ''], help='Include part-of-speech tags as input, output (for aux function) or neither')
arg_parser.add_argument('--use_attention', default=False, action='store_true', help='Use attention mechanism')
arg_parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
arg_parser.add_argument('--name', default='default', type=str, help='Model name')

# En SCAN lo usan en 0.5
arg_parser.add_argument('--teacher_forcing', default=0.5, type=float, help='Probability of using teacher forcing during training')

# En SCAN prueban con 1 o 2
arg_parser.add_argument('--hidden_layers', default=1, choices=[1, 2], type=int, help='Number of hidden layers')

# En SCAN prueban con 25, 50, 100, 200 o 400
arg_parser.add_argument('--hidden_size', default=25, type=int, help='Hidden size of recurrent layer')

# En SCAN prueban con 0, 0.1 o 0.5
arg_parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')

# En SCAN prueban usando 100000 pruebas, lo que corresponde a aprox 100000/16728 = ~6 epochs
arg_parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')

args = None

def parse_args():
    global args
    _args, _ = arg_parser.parse_known_args()

    if not os.path.exists(f'{BASE_LOG_PATH}/{_args.name}'):
        os.makedirs(f'{BASE_LOG_PATH}/{_args.name}')

    args = _args