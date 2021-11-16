from argparse import ArgumentParser
import os

BASE_LOG_PATH = 'snap'

arg_parser = ArgumentParser()
arg_parser.add_argument('--include_pos_tag', default='', choices=['input', 'aux', ''], help='Include part-of-speech tags as input, output (for aux function) or neither')
arg_parser.add_argument('--teacher_forcing', default=0.5, type=float, help='Probability of using teacher forcing during training')
arg_parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
arg_parser.add_argument('--name', default='default', type=str, help='Model name')

args = arg_parser.parse_args()

if not os.path.exists(f'{BASE_LOG_PATH}/{args.name}'):
    os.makedirs(f'{BASE_LOG_PATH}/{args.name}')