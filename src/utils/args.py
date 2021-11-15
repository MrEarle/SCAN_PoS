from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument('--include_pos_tag', default='', choices=['input', 'aux', ''], help='Include part-of-speech tags as input, output (for aux function) or neither')

args = arg_parser.parse_args()