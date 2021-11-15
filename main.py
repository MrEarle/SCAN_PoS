from src.utils.args import args
from src.tasks import train

if __name__ == '__main__':
    print('\n')
    print(args, end='\n\n')
    train.train()
