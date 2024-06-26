import argparse
from src.functions import train, data, look_trough_dataset, test_and_visualize_model
from src.test import *

#functions = {'data':data,
#             'display':look_trough_dataset,
#             'test':test_and_visualize_model,
#             'loop':endless_loop}

def parse_arguments():
    parser = argparse.ArgumentParser(description='This is a Parser')
    parser.add_argument('--train', action='store_true', help='Bro KA')
    parser.add_argument('--data', action='store_true', help='Bro KA')
    parser.add_argument('--display', action='store_true', help='Bro KA')
    parser.add_argument('--test', action='store_true', help='Bro KA')
    parser.add_argument('--debug', action='store_true', help='Bro KA')
    return parser.parse_args()



def main():
    args = parse_arguments()
    if args.data:
        data()
    if args.train:
        train()
    if args.display:
        look_trough_dataset()
    if args.test:
        test_and_visualize_model()
    if args.debug:
        test()
    if not args.data and not args.train and not args.display and not args.test:
        print('No Argument')
    
if __name__ == '__main__':
    main()