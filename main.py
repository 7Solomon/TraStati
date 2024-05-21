import argparse
from functions import train, data, look_trough_dataset, test_and_visualize_model
from experimental_functions import endless_loop

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
    parser.add_argument('--loop', action='store_true', help='Bro KA')
    return parser.parse_args()



def main():
    args = parse_arguments()
    if args.data:
        data()
    elif args.train:
        train()
    elif args.display:
        look_trough_dataset()
    elif args.test:
        test_and_visualize_model()
    elif args.loop:
        endless_loop()
    else:
        print('No Argument')
    
if __name__ == '__main__':
    main()