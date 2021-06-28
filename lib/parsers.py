import argparse

def train_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', type=int,
            help="Number of epochs, default=1", default=1)

    parser.add_argument('-mbs', '--minibatchsize', type=int,
            default=3, help="Mini Batch Size, default=3")

    parser.add_argument('-d', '--numberofdirs', type=int,
            default=0, help="Number of directories, default=0")

    parser.add_argument('-t', '--tail', type=str,
            default='', help="Tail to add to the filename, "+
                             "default=''")

    parser.add_argument('-l', '--loss', type=int,
            default=0, help="Choose mse (0) or custom (N>0), "
            "default = 0 (mse_loss)")

    args = parser.parse_args()

    return args
