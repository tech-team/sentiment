import argparse

import pandas as pd

from data_utils import clean_tweet


def main():
    from data_utils import RUBTSOVA_HEADER

    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='F',
                        nargs='+',
                        help='CSV files to process',
                        type=argparse.FileType('r'))
    parser.add_argument('-o', '--output',
                        help='Output file',
                        type=argparse.FileType('w+'),
                        required=True)

    args = parser.parse_args()

    for f in args.files:
        df = pd.read_csv(f, header=None, sep=';', index_col=False, names=RUBTSOVA_HEADER)

        for tweet in df['ttext']:
            text = clean_tweet(tweet)
            args.output.write(text + ' ')


if __name__ == "__main__":
    main()
