import argparse

from data_utils import clean_tweet


def main():
    """Converts mysql's output to words corpus
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='F',
                        help='File with tweets separated by \n',
                        type=argparse.FileType('r', encoding='utf8'))
    parser.add_argument('-o', '--output',
                        help='Output file',
                        type=argparse.FileType('w+', encoding='utf8'),
                        required=False)

    args = parser.parse_args()

    for i, tweet in enumerate(args.file):
        if i == 0:  # skip first line
            continue

        tweet = tweet.replace('\\n', ' ')  # replace new lines inside tweets
        text = clean_tweet(tweet)
        args.output.write(text + ' ')

        if i % 1000 == 0:
            print('Tweets parsed: {}'.format(i))

    print('Done')


if __name__ == "__main__":
    main()
