import argparse
import pandas as pd
import re


urls_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mentions_regex = re.compile(r'@(\w+)')

positive_smiles_regex = re.compile(
    ":-\)|:\)|;\)|:D|:o\)|:]|:3|:c\)|:>|=]|8\)|=\)|:}|:^\)|:-D|8-D|8D|XD|=-D|=D|=-3|=3|\)\)|\^\^|\^_\^|\\o\/|\m\/|<3".lower())

negative_smiles_regex = re.compile(r";\(|:\(|:\[|:{|\(\(|:'\(|:'\[|:3|:c|:с".lower())

positive_smile_replacement = ' smpositive '
negative_smile_replacement = ' smnegative '

only_letters_regex = re.compile(r'[^\w ]|\d|_')
one_space_regex = re.compile(r' +')


def clean_tweet(tweet):
    text = tweet.lower()

    text = mentions_regex.sub('', text)
    text = urls_regex.sub('', text)

    text = positive_smiles_regex.sub(positive_smile_replacement, text)
    text = negative_smiles_regex.sub(negative_smile_replacement, text)

    text = only_letters_regex.sub(' ', text)
    text = one_space_regex.sub(' ', text)

    text = text.strip()

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='F',
                        nargs='+',
                        help='CSV files to process',
                        type=argparse.FileType('r'))
    parser.add_argument('-o', '--output',
                        help='Output file',
                        type=argparse.FileType('w+'),
                        required=False)

    args = parser.parse_args()

    for f in args.files:
        df = pd.read_csv(f, header=None, sep=';', index_col=False, names=[
            'id', 'tdate', 'tname', 'ttext', 'ttype',
            'trep', 'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount'])

        for tweet in df['ttext']:
            text = clean_tweet(tweet)
            args.output.write(text + ' ')


if __name__ == "__main__":
    main()
