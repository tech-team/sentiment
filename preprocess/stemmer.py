import nltk
from nltk.stem.snowball import RussianStemmer


def stem_corpus(input_path, output_path):
    stem = RussianStemmer()
    last_word = ''

    i = 0
    with open(output_path, 'w', encoding='utf8') as o:
        with open(input_path, 'r', encoding='utf8') as f:
            while True:
                s = f.read(1024 * 1024)
                if not s or not len(s):
                    o.write(last_word)
                    break

                words = s.split(' ')

                if s[0] != ' ':
                    # last_word was incomplete
                    words[0] = last_word + words[0]

                for word in words[:-1]:
                    stemmed = stem.stem(word)
                    o.write(stemmed + ' ')

                i += 1
                print('Stemmed {} MBs'.format(i))

                last_word = words[-1]
