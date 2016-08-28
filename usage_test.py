from cnn_model import load_model, predict


def main():
    cnn = load_model(False, 'meta_config.yaml')

    sentences = [
        'я люблю это',
        'мне не нравится это',
    ]

    for sentence in sentences:
        prediction = predict(cnn, sentence)
        print(sentence)
        print(prediction)
        print('--------')

    cnn.close()


if __name__ == '__main__':
    main()
