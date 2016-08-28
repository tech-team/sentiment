# sentiment
Sentiment Analysis using TensorFlow

## Before run:
cp meta_config.yaml.sample meta_config.yaml

## Train:
python cnn_model.py train -c meta_config.yaml --interactive

## Use:
python cnn_model.py load -c meta_config.yaml --interactive
