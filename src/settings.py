from torch import cuda

IMAGES_PATH = "../../data/Images/"
CAPTIONS_PATH = "../../data/captions.csv"

# Model related configs
TOKENIZER_NAME = 'distilbert-base-uncased'
EPOCHS = 4
BATCH_SIZE = 32
PROJECTION_DIM = 256
DEVICE = "cuda" if cuda.is_available() else "cpu"
