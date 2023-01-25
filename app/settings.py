from torch import cuda

IMAGES_PATH = "../data/Images/"
CAPTIONS_PATH = r".\data\captions.csv"

MODEL_PATH = r".\models\best_model.pth"
IMG_EMB_PATH = r".\models\img_embeddings.pt"
TEXTS_EMB_PATH = r".\models\text_embeddings.pt"

# Model related configs
TOKENIZER_NAME = 'distilbert-base-uncased'
EPOCHS = 4
BATCH_SIZE = 32
PROJECTION_DIM = 256
DEVICE = "cuda" if cuda.is_available() else "cpu"
