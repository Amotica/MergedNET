class Config(object):
    # Datasets
    DATASET = "Dataset name"
    DATASET_FOLDER = 'folder-to-dataset/'  # save_path = 'omniglot/data/'
    TRAIN_FOLDER = "training-data-folder/"
    TEST_FOLDER = 'test-data-folder/'
    TRAIN_PICKLE_FILE = "train.pickle"
    VAL_PICKLE_FILE = "val.pickle"

    # Learning Metrics
    SIMILARITY_METRIC = "L1"  # L1, L2, cosine, max, concat

    # backbone Parameters
    FEATURE_SIZE = 32
    BACKBONE = "RESNET18"  # siamese / Resnet50

    # Network parameters - INPUT
    IMAGE_WIDTH = 105
    IMAGE_HEIGHT = 105
    IMAGE_CHANNELS = 3

    # Hyper parameters
    BATCH_SIZE = 32
    EPOCHS = 20000  # No. of training iterations
    N_WAY = 20  # how many classes for testing one-shot tasks
    EVALUATE_INTERVAL = 200  # interval for evaluating on one-shot tasks
    N_VAL = 250  # how many one-shot tasks to validate on
    MODEL_PATH = 'weights/'
