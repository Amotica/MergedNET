# ====================================================================
# Parts of these codes are based on the implementation at:
# https://github.com/hlamba28/One-Shot-Learning-with-Siamese-Networks
# ====================================================================
import numpy as np
from cv2 import imread, resize
import time, glob
import cv2

from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
import keras
from keras.layers.core import Lambda, Flatten, Dense

from keras.regularizers import l2
from keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng
from config import Config
import os

from classification_models.resnet.models import ResNet18
from keras_efficientnets import EfficientNetB0

from keras.applications.mobilenet_v2 import MobileNetV2

import joblib

# ==================================================
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes it works better.
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
# ======================================================

class trainConfig(Config):
    # Datasets
    DATASET = "cifar100" #["cifar100", "CUB_200_2011", "miniImageNet", "caltech256"]
    TRAIN_PICKLE_FILE = "train.pickle"
    VAL_PICKLE_FILE = "val.pickle"

    # Learning Metrics
    SIMILARITY_METRIC = "L1" # L1, L2, cosine, max, concat
    LR_FACTOR = 0.1
    LR = 0.00001  # L1 = L2 = 0.001; cosine = 0.00001
    PATIENCE = 5
    MIN_LR = 0.000000000001

    # backbone Parameters
    FEATURE_SIZE = 32
    BACKBONE = "MobileNetV2" # siamese / Resnet18 / EfficientNetB0 / MobileNetV2

    # Network parameters - INPUT
    IMAGE_WIDTH = 105
    IMAGE_HEIGHT = 105
    IMAGE_CHANNELS = 3

    # Hyper parameters
    BATCH_SIZE = 64 #64
    EPOCHS = 40000  # No. of training iterations
    N_WAY = 5  # how many classes for testing one-shot tasks
    EVALUATE_INTERVAL = 200  # interval for evaluating on one-shot tasks
    N_VAL = 250  # how many one-shot tasks to validate on
    MODEL_PATH = "weights/"
    Shots = 5



class onShotLearnig:

    def __init__(self, Backbone, similarityMetric, imageShape, dataset):
        self.config = trainConfig()

        self.config.BACKBONE = Backbone
        self.config.SIMILARITY_METRIC = similarityMetric

        self.config.IMAGE_WIDTH = imageShape[0]
        self.config.IMAGE_HEIGHT = imageShape[1]
        self.config.IMAGE_CHANNELS = imageShape[2]

        self.config.DATASET = dataset

        self.config.DATASET_FOLDER = "datasets/" + dataset + "/data/"
        self.config.TRAIN_FOLDER = "datasets/" + dataset + "/train/"
        self.config.TEST_FOLDER = "datasets/" + dataset + "/val/"

        self.config.MODEL_PATH = self.config.MODEL_PATH + dataset + "/" + self.config.BACKBONE + "/" + similarityMetric + "/"


    def loadimgs(self, path, n=0):
        #print(path)
        X, y=[], []
        cat_dict = {}
        lang_dict = {}
        curr_y = n
        # we load every alphabet seperately so we can isolate them later
        for alphabet in os.listdir(path):
            print("loading alphabet: " + alphabet)
            lang_dict[alphabet] = [curr_y,None]
            alphabet_path = os.path.join(path,alphabet)
            # every letter/category has it's own column in the array, so  load seperately
            for letter in os.listdir(alphabet_path):
                cat_dict[curr_y] = (alphabet, letter)
                category_images=[]
                letter_path = os.path.join(alphabet_path, letter)
                # read all the images in the current category
                for filename in os.listdir(letter_path):
                    image_path = os.path.join(letter_path, filename)
                    image = imread(image_path)
                    # resize the images as required
                    image = resize(image, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
                    image = image.astype('float32')
                    image /= 255.0
                    category_images.append(image)
                    y.append(curr_y)
                try:
                    X.append(np.stack(category_images))
                # edge case  - last one
                except ValueError as e:
                    print(e)
                    print("error - category_images:", category_images)
                curr_y += 1
                lang_dict[alphabet][1] = curr_y - 1
        y = np.vstack(y)

        X = np.stack(X)
        #print(y.shape)
        print(X.shape)
        return X,y,lang_dict

    def L1_similarity(self, vects):
        '''
        :param vects:
        :return:
        '''
        x, y = vects

        return K.abs(x - y)

    def L2_similarity(self, vect):
        x, y = vect
        abs_square = K.square(K.abs(x - y))
        result = K.maximum(abs_square, K.epsilon())
        return result

    def cosine(self, vects):
        '''
            The Dot layer in Keras now supports built-in Cosine similarity using the normalize = True parameter.
            From the Keras Docs:
            keras.layers.Dot(axes, normalize=True)
            normalize: Whether to L2-normalize samples along the dot product axis before taking the dot product.
            If set to True, then the output of the dot product is the cosine proximity between the two samples.
        '''
        x, y = vects
        #cos_layer = keras.layers.dot([x, y], axes=1, normalize=True)
        cos_layer = keras.layers.Dot(axes=1, normalize=True)([x, y])
        return cos_layer

    def maxSimilarity(self, vects):
        '''
        :param vects:
        :return:
        '''
        import tensorflow as tf
        x, y = vects
        # L1 Similarity
        L1_similarity_layer = keras.layers.Lambda(self.L1_similarity)([x, y])

        # L2 Similarity
        L2_similarity_layer = keras.layers.Lambda(self.L2_similarity)([x, y])

        # Cosine Similarity
        cosine_similarity_layer = keras.layers.Lambda(self.cosine)([x, y])

        return keras.layers.Maximum(trainable=True, name="MaxSimilarity")([L1_similarity_layer, L2_similarity_layer, cosine_similarity_layer])

    def concatSimilarity(self, vects):
        '''
        :param vects:
        :return:
        '''
        import tensorflow as tf
        x, y = vects
        # L1 Similarity
        L1_similarity_layer = keras.layers.Lambda(self.L1_similarity)([x, y])

        # L2 Similarity
        L2_similarity_layer = keras.layers.Lambda(self.L2_similarity)([x, y])

        # Cosine Similarity
        cosine_similarity_layer = keras.layers.Lambda(self.cosine)([x, y])
        return keras.layers.Concatenate(trainable=True, name="ConcatSimilarity")([L1_similarity_layer, L2_similarity_layer, cosine_similarity_layer])


    def initialize_weights(self, shape, dtype=None):
        """
        :param shape: image shape
        :param dtype: the datatype of the image
        :return:  initialize CNN layer weights with mean as 0.0 and
        standard deviation of 0.01 (see: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf)
        """
        return np.random.normal(loc=0.0, scale=1e-2, size=shape)

    def initialize_bias(self, shape, dtype=None):
        """
        :param shape: image shape
        :param dtype: the datatype of the image
        :return: initialize CNN layer bias with mean as 0.5 and
        standard deviation of 0.01 (see: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        """
        return np.random.normal(loc=0.5, scale=1e-2, size=shape)

    def getResNet18(self, input_shape):
        """
        :param input_shape: the input image shape
        :return: the ResNet18 Backbone model
        """

        # Define the tensors for the two input images
        input = Input(input_shape, name="MainInput")
        left_input = Input(input_shape, name="LeftInput")
        right_input = Input(input_shape, name="RightInput")

        # create the base pre-trained model
        base_model = ResNet18(weights=None, include_top=False, input_shape=input_shape)
        x = base_model.output
        # Flatten
        x = Flatten()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer=self.initialize_weights,
                  bias_initializer=self.initialize_bias)(x) #sigmoid

        model = Model(inputs=[base_model.input], outputs=[x], name="ResNet18_Model")

        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        prediction = None
        # Add a customized layer to compute the similarity metric between the encodings
        if self.config.SIMILARITY_METRIC == "L1":
            similarity_layer = Lambda(self.L1_similarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "L2":
            similarity_layer = Lambda(self.L2_similarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "cosine":
            similarity_layer = Lambda(self.cosine)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "concat":
            similarity_layer = Lambda(self.concatSimilarity,trainable=True, name="ConcatSimilarity")([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "max":
            similarity_layer = Lambda(self.maxSimilarity,trainable=True, name="MaxSimilarity")([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias, name="Twin_DenseLayer")(similarity_layer)

        # Connect the inputs with the outputs
        ResNetSiamese = Model(inputs=[left_input, right_input], outputs=prediction, name="Twin_Model")

        return ResNetSiamese

    def getEfficientNetB0(self, input_shape):
        """
        :param input_shape: the input image shape
        :return: the EfficientNetB0 Backbone model
        """
        # Define the tensors for the two input images
        input = Input(input_shape)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        # create the base pre-trained model
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)

        x = base_model.output
        # Flatten
        x = Flatten()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer=self.initialize_weights,
                  bias_initializer=self.initialize_bias)(x) #sigmoid

        model = Model(inputs=[base_model.input], outputs=[x])

        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        prediction = None
        # Add a customized layer to compute the similarity metric between the encodings
        if self.config.SIMILARITY_METRIC == "L1":
            similarity_layer = Lambda(self.L1_similarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "L2":
            similarity_layer = Lambda(self.L2_similarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "cosine":
            similarity_layer = Lambda(self.cosine)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "concat":
            similarity_layer = Lambda(self.concatSimilarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "max":
            similarity_layer = Lambda(self.maxSimilarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)

        # Connect the inputs with the outputs
        EfficientNetB0Siamese = Model(inputs=[left_input, right_input], outputs=prediction)

        return EfficientNetB0Siamese

    def getMobileNetV2(self, input_shape):
        # Define the tensors for the two input images
        input = Input(input_shape)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        # create the base pre-trained model
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
        x = base_model.output
        # Flatten
        x = Flatten()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer=self.initialize_weights,
                  bias_initializer=self.initialize_bias)(x) #sigmoid

        model = Model(inputs=[base_model.input], outputs=[x])

        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        prediction = None
        # Add a customized layer to compute the similarity metric between the encodings
        if self.config.SIMILARITY_METRIC == "L1":
            similarity_layer = Lambda(self.L1_similarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "L2":
            similarity_layer = Lambda(self.L2_similarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "cosine":
            similarity_layer = Lambda(self.cosine)([encoded_l, encoded_r])
            # similarity_layer = keras.layers.dot([encoded_l, encoded_r], axes=-1, normalize=True)
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "concat":
            similarity_layer = Lambda(self.concatSimilarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)
        elif self.config.SIMILARITY_METRIC == "max":
            similarity_layer = Lambda(self.maxSimilarity)([encoded_l, encoded_r])
            prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(similarity_layer)

        # Connect the inputs with the outputs
        MobileNetV2Siamese = Model(inputs=[left_input, right_input], outputs=prediction)

        return MobileNetV2Siamese

    def get_batch(self, batch_size, X, X_classes):
        """Create batch of n pairs, half same class, half different class"""

        categories = X_classes
        n_classes, n_examples, w, h, channels = X.shape

        # randomly sample several classes to use in the batch
        categories = rng.choice(n_classes, size=(batch_size,), replace=False)

        # initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, h, w, channels)) for i in range(2)]

        # initialize vector for the targets
        targets = np.zeros((batch_size,))

        # make one half of it '1's, so 2nd half of batch has same class
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, channels)
            idx_2 = rng.randint(0, n_examples)

            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category
            else:
                # add a random number to the category modulo n classes to ensure 2nd image has a different category
                category_2 = (category + rng.randint(1, n_classes)) % n_classes

            pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, channels)

        return pairs, targets

    def MakeNShotTask(self, N, K, X, X_classes, language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        categories = X_classes
        n_classes, n_examples, w, h, channels = X.shape

        indices = rng.randint(0, n_examples, size=(N,))
        if language is not None:  # if language is specified, select characters for that language
            low, high = categories[language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            categories = rng.choice(range(low, high), size=(N,), replace=False)

        else:  # if no language specified just pick a bunch of random letters
            categories = rng.choice(range(n_classes), size=(N,), replace=False)

        true_category = categories[0]

        test_image = []
        support_set = []
        targets = []

        # Make the K-Shot N-Way tests tasks
        ex = rng.choice(n_examples, replace=False, size=(K+1,))
        for s in range(0, K):
            test = np.asarray([X[true_category, ex[s], :, :]] * N).reshape(N, w, h, channels)
            test_image.append(test)

            support = X[categories, indices, :, :]
            support[0, :, :] = X[true_category, ex[-1]]
            support = support.reshape(N, w, h, channels)
            support_set.append(support)

            target = np.zeros((N,))
            target[0] = 1
            targets.append(target)

        # reshape them all
        test_image = np.array(test_image)
        test_image = test_image.reshape(test_image.shape[0] * test_image.shape[1], w, h, channels)

        support_set = np.array(support_set)
        support_set = support_set.reshape(support_set.shape[0] * support_set.shape[1], w, h, channels)

        targets = np.array(targets)
        targets = targets.reshape(targets.shape[0] * targets.shape[1], )

        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]

        return pairs, targets

    def TestNShot(self, model, N, k, Shots, X, X_classes, verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {}-Way {}-shot learning tasks ... \n".format(k, N, Shots))
        for i in range(k):
            inputs, targets = self.MakeNShotTask(N, Shots, X, X_classes)
            # prob_all - loop shots...
            probs = model.predict(inputs)

            targetsIndx = np.where(targets == 1)

            if np.argmax(probs) in np.asarray(targetsIndx)[0]:
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {}-Way {}-shot learning accuracy \n".format(percent_correct, N, Shots))
        return percent_correct

    def make_oneshot_task(self, N,  X, X_classes, language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        categories = X_classes
        n_classes, n_examples, w, h, channels = X.shape

        indices = rng.randint(0, n_examples, size=(N,))
        if language is not None:  # if language is specified, select characters for that language
            low, high = categories[language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            categories = rng.choice(range(low, high), size=(N,), replace=False)

        else:  # if no language specified just pick a bunch of random letters
            categories = rng.choice(range(n_classes), size=(N,), replace=False)
        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
        test_image = np.asarray([X[true_category, ex1, :, :]] * N).reshape(N, w, h, channels)
        support_set = X[categories, indices, :, :]
        support_set[0, :, :] = X[true_category, ex2]
        support_set = support_set.reshape(N, w, h, channels)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]

        #print(targets)
        return pairs, targets

    def test_oneshot(self, model, N, k,  X, X_classes, verbose = 0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N, X, X_classes)
            # prob_all - loop shots...
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
        return percent_correct

    def save_images_oneshot(self, model, N, k,  X, X_classes, verbose = 0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N, X, X_classes)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
        return percent_correct

    def pickleImages(self, data_folder, save_path, filename="train.pickle"):
        X, y, c = self.loadimgs(data_folder)
        with open(os.path.join(save_path, filename), "wb") as f:
            #pickle.dump((X, c), f, protocol=4)
            joblib.dump((X, c), f)

    def loadPickledData(self, pickledPath, filename="train.pickle"):
        with open(os.path.join(pickledPath, filename), "rb") as f:
            #Xtrain, train_classes = pickle.load(f)
            Xtrain, train_classes = joblib.load(f)
        return Xtrain, train_classes

    def progress(self, prog_count, evaluate_every, suffix=''):
        bar_len = 60
        filled_len = int(round(bar_len * prog_count / float(evaluate_every)))

        percents = round(100.0 * prog_count / float(evaluate_every), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        print('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))

    def evaluate(self, weightFilename, NumOfEval):
        # build the model
        accuracy = []
        if self.config.BACKBONE == "siamese":
            model = self.get_siamese_model(
                (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        if self.config.BACKBONE == "ResNet18":
            model = self.getResNet18((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        if self.config.BACKBONE == "EfficientNetB0":
            model = self.getEfficientNetB0((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        if self.config.BACKBONE == "MobileNetV2":
            model = self.getMobileNetV2((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))

        model.summary()

        # Load the test data
        Xval, val_classes = self.loadPickledData(pickledPath=self.config.DATASET_FOLDER,
                                                 filename=self.config.VAL_PICKLE_FILE)
        print("Evaluating: " + self.config.DATASET + " with " + self.config.BACKBONE + " backbone and " + self.config.SIMILARITY_METRIC + " similarity metric", end="\n\n")
        print(list(val_classes.keys()))

        # load the wrights
        model.load_weights(weightFilename)
        # evaluate the model
        for i in range(NumOfEval):
            print("Evaluating " + str(i+1) + " of " + str(NumOfEval) + " ...")
            val_acc = self.TestNShot(model, self.config.N_WAY, self.config.N_VAL, self.config.Shots, X=Xval, X_classes=val_classes,
                                        verbose=True)

            accuracy.append(val_acc)

        # calculate the mean and sd
        accuracy = np.array(accuracy)
        average = np.mean(accuracy)
        sd = np.std(accuracy)

        filename = "weights/average_evaluation.csv"
        if os.path.isfile(filename):
            csvfile = open(filename, 'a')
        else:
            csvfile = open(filename, 'w')
            csvLine = "DATASET, BACKBONE, SIMILARITY_METRIC, AVERAGE, STD, Samples# \n"
            csvfile.write(csvLine)

        csvLine = str(self.config.DATASET) + "," + str(self.config.BACKBONE) + "," + str(self.config.SIMILARITY_METRIC) + "," + str(average) + "," + str(sd) +  "," + str(NumOfEval) + "\n"
        csvfile.write(csvLine)
        csvfile.close()

    def predict(self, weightFilename, inputs_set, target_set):
        # build the model
        if self.config.BACKBONE == "siamese":
            model = self.get_siamese_model(
                (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        if self.config.BACKBONE == "ResNet18":
            model = self.getResNet18((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        if self.config.BACKBONE == "EfficientNetB0":
            model = self.getEfficientNetB0((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        if self.config.BACKBONE == "MobileNetV2":
            model = self.getMobileNetV2((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        model.summary()

        # load the wrights
        model.load_weights(weightFilename)

        # create the prediction folder
        if not os.path.exists(self.config.MODEL_PATH + "/predictions"):
            os.makedirs(self.config.MODEL_PATH + "/predictions")

        for i, inputs in enumerate(inputs_set):
            targets = target_set[i]
            indx = np.where(targets == 1)[0]

            probs = model.predict(inputs)
            pairs = np.array(inputs[0]).shape[0] # number of pairs

            #create folder Nth-way test
            if not os.path.exists(self.config.MODEL_PATH + "/predictions/" + str(i)):
                os.makedirs(self.config.MODEL_PATH + "/predictions/" + str(i))

            # Save the prob as csv
            #targets
            np.savetxt(self.config.MODEL_PATH + "/predictions/" + str(i) + "/targets.csv", targets, delimiter=",")

            # save the targets as csv
            #probs
            np.savetxt(self.config.MODEL_PATH + "/predictions/" + str(i) + "/preditions.csv", probs, delimiter=",")

            horizontal_images = []
            test_image=[]
            for pair in range(pairs):
                # save the test image
                if pair == 0:
                    test_image = np.array(inputs[0][pair]) * 255
                    test_image = test_image.astype(np.uint8)
                    test_image = cv2.copyMakeBorder(test_image, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=[255, 255, 255])

                # save the support image
                support_image = np.array(inputs[1][pair]) * 255
                support_image = support_image.astype(np.uint8)
                if targets[pair] == 1:
                    support_image = cv2.copyMakeBorder(support_image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 255])
                    support_image = cv2.copyMakeBorder(support_image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                else:
                    support_image = cv2.copyMakeBorder(support_image, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                horizontal_images.append(support_image)

            probs, horizontal_images = zip(*sorted(zip(probs, horizontal_images), key=lambda x: x[0], reverse=True))

            images = []
            images.append(test_image)
            for horizontal_image in horizontal_images:
                images.append(horizontal_image)

            hConcat_image = np.concatenate((images), axis=1)
            cv2.imwrite(self.config.MODEL_PATH + "/predictions/" + str(i) + "/prediction.png", hConcat_image)




    def train(self):
        # Check if model folder exists and create it
        if not os.path.exists(self.config.MODEL_PATH):
            os.makedirs(self.config.MODEL_PATH)

        # Build the model
        if self.config.BACKBONE == "siamese":
            model = self.get_siamese_model((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        if self.config.BACKBONE == "ResNet18":
            model = self.getResNet18((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        if self.config.BACKBONE == "EfficientNetB0":
            model = self.getEfficientNetB0((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        if self.config.BACKBONE == "MobileNetV2":
            model = self.getMobileNetV2((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNELS))
        model.summary()

        optimizer = Adam(lr = self.config.LR)
        model.compile(loss="binary_crossentropy",optimizer=optimizer)

        # Loading the train and val data
        Xtrain, train_classes = self.loadPickledData(pickledPath=self.config.DATASET_FOLDER, filename=self.config.TRAIN_PICKLE_FILE)
        print("Training alphabets:", end="\n\n")
        print(list(train_classes.keys()))
        Xval, val_classes = self.loadPickledData(pickledPath=self.config.DATASET_FOLDER, filename=self.config.VAL_PICKLE_FILE)
        print("Validation alphabets:", end="\n\n")
        print(list(val_classes.keys()))


        best_accuracy = -1
        lr = self.config.LR
        patience = self.config.PATIENCE
        bestFilename = None
        csvfile = open(self.config.MODEL_PATH + "/" + self.config.DATASET + "_" + self.config.BACKBONE + "_" + self.config.SIMILARITY_METRIC + "_best_results.csv", 'w')
        csvLine = "bestAccuracy, bestFilename, Train_Loss, LearningRate\n"
        csvfile.write(csvLine)
        csvfile.close()
        print("Starting training process!")
        print("-------------------------------------")
        t_start = time.time()
        progress_bar = 0
        for i in range(1, self.config.EPOCHS+1):
            progress_bar += 1
            (inputs,targets) = self.get_batch(self.config.BATCH_SIZE, X=Xtrain, X_classes=train_classes)

            loss = model.train_on_batch(inputs, targets)

            self.progress(progress_bar, self.config.EVALUATE_INTERVAL, suffix="")
            if i % self.config.EVALUATE_INTERVAL == 0:
                progress_bar = 0
                print("\n ------------- \n")
                print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
                print("Train Loss: {0}".format(loss))
                print("Current best accuracy: {}%".format(best_accuracy))
                val_acc = self.test_oneshot(model, self.config.N_WAY, self.config.N_VAL, X=Xval, X_classes=val_classes, verbose=True)
                weightFilename = os.path.join(self.config.MODEL_PATH, 'weights.{}.h5'.format(i))
                if val_acc >= best_accuracy:
                    print("Current best: {0}, previous best: {1}".format(val_acc, best_accuracy))
                    best_accuracy = val_acc
                    bestFilename = weightFilename
                    model.save_weights(bestFilename)
                    csvfile = open(
                        self.config.MODEL_PATH + "/" + self.config.DATASET + "_" + self.config.BACKBONE + "_" + self.config.SIMILARITY_METRIC + "_best_results.csv",
                        'a')
                    csvLine = str(best_accuracy) + "," + str(bestFilename) + "," + str(loss) + "," + str(lr) + "\n"
                    csvfile.write(csvLine)
                    csvfile.close()
                    patience = self.config.PATIENCE
                else:
                    patience -= 1
                    # reduce the learning rate if patience is zero or less
                    if patience <= 0:
                        # reduce learning rate
                        lr = lr * self.config.LR_FACTOR
                        K.set_value(model.optimizer.learning_rate, lr)
                        patience = self.config.PATIENCE

                        if lr <= self.config.MIN_LR:
                            break


if __name__ == '__main__':
    # Evaluation
    # ===============
    eval = True
    # ===============

    if eval:

        NumOfEval = 600

        Backbone="MobileNetV2" # Resnet18 / EfficientNetB0 / MobileNetV2
        imageShape = (105, 105, 3)
        datasets = ["caltech256"]  # ["cifar100", "CUB_200_2011", "miniImageNet", "caltech256"]
        similarityMetrics = ["max", "concat"]  # ["L1", "L2", "max", "concat", "cosine"]

        for i, dataset in enumerate(datasets):
            for similarityMetric in similarityMetrics:
                # All files ending with .h5
                weightFolder = "weights/" + dataset + "/" + Backbone + "/" + similarityMetric
                weightFilenames = glob.glob(weightFolder + "/*.h5")
                weightFilename = weightFilenames[-1]

                oneShot = onShotLearnig(Backbone, similarityMetric, imageShape, dataset)
                oneShot.evaluate(weightFilename, NumOfEval)

    else:
        # Training
        imageShape = [(105, 105, 3)] #[(128, 128, 3), (128, 128, 3), (128, 128, 3), (105, 105, 3)]

        datasets = ["caltech256"] #["cifar100", "CUB_200_2011", "miniImageNet", "caltech256"]
        Backbones = ["EfficientNetB0"] # ["ResNet18", "siamese", EfficientNetB0]

        similarityMetrics = ["max"] #["L1", "L2", "max", "concat", "cosine"]

        for i, dataset in enumerate(datasets):
            for similarityMetric in similarityMetrics:
                print("=============================================")
                print("Dataset: ", dataset, "Similarity Metric: ", similarityMetric)
                print("=============================================")

                DATASET_FOLDER = "datasets/" + dataset + "/data/"
                TRAIN_FOLDER = "datasets/" + dataset + "/train/"
                TEST_FOLDER = "datasets/" + dataset + "/val/"
                train_filename = "train.pickle"
                val_filename = "val.pickle"

                oneShot = onShotLearnig(Backbones[i], similarityMetric, imageShape[i], dataset)

                if not os.path.exists(os.path.join(DATASET_FOLDER, train_filename)):
                    oneShot.pickleImages(data_folder=TRAIN_FOLDER, save_path=DATASET_FOLDER, filename="train.pickle")
                if not os.path.exists(os.path.join(DATASET_FOLDER, val_filename)):
                    oneShot.pickleImages(data_folder=TEST_FOLDER, save_path=DATASET_FOLDER, filename="val.pickle")

                oneShot.train()






