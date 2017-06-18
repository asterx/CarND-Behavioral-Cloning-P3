# coding=utf-8
import os
from net.net import net
from keras import backend
from utils.data import test_split, batches
from utils.logger import Logger

import warnings
warnings.filterwarnings('ignore')

TRAIN_SIZE = 0.8
EPOCHS = 20

# Split data into training and validation sets
data_train, data_val = test_split('./data/driving_log_fixed.csv', train_size = TRAIN_SIZE)
print("Initial data; train: {}, validate: {}".format(len(data_train), len(data_val)))


# Training and validating
logits = net()
logits.fit_generator(
        batches(data_train),
        samples_per_epoch = data_train.shape[0],
        nb_epoch = EPOCHS,
        validation_data = batches(data_val, augment = False),
        nb_val_samples = data_val.shape[0],
        callbacks = [Logger()],
    )

# Save model
with open('./model/model.json', 'w') as file:
    file.write(logits.to_json())

backend.clear_session()
