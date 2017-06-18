# coding=utf-8
import os, shutil
from keras.callbacks import Callback

class Logger(Callback):
    def __init__(self, path = './model'):
        super(Logger, self).__init__()
        self.path = path
        shutil.rmtree(self.path, ignore_errors = True)
        os.makedirs(self.path, exist_ok = True)

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.path, 'epoch_{}.h5'.format(epoch)))
