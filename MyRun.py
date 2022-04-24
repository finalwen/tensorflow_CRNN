# encoding = "utf8"
import tensorflow as tf
import argparse
from Model import CRNN
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

crnn = CRNN(batch_size=32,
            init_learning_rate=0.1,
            epochs=1000,
            dataset_path="D:/tmp/lstm_ctc_data2_tfrecord/train_dataset.tfrecord",
            early_stopping_step=2000,
            model_dir="./model",
            checkpoint_dir="./saver"
            )
crnn.train()
