import numpy as np
import tensorflow as tf

from util.config import Config


def create_synthetic_dataset(batch_size=1, avg_spectrogram_length=565, avg_transcript_length=50, length=None):
    def gen():
        while True:
            filenames = ["dummy"] * batch_size
            features = np.random.uniform(low=-1, high=+1, size=(batch_size, avg_spectrogram_length, Config.n_input))
            dims = [avg_spectrogram_length] * batch_size

            transcripts = np.random.randint(low=0, high=26, size=(batch_size, avg_transcript_length))

            yield (filenames, (features, dims), transcripts)

    ds = tf.data.Dataset.from_generator(gen, (tf.string, (tf.float32, tf.int32), tf.int32))

    def transcript_to_sparse(files, features, transcripts):
        return files, features, tf.sparse.from_dense(transcripts)

    ds = ds.map(transcript_to_sparse)

    if length is not None:
        ds = ds.take(length)

    return ds
