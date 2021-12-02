import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed


inputs = tf.keras.layers.Input(shape=(NUM_SEQ, SEQUENCE_LENGH, 1))
x = TimeDistributed(tf.keras.layers.Conv1D(filters=4, kernel_size=9, padding='valid'))(inputs)
x = TimeDistributed(tf.keras.layers.Conv1D(filters=4, kernel_size=9, padding='valid'))(x)
x = TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2))(x)
x = TimeDistributed(tf.keras.layers.Conv1D(filters=8, kernel_size=7, padding='valid'))(x)
x = TimeDistributed(tf.keras.layers.Conv1D(filters=8, kernel_size=7, padding='valid'))(x)
x = TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2))(x)
x = TimeDistributed(tf.keras.layers.Conv1D(filters=16, kernel_size=5, padding='valid'))(x)
x = TimeDistributed(tf.keras.layers.Conv1D(filters=16, kernel_size=5, padding='valid'))(x)
x = TimeDistributed(tf.keras.layers.Flatten())(x)
x = tf.keras.layers.LSTM(48)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=inputs,
                              outputs=x, name='cnnlstm')
print(model.summary())
