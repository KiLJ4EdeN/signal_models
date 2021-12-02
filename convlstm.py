import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM1D, MaxPooling1D, TimeDistributed


inputs = tf.keras.layers.Input(shape=(NUM_SEQ, SEQUENCE_LENGH, 1))

x = ConvLSTM1D(filters=8, kernel_size=9, strides=1,
               padding='valid', return_sequences=True)(inputs)
x = TimeDistributed(MaxPooling1D(pool_size=2))(x)

x = ConvLSTM1D(filters=16, kernel_size=7, strides=1,
               padding='valid', return_sequences=True)(x)
x = TimeDistributed(MaxPooling1D(pool_size=2))(x)

x = ConvLSTM1D(filters=32, kernel_size=5, strides=1,
               padding='valid', return_sequences=False)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=inputs,
                              outputs=x, name='convlstm')
print(model.summary())
