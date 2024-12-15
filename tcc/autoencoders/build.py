import tensorflow as tf
from tensorflow.keras import Model, layers


def build_autoencoder(input_shape, encoding_dim, batch_size):
    """
    Constructs a convolutional autoencoder for images with named encoder and decoder parts.

    Args:
        input_shape (tuple): Dimensions of the images (height, width, channels).
        encoding_dim (int): Dimension of the latent space.
        batch_size (int): Batch size for input layer.

    Returns:
        tuple: (autoencoder, encoder) Keras models.
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size, name="encoder_input")

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_conv3")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool3")(x)

    # Bottleneck
    shape_before_flattening = x.shape[1:]
    flattened_dim = tf.reduce_prod(shape_before_flattening).numpy()
    x = layers.Flatten(name="enc_flatten")(x)
    encoded = layers.Dense(encoding_dim, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense(flattened_dim, activation="relu", name="dec_dense")(encoded)
    x = layers.Reshape((shape_before_flattening[0],
                        shape_before_flattening[1],
                        shape_before_flattening[2]), name="dec_reshape")(x)
    x = layers.Conv2DTranspose(
        128, (3, 3), strides=2, padding="same", activation="relu", name="dec_convT1")(x)
    x = layers.Conv2DTranspose(
        64, (3, 3), strides=2, padding="same", activation="relu", name="dec_convT2")(x)
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=2, padding="same", activation="relu", name="dec_convT3")(x)
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="decoder_output")(x)

    autoencoder = Model(inputs, decoded, name="autoencoder_model")
    encoder = Model(inputs, encoded, name="encoder_model")

    return autoencoder, encoder
