import tensorflow as tf
from tensorflow.keras import Model, layers


def build_autoencoder(input_shape, encoding_dim, batch_size):
    """
    Constructs a convolutional autoencoder for images with a normalization preprocessing layer.

    Args:
        input_shape (tuple): Dimensions of the images (height, width, channels).
        encoding_dim (int): Dimension of the latent space.

    Returns:
        tuple: Autoencoder and encoder (Keras models).
    """
    # Input
    inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size)

    # # Normalization Layer
    # # Normalize pixel values to [0, 1]
    # normalized = layers.Rescaling(1.0 / 255)(inputs)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu",
                      padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Bottleneck
    shape_before_flattening = x.shape[1:]  # Static shape as a tuple
    flattened_dim = tf.reduce_prod(
        shape_before_flattening).numpy()  # Convert to scalar
    x = layers.Flatten()(x)
    encoded = layers.Dense(encoding_dim, activation="relu")(x)

    # Decoder
    x = layers.Dense(flattened_dim, activation="relu")(encoded)
    x = layers.Reshape(
        (shape_before_flattening[0],
         shape_before_flattening[1], shape_before_flattening[2])
    )(x)
    x = layers.Conv2DTranspose(
        128, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(
        64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=2, padding="same", activation="relu")(x)
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    return autoencoder, encoder
