import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_shape, encoding_dim):
    """
    Constructs a convolutional autoencoder for images.

    Args:
        input_shape (tuple): Dimensions of the images (height, width, channels).
        encoding_dim (int): Dimension of the latent space.

    Returns:
        tuple: Autoencoder and encoder (Keras models).
    """
    # Encoder
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Bottleneck
    # Get the shape excluding the batch dimension
    shape_before_flattening = tf.shape(x)[1:]
    x = layers.Flatten()(x)
    encoded = layers.Dense(encoding_dim, activation="relu")(x)

    # Decoder
    x = layers.Dense(tf.reduce_prod(shape_before_flattening),
                     activation="relu")(encoded)
    x = layers.Reshape(
        (shape_before_flattening[0], shape_before_flattening[1], shape_before_flattening[2]))(x)
    x = layers.Conv2DTranspose(
        128, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(
        64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=2, padding="same", activation="relu")(x)
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)

    return autoencoder, encoder


def train_autoencoder(autoencoder, X, epochs=50, batch_size=64, validation_split=0.1):
    """
    Treina o autoencoder no dataset.

    Args:
        autoencoder (Model): Modelo Keras do autoencoder.
        X (ndarray): Dados de entrada para o treinamento.
        epochs (int): Número de épocas.
        batch_size (int): Tamanho do batch.
        validation_split (float): Proporção dos dados para validação.

    Returns:
        Model: Autoencoder treinado.
    """
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size,
                    validation_split=validation_split)
    return autoencoder


def cluster_latent_space(encoder, X, min_clusters=2, max_clusters=10):
    """
    Realiza a clusterização do espaço latente usando K-Means.

    Args:
        encoder (Model): Modelo Keras do encoder para extrair as características latentes.
        X (ndarray): Dados de entrada.
        min_clusters (int): Número mínimo de clusters a considerar.
        max_clusters (int): Número máximo de clusters a considerar.

    Returns:
        dict: Resultados contendo `best_k`, `silhouette_scores`, e `final_labels`.
    """
    latent_features = encoder.predict(X)
    silhouette_scores = []
    possible_k = range(min_clusters, max_clusters + 1)

    for k in possible_k:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(latent_features)
        score = silhouette_score(latent_features, labels)
        silhouette_scores.append(score)

    best_k = possible_k[np.argmax(silhouette_scores)]
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_kmeans.fit_predict(latent_features)

    return {
        "best_k": best_k,
        "silhouette_scores": silhouette_scores,
        "final_labels": final_labels,
    }


def plot_silhouette_scores(silhouette_scores, min_clusters=2):
    """
    Plota os scores de silhouette para diferentes números de clusters.

    Args:
        silhouette_scores (list): Lista de scores de silhouette.
        min_clusters (int): Número mínimo de clusters considerado.
    """
    possible_k = range(min_clusters, min_clusters + len(silhouette_scores))
    plt.plot(possible_k, silhouette_scores, marker="o")
    plt.title("Escolha do número de clusters")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.show()
