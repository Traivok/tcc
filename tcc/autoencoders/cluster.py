import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
