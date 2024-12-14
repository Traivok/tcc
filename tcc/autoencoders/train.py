import wandb
from wandb.keras import WandbCallback


def train_autoencoder(autoencoder, X, epochs=50, batch_size=64, validation_split=0.1):
    """
    Trains the autoencoder on the dataset.

    Args:
        autoencoder (Model): Keras model of the autoencoder.
        X (ndarray): Input data for training.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        validation_split (float): Proportion of data for validation.

    Returns:
        Model: Trained autoencoder.
    """
    # Initialize wandb
    wandb.init(project="tcc_autoencoder")

    # Compile the model
    autoencoder.compile(optimizer="adam", loss="mse")

    # Train the model with WandbCallback
    autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[WandbCallback()]
    )

    return autoencoder
