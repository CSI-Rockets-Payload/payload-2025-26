import tensorflow as tf
from tensorflow.keras import layers, Model

# Library Implementation of 1D CNN 

class CNN1D(Model):
    def __init__(self, 
                in_channels: int = 2,
                enc_channels = (16, 32, 64),
                enc_kernels  = (5, 5, 3),
                dec_channels = (64, 32, 16),
                dec_kernels  = (3, 3, 3),
                latent_dim: int = 16
    ):
        """
        Parameters of the CNN
        To add more for hyperparameter tuning later...
        """
        self.in_channels = in_channels
        self.enc_channels = enc_channels
        self.enc_kernels = enc_kernels
        self.dec_channels = dec_channels
        self.dec_kernels = dec_kernels
        self.latent_dim = latent_dim

    
    def encoder(self):
        """
        Build the encoder: (T, C) -> (latent_dim,)
        2 convs, GAP, then Dense(latent_dim).
        """
        inp = layers.Input(shape=(None, self.in_channels))   
        x = inp
        for c in self.enc_channels:
            x = layers.Conv1D(filters=c, kernel_size=self.enc_kernels,
                            strides=2, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        z = layers.Dense(self.latent_dim, name="latent")(x)
        return Model(inp, z, name="encoder_1d_cnn")
    

    def decoder(self):
        """
        Build the decoder: (latent_dim,) -> (~T, C)
        """
        inp = layers.Input(shape=(None, self.enc_channels))  
        x = layers.Conv1D(32, self.dec_kernels, padding="same", activation="relu")(inp)
        out = layers.Conv1D(self.dec_channels, self.dec_kernels, padding="same", activation=None, name="reconstruction")(x)
        return Model(inp, out, name="decoder_1d_cnn")
    
    def build_cnn1d_autoencoder(self):
        """
        Autoencoder: (T, C) -> (T, C)
        """
        enc = self.encoder()
        dec = self.decoder()

        inp = layers.Input(shape=(None, self.enc_channels))
        z = enc(inp)          
        x_hat = dec(z)         
        ae = Model(inp, x_hat, name="autoencoder_1d_cnn_simple")
        return enc, ae
    
    def fit(): 
        """
        Fit model
        - Loss function: Mean Squared Error (MSE)
        - Specify number of epochs 
        """

    def visualize():
        """
        Plot loss over training 
        """
