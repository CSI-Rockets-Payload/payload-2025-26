import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
import matplotlib.pyplot as plt
import numpy as np


# Library Implementation of 1D CNN 

class CAE1D(Model):
    def __init__(self, 
                input_dim: int = 52,
                enc_channels = (16, 32, 32),
                enc_kernels  = (4, 4, 1),
                enc_strides = (2, 2, 1),
                dec_channels = (32, 16, 1),
                dec_kernels  = (3, 3, 3),
                dec_strides = (2, 2, 1),
                num_classes = 2,
                activation = "tanh",
                output_activation = "softmax"
    ):
        """
        Parameters of the CNN
        Implements the 1D Convolutional Autoencoder architecture
        from Chen, Yu & Wang (2020) - Journal of Process Control.
        """
        super().__init__()
        self.input_dim = input_dim
        self.enc_channels = enc_channels
        self.enc_kernels = enc_kernels
        self.enc_strides = enc_strides
        self.dec_channels = dec_channels
        self.dec_kernels = dec_kernels
        self.dec_strides = dec_strides
        self.num_classes = num_classes
        self.activation = activation
        self.output_activation = output_activation
        
        # Architecture building blocks
        self.encoder_net = self._encoder()
        self.decoder_net = self._decoder()
        self.classifier_net = self._classifier()

    
    # Phase 1: unsupervised
    def _encoder(self):
        """
        Build the encoder: Input vector -> convolution layers
        """
        inp = layers.Input(shape=(self.input_dim, 1))
        x = inp
        for c, k, s in zip(self.enc_channels, self.enc_kernels, self.enc_strides):
            x = layers.Conv1D(c, k, strides=s, padding="same",
                              activation=self.activation)(x)
        return Model(inp, x, name="encoder_1dcae")

    def _decoder(self):
        """
        Build the decoder: De-convolution layer -> output
        """
        inp = layers.Input(shape=(None, self.enc_channels[-1]))
        x = inp
        for c, k, s in zip(self.dec_channels, self.dec_kernels, self.dec_strides):
            # mirror deconvolutions with transposed convs
            # x = layers.Conv1DTranspose(c, k, strides=s, padding="same",
            #                            activation=self.activation)(x)
            x = layers.UpSampling1D(size= s)(x)
            x = layers.Conv1D(c, k, padding='same', activation=self.activation)(x)
        out = layers.Activation(self.output_activation, name="reconstruction")(x)
        return Model(inp, out, name="decoder_1dcae")
    
    def build_cnn1d_autoencoder(self):
        """
        Autoencoder architecture for phase 1
        """
        inp = layers.Input(shape=(self.input_dim, 1))
        z = self.encoder_net(inp)
        x_hat = self.decoder_net(z)
        ae = Model(inp, x_hat, name="1DCAE_unsupervised")
        ae.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
        return ae
    
    # Phase 2: supervised
    def _classifier(self):
        """
        Fine-tuning classifer for phase 2
        """
        latent_channels = self.enc_channels[-1]
        inp = layers.Input(shape=(None, latent_channels))
        x = layers.GlobalAveragePooling1D()(inp)
        out = layers.Dense(self.num_classes, activation="softmax")(x)
        return Model(inp, out, name="classifier")

    # Model training 
    def train_two_phase(self, X_train, y_train, X_val=None, y_val=None,
                        epochs_unsup=100, epochs_sup=50, batch_size=32):
        """
        Phase I : Unsupervised reconstruction training (autoencoder)
        Phase II : Fine-tuning classification head on latent features
        """

        # ---------- Phase I: Autoencoder ----------
        print("Phase I - Unsupervised training (Autoencoder)...")
        ae = self.build_cnn1d_autoencoder()
        hist1 = ae.fit(
            X_train, X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=epochs_unsup,
            batch_size=batch_size,
            verbose=1
        )

        # ---------- Phase II: Classifier ----------
        print("\nPhase II - Fine-tuning (Classifier)...")

        # Optionally allow encoder to be fine-tuned (here we just use it as a feature extractor)
        self.encoder_net.trainable = True

        # Encode data once using the trained encoder
        Z_train = self.encoder_net.predict(X_train, batch_size=batch_size)
        if X_val is not None:
            Z_val = self.encoder_net.predict(X_val, batch_size=batch_size)
        else:
            Z_val = None

        # Use the classifier built in __init__ via _classifier()
        clf = self.classifier_net
        clf.compile(
            optimizer=optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        hist2 = clf.fit(
            Z_train, y_train,
            validation_data=(Z_val, y_val) if Z_val is not None else None,
            epochs=epochs_sup,
            batch_size=batch_size,
            verbose=1
        )

        return ae, clf, hist1, hist2


    def visualize(self, hist1=None, hist2=None):
        """
        Visualize training curves for phase 1 and 2
        """
        plt.figure(figsize=(12, 5))

        # Phase I: Unsupervised training (Reconstruction loss)
        if hist1 is not None:
            plt.subplot(1, 2, 1)
            plt.plot(hist1.history['loss'], label='Train Loss')
            if 'val_loss' in hist1.history:
                plt.plot(hist1.history['val_loss'], label='Val Loss')
            plt.title("Phase I: Reconstruction Loss")
            plt.xlabel("Epochs (N iterations)")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

        # Phase II: Fine-tuning (Classification)
        if hist2 is not None:
            plt.subplot(1, 2, 2)
            if 'accuracy' in hist2.history:
                plt.plot(hist2.history['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in hist2.history:
                plt.plot(hist2.history['val_accuracy'], label='Val Accuracy')
            plt.title("Phase II: Fine-tuning")
            plt.xlabel("Epochs (M iterations)")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Building dummy CAE1D model...")

    # dummy data (e.g., 128 samples, 52 time steps, 1 feature)
    X = np.random.rand(128, 52, 1).astype(np.float32)

    # IMPORTANT: match num_classes with your model
    num_classes = 2
    y_int = np.random.randint(0, num_classes, 128)
    y = tf.keras.utils.to_categorical(y_int, num_classes=num_classes)

    model = CAE1D(input_dim=52, num_classes=num_classes)

    # Run a short 2-phase training just to get histories
    ae, clf, hist1, hist2 = model.train_two_phase(
        X_train=X,
        y_train=y,
        X_val=None,
        y_val=None,
        epochs_unsup=3,    # keep tiny so it runs fast
        epochs_sup=3,
        batch_size=16
    )

    # Visualize both phases
    model.visualize(hist1=hist1, hist2=hist2)

    ae = model.build_cnn1d_autoencoder()
    ae.summary()
    out = ae.predict(X)
    print("Output shape:", out.shape)
    # print("Building dummy CAE1D model...")

    # # dummy data (8 samples, 52 time steps, 1 feature)
    # X = np.random.rand(8, 52, 1).astype(np.float32)
    # y = tf.keras.utils.to_categorical(np.random.randint(0, 3, 8))

    # model = CAE1D()
    # ae = model.build_cnn1d_autoencoder()
    # ae.summary()
    # out = ae.predict(X)
    # print("Output shape:", out.shape)