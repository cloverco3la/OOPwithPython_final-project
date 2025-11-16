from tensorflow import keras
from .bicoder import Encoder, Decoder
import pickle

class VAE(keras.Model):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self,x, training = False):
        mu, log_var = self.encoder(x)
        if training:
            z = self.encoder.sample(mu, log_var)
        else:
            z = mu
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

