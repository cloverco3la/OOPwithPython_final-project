import tensorflow as tf

class BiCoder(tf.keras.layers.Layer):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def call(self,x):
        raise NotImplementedError

class Encoder(BiCoder):
    def __init__(self, network):
        super().__init__(network)

    def sample(self, mu, log_var):
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def call(self,x):
        out = self.network(x)
        mu, log_var = tf.split(out, 2, axis=1) 
        return mu, log_var
    
class Decoder(BiCoder):
    def __init__(self, network):
        super().__init__(network)
    
    def call(self,z):
        return self.network(z)

