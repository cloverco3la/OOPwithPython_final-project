import argparse as ap
import classes.dataloader as d
from classes.bicoder import Encoder, Decoder
import tensorflow as tf
from classes.vae import VAE
from neural_networks import encoder_mlp, decoder_mlp, encoder_conv, decoder_conv
import utils as u
from losses import kl_divergence 


def compute_loss(x, x_hat, mu, log_var):
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    x_hat = tf.reshape(x_hat, [tf.shape(x_hat)[0], -1])
    recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
    kl = tf.reduce_mean(kl_divergence(mu, log_var))
    return recon + kl, recon, kl

@tf.function
def train_step(vae, x, optimizer):
    with tf.GradientTape() as tape:
        x_hat, mu, log_var = vae(x, training=True)
        loss, recon, kl = compute_loss(x, x_hat, mu, log_var)

    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    return loss, recon, kl

parser = ap.ArgumentParser(description="Test or train VAE model")
parser.add_argument("--mode", type=str, default="test", choices=["test", "train"], help="Choose 'test' for running a check, or 'train' for training.")
parser.add_argument("--dset", type=str, default="mnist_bw", choices=["mnist_bw", "mnist_color"], help="Data set to use.")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--visualize_latent", action="store_true",help="Visualize latent space with t-SNE after training.")
parser.add_argument("--generate_from_prior", action="store_true",help="Generate samples from prior after traning.")
parser.add_argument("--generate_from_posterior", action="store_true")

arg = parser.parse_args()

if arg.dset == "mnist_bw":
    encoder_net = Encoder(encoder_mlp)
    decoder_net = Decoder(decoder_mlp)
else:
    encoder_net = Encoder(encoder_conv)
    decoder_net = Decoder(decoder_conv)

enc = encoder_net
dec = decoder_net
vae = VAE(enc, dec)
optimizer = u.optimizer

loader = d.DataLoader(arg.dset, color_version="m3" )
loader.download()
loader.load_data()
train_dataset = loader.get_dataset()

if arg.mode == "test":
    x_np = loader.x
    x_batch = tf.convert_to_tensor(x_np[:100])
    x_hat, mu, log_var = vae(x_batch,training = False)

    print(f"Test forward shapes:\n x_hat: {x_hat.shape}, mu: {mu.shape}, log_var: {log_var.shape}")

else:
    for epoch in range(1, arg.epochs + 1):
        for batch in train_dataset:
            batch = tf.convert_to_tensor(batch)
            loss, recon, kl = train_step(vae, batch, optimizer)
        
        print(f"[Epoch {epoch}] loss={loss:.4f} recon={recon:.4f} kl={kl:.4f}")
