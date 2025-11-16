from neural_networks import encoder_mlp, decoder_mlp, encoder_conv, decoder_conv
from classes.bicoder import Encoder, Decoder
import tensorflow as tf
from classes.vae import VAE
from losses import kl_divergence 
import argparse as ap
import subprocess as s
import utils as u
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle


def download(url, outpath):
    print(f"Downloading as {outpath}")
    try:
        s.run(["wget", "-O", outpath, url], check=True)
    except s.CalledProcessError:
        print("Could not download the file!")
        exit(1)

def load_test_img(path, is_color):
    if not is_color:
        x_np = np.load(path)
        x_np = x_np.astype("float32")/255.0
        x_np = x_np.reshape(-1, 28*28)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)  

        if isinstance(data, dict):
            x_list = [data[i] for i in sorted(data.keys())]
            x_np = np.concatenate(x_list, axis=0)
        
        else:
            x_np = data
        
        x_np = x_np.astype("float32") / 255.0

    return x_np

def load_train_img(path, is_color):
    if not is_color:
        x_np = np.load(path)
        x_np = x_np.astype("float32") / 255.0
        x_np = x_np.reshape(-1, 28 * 28)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            x_list = [data[i] for i in sorted(data.keys())]
            x_np = np.concatenate(x_list, axis=0)
        
        else:
            x_np = data
        
        x_np = x_np.astype("float32") / 255.0

    return x_np

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


parser = ap.ArgumentParser(description="Train VAE model")
parser.add_argument("--dset", type=str, default="mnist_bw", choices=["mnist_bw", "mnist_color"], help="Data set to use.")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--visualize_latent", action="store_true",help="Visualize latent space with t-SNE after training.")
parser.add_argument("--generate_from_prior", action="store_true",help="Generate samples from prior after traning.")
parser.add_argument("--generate_from_posterior", action="store_true")

arg = parser.parse_args()


if arg.dset == "mnist_bw":
    test_file = "data_sets/mnist_bw_te.npy"
    train_file = "data_sets/mnist_bw.npy"
    label_file = "data_sets/mnist_bw_y_te.npy"

    test_url = u.bw_test_url
    train_url = u.bw_train_url
    label_url = u.bw_label_url

    encoder_net = Encoder(encoder_mlp)
    decoder_net = Decoder(decoder_mlp)
    is_color = False
else:
    test_file = "data_sets/mnist_color_te.pkl"
    train_file = "data_sets/mnist_color.pkl"
    label_file = "data_sets/mnist_color_y_te.npy"

    test_url = u.color_test_url
    train_url = u.color_train_url
    label_url = u.color_label_url

    encoder_net = Encoder(encoder_conv)
    decoder_net = Decoder(decoder_conv)
    is_color = True

enc = encoder_net
dec = decoder_net
vae = VAE(enc, dec)
optimizer = u.optimizer


print("__TRAINING VAE__")

download(train_url, train_file)

x_np = load_train_img(train_file, is_color)
x_np = x_np[:50000]

dataset = tf.data.Dataset.from_tensor_slices(x_np).shuffle(50000).batch(128)
    
for epoch in range(1, arg.epochs + 1):
    for batch in dataset:
        batch = tf.convert_to_tensor(batch)
        loss, recon, kl = train_step(vae, batch, optimizer)
    print(f"[Epoch {epoch}] loss={loss:.4f} recon={recon:.4f} kl={kl:.4f}")

x_sample = tf.convert_to_tensor(x_np[:100])
_, mu_sample, _ = vae(x_sample,training = False)
latent_dim = mu_sample.shape[1]

x_test = None
y_test = None

if arg.visualize_latent or arg.generate_from_posterior:
    print("__LOADING TEST SET__")
    download(test_url,test_file)
    download(label_url,label_file)

    x_test = load_test_img(test_file,is_color)
    y_test = np.load(label_file)

if arg.visualize_latent and x_test is not None:
    print("__VISUALIZING LATENT SPACE WITH TSNE__")
    x_all = tf.convert_to_tensor(x_test)
    mu_all, _ = enc(x_all)

    N = min(mu_all.shape[0], y_test.shape[0])
    mu_sub = mu_all[:N]
    y_sub = y_test[:N]

    z_2d = TSNE(n_components=2).fit_transform(mu_sub.numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_sub, cmap="tab10", s=5)
    plt.colorbar(scatter)
    plt.title("Latent Space (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()

if arg.generate_from_prior:
    print("__GENERATING SAMPLES FROM PRIOR p(z)__")
    z = tf.random.normal([100, latent_dim])
    x_generated = dec(z)
    if not is_color:
        x_generated = tf.reshape(x_generated, [-1, 28, 28])

    u.plot_grid(x_generated.numpy(), name="prior")

if arg.generate_from_posterior and x_test is not None:
    print("__GENERATING RECONSTRUCTIONS FROM POSTERIOR q(z|x)__")
    x_batch_test = tf.convert_to_tensor(x_test[:100])
    x_hat_post, _ , _ = vae(x_batch_test, training = False)
    if not is_color:
        x_hat_post = tf.reshape(x_hat_post, [-1, 28, 28])

    u.plot_grid(x_hat_post.numpy(), name = "posterior") 
