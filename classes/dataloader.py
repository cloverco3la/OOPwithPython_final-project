import utils as u
import subprocess as s
import numpy as np
import pickle as p
import os
import tensorflow as tf


class DataLoader():
    def __init__(self, dset, batch_size=128, color_version="m0"):
        assert dset in ("mnist_bw","mnist_color")
        assert color_version in ("m0", "m1", "m2", "m3", "m4")

        self.dset = dset
        self.batch_size = batch_size
        self.color_version = color_version

        if dset == "mnist_bw":
            self.is_color = False
            self.data_file = "data_sets/mnist_bw.npy"
            self.data_url  = u.bw_train_url
        else:
            self.is_color = True
            self.data_file = "data_sets/mnist_color.pkl"
            self.data_url  = u.color_train_url
        
        self.x = None

    def download(self):
            print(f"Downloading {self.data_file} ...")
            try: 
                s.run(["wget", "-O", self.data_file, self.data_url], check=True)
            except s.CalledProcessError:
                print("Could not download the file!")
                raise
    
    def load_data(self):
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"{self.data_file} not found. Call download() first.")
        
        if self.dset == "mnist_bw":
            x_np = np.load(self.data_file)
            x_np = x_np.astype("float32")/255.0
            x_np = x_np.reshape(-1, 28*28)
            self.x = x_np
        else:
            with open(self.data_file,"rb") as f:
                data = p.load(f)
            
            if not isinstance(data, dict):
                raise TypeError("mnist_color.pkl is expected to be a dict with keys 'm0'..'m4'.")

            x_np = data[self.color_version]
            self.x = x_np.astype("float32")

        return self.x
    
    def get_dataset(self, shuffle = True, shuffle_buffer = 50000):
        if self.x is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        ds = tf.data.Dataset.from_tensor_slices(self.x)

        if shuffle:
            ds = ds.shuffle(min(shuffle_buffer, self.x.shape[0]))

        ds = ds.batch(self.batch_size)

        return ds



    