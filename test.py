from train_vae import load_test_img
x_np = load_test_img("data_sets/mnist_color_te.pkl", is_color=True)
print(x_np.shape)
print(x_np.dtype)