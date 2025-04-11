import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

np.random.seed(42)
tf.random.set_seed(42)

NUM_POINTS = 100
NUM_FEATURES = 3
NOISE_DIM = 50
EPOCHS = 3000
BATCH_SIZE = 4
PLOT_INTERVAL = 1000
OUTPUT_DIR = "demos/deep_convolutional_gan"

os.makedirs(f"{OUTPUT_DIR}/gan_plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

x_values = np.linspace(0, 10, NUM_POINTS)

def simulate_ddos():
    spike_windows = np.random.choice(NUM_POINTS, 5, replace=False)
    spike_times = np.zeros(NUM_POINTS)
    for window in spike_windows:
        start = max(0, window - 2)
        end = min(NUM_POINTS, window + 3)
        spike_times[start:end] += np.random.poisson(0.5, end - start)
    spike_magnitudes = (np.random.pareto(2, NUM_POINTS) * 0.3) * (spike_times > 0)
    baseline = 0.4 + 0.1 * (x_values / 10)
    volume = baseline + spike_magnitudes + 0.05 * np.random.normal(0, 1, NUM_POINTS)
    packet_size = 0.3 + 0.5 * spike_magnitudes + 0.03 * np.random.normal(0, 1, NUM_POINTS)
    connections = 0.2 + 0.8 * spike_magnitudes + 0.04 * np.random.normal(0, 1, NUM_POINTS)
    return np.stack([volume, packet_size, connections], axis=-1)

def normalize_and_augment(data):
    data = (data - data.min()) / (data.max() - data.min())
    aug = [data * f for f in [0.9, 1.0, 1.1, 0.95, 1.05]]
    aug.append(data + np.random.normal(0, 0.05, data.shape))
    return np.clip(np.stack(aug), 0, 1).astype(np.float32)

raw_ddos = simulate_ddos()
ddos_data = normalize_and_augment(raw_ddos)
ddos_data = ddos_data.reshape(-1, NUM_POINTS, 1, NUM_FEATURES)

def build_generator():
    inputs = Input(shape=(NOISE_DIM,))
    x = layers.Dense(25 * 128)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((25, 1, 128))(x)
    x = layers.Conv2DTranspose(64, (5, 1), strides=(2, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(32, (5, 1), strides=(2, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2DTranspose(NUM_FEATURES, (5, 1), padding="same", activation="sigmoid")(x)
    return Model(inputs, outputs)

def build_discriminator():
    inputs = Input(shape=(NUM_POINTS, 1, NUM_FEATURES))
    x = layers.Conv2D(64, (5, 1), strides=(2, 1), padding="same")(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, (5, 1), strides=(2, 1), padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)

@tf.function
def train_step(real_data, batch_size, generator, discriminator, g_optimizer, d_optimizer, loss_fn):
    noise = tf.random.normal([batch_size, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_data = generator(noise, training=True)
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)
        d_loss_real = loss_fn(tf.ones_like(real_output) * 0.9, real_output)
        d_loss_fake = loss_fn(tf.zeros_like(fake_output) + 0.1, fake_output)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)
    d_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    g_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    return d_loss, g_loss

def train():
    generator = build_generator()
    discriminator = build_discriminator()
    g_optimizer = tf.keras.optimizers.Adam(1e-4)
    d_optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    losses = []
    for epoch in tqdm(range(EPOCHS)):
        idx = np.random.randint(0, len(ddos_data), BATCH_SIZE)
        real_batch = ddos_data[idx]
        d_loss, g_loss = train_step(real_batch, BATCH_SIZE, generator, discriminator, g_optimizer, d_optimizer, loss_fn)
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: D Loss = {d_loss:.4f}, G Loss = {g_loss:.4f}")
            losses.append((epoch + 1, float(d_loss), float(g_loss)))

    generator.save(f"{OUTPUT_DIR}/checkpoints/generator.keras")
    pd.DataFrame(losses, columns=["epoch", "d_loss", "g_loss"]).to_csv(f"{OUTPUT_DIR}/gan_plots/losses.csv", index=False)

    noise = tf.random.normal([1, NOISE_DIM])
    fake = generator(noise, training=False)[0, :, 0, :].numpy()
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    labels = ["Volume", "Packet Size", "Connections"]
    for i in range(3):
        axs[i].plot(x_values, fake[:, i], label="Generated", color="blue")
        axs[i].plot(x_values, raw_ddos[:, i], label="Real", color="green")
        axs[i].set_title(labels[i])
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gan_plots/final_result.png")
    plt.close()

if __name__ == "__main__":
    train()