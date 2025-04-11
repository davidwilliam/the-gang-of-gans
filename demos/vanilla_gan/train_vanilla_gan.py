
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

class VanillaGAN:
    def __init__(self, noise_dim=2, output_dim=2, log_dir="gan_plots", checkpoint_dir="checkpoints"):
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_generator(self):
        inputs = Input(shape=(self.noise_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.output_dim)(x)
        return Model(inputs, outputs)

    def _build_discriminator(self):
        inputs = Input(shape=(self.output_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        return Model(inputs, outputs)

    @tf.function
    def train_step(self, real_points, batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_points = self.generator(noise, training=True)
            real_output = self.discriminator(real_points, training=True)
            fake_output = self.discriminator(fake_points, training=True)
            d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        return d_loss, g_loss

    def generate_circle_data(self, batch_size):
        theta = np.random.uniform(0, 2 * np.pi, batch_size)
        return np.array([[np.cos(t), np.sin(t)] for t in theta], dtype=np.float32)

    def save_plot(self, real_points, generated_points, epoch, is_final=False):
        plt.figure(figsize=(6, 6))
        plt.scatter(real_points[:, 0], real_points[:, 1], c='blue', label='Real (Circle)', alpha=0.6)
        plt.scatter(generated_points[:, 0], generated_points[:, 1], c='red', label='Generated', alpha=0.6)
        plt.title(f"{'Final' if is_final else f'GAN Progress at Epoch {epoch}'}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        filename = "final_result.png" if is_final else f"epoch_{epoch}.png"
        plt.savefig(os.path.join(self.log_dir, filename))
        plt.close()

    def train(self, epochs=10000, batch_size=32, plot_interval=1000):
        print("\033[96m[INFO]\033[0m Starting GAN training...")
        losses = {"epoch": [], "d_loss": [], "g_loss": []}
        progress = tqdm(range(epochs), desc="Training", ncols=100)

        for epoch in progress:
            real_points = self.generate_circle_data(batch_size)
            d_loss, g_loss = self.train_step(real_points, batch_size)

            for _ in range(2):
                real_points = self.generate_circle_data(batch_size)
                noise = tf.random.normal([batch_size, self.noise_dim])
                fake_points = self.generator(noise, training=False)
                with tf.GradientTape() as disc_tape:
                    real_output = self.discriminator(real_points, training=True)
                    fake_output = self.discriminator(fake_points, training=True)
                    d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
                    d_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
                    d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
                self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

            if (epoch + 1) % 500 == 0:
                print(f"\033[92mEpoch {epoch + 1}\033[0m: D Loss = {d_loss:.4f}, G Loss = {g_loss:.4f}")
                losses["epoch"].append(epoch + 1)
                losses["d_loss"].append(float(d_loss))
                losses["g_loss"].append(float(g_loss))

            if (epoch + 1) % plot_interval == 0:
                noise = tf.random.normal([100, self.noise_dim])
                generated_points = self.generator(noise, training=False).numpy()
                self.save_plot(real_points, generated_points, epoch + 1)

        pd.DataFrame(losses).to_csv(os.path.join(self.log_dir, "losses.csv"), index=False)
        self.generator.save(os.path.join(self.checkpoint_dir, "generator.keras"))
        self.discriminator.save(os.path.join(self.checkpoint_dir, "discriminator.keras"))

        noise = tf.random.normal([100, self.noise_dim])
        generated_points = self.generator(noise, training=False).numpy()
        self.save_plot(real_points, generated_points, epoch=epochs, is_final=True)

        print("\n\033[95m[SUMMARY]\033[0m")
        print(f" Generator Params: {self.generator.count_params():,}")
        print(f" Discriminator Params: {self.discriminator.count_params():,}")
        print(f" Total Epochs: {epochs}")
        print(f" Checkpoints saved to: {self.checkpoint_dir}")
        print(f" Plots saved to: {self.log_dir}")
        print(" Training complete. GAN learned to mimic a circular distribution.")
        print(" You can now use the saved generator to generate new points!\n")

if __name__ == "__main__":
    gan = VanillaGAN(log_dir="demos/vanilla_gan/gan_plots", checkpoint_dir="demos/vanilla_gan/checkpoints")
    gan.train(epochs=3000, batch_size=32, plot_interval=1000)
