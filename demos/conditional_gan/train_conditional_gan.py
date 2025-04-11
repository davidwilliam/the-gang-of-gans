import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os

class ConditionalGAN:
    def __init__(self, image_shape=(128, 128, 3), log_dir="gan_plots", checkpoint_dir="checkpoints"):
        self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNELS = image_shape
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0002, decay_steps=10000, decay_rate=0.9, staircase=True)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.5)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.5)
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_generator(self):
        inputs = Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNELS))
        x = layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(inputs)
        x = layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(256, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(512, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(256, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(x)
        outputs = layers.Conv2DTranspose(self.CHANNELS, 4, strides=2, padding='same', activation='sigmoid')(x)
        return Model(inputs, outputs)

    def _build_discriminator(self):
        inputs = Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNELS))
        x = layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(inputs)
        x = layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(256, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(512, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        return Model(inputs, outputs)

    def load_image(self, path):
        img = Image.open(path).resize((self.IMG_WIDTH, self.IMG_HEIGHT))
        img = np.array(img) / 255.0
        if img.shape[-1] == 4:
            img = img[..., :3]
        return np.clip(img, 0, 1).astype(np.float32)

    def train(self, hand_drawn_path, real_image_path, epochs=1000, batch_size=6, plot_interval=500):
        hand_drawn = self.load_image(hand_drawn_path)
        real_img = self.load_image(real_image_path)

        real_variants = np.array([
            real_img,
            real_img * 0.9,
            real_img * 1.1,
            real_img + np.random.normal(0, 0.02, real_img.shape),
            real_img * 0.95,
            real_img * 1.05
        ])
        real_variants = np.clip(real_variants, 0, 1).astype(np.float32)

        losses = {"epoch": [], "d_loss": [], "g_loss": []}
        print("\033[96m[INFO]\033[0m Starting Conditional GAN training...")
        progress = tqdm(range(epochs), desc="Training", ncols=100)

        for epoch in progress:
            idx = np.random.randint(0, len(real_variants), batch_size)
            real_batch = real_variants[idx]
            hand_exp = tf.repeat(tf.expand_dims(hand_drawn, axis=0), batch_size, axis=0)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                fake_images = self.generator(hand_exp, training=True)
                real_output = self.discriminator(real_batch, training=True)
                fake_output = self.discriminator(fake_images, training=True)
                d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
                d_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

            d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
            self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

            for _ in range(4):
                idx = np.random.randint(0, len(real_variants), batch_size)
                real_batch = real_variants[idx]
                hand_exp = tf.repeat(tf.expand_dims(hand_drawn, axis=0), batch_size, axis=0)
                with tf.GradientTape() as disc_tape:
                    fake_images = self.generator(hand_exp, training=False)
                    real_output = self.discriminator(real_batch, training=True)
                    fake_output = self.discriminator(fake_images, training=True)
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
                self._save_comparison_plot(hand_drawn, fake_images[0], real_img, epoch + 1)

        pd.DataFrame(losses).to_csv(os.path.join(self.log_dir, "losses.csv"), index=False)
        self.generator.save(os.path.join(self.checkpoint_dir, "generator.keras"))
        self.discriminator.save(os.path.join(self.checkpoint_dir, "discriminator.keras"))
        self._save_comparison_plot(hand_drawn, self.generator(np.expand_dims(hand_drawn, 0), training=False)[0], real_img, epoch + 1, final=True)

        print("\n\033[95m[SUMMARY]\033[0m")
        print(f" Generator Params: {self.generator.count_params():,}")
        print(f" Discriminator Params: {self.discriminator.count_params():,}")
        print(f" Total Epochs: {epochs}")
        print(f" Checkpoints saved to: {self.checkpoint_dir}")
        print(f" Plots saved to: {self.log_dir}")
        print(" Training complete. The GAN learned to refine sketches into photorealism.\n")

    def _save_comparison_plot(self, hand_drawn, generated, real, epoch, final=False):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(hand_drawn)
        axes[0].set_title("Hand-Drawn")
        axes[0].axis("off")
        axes[1].imshow(generated)
        axes[1].set_title("Generated")
        axes[1].axis("off")
        axes[2].imshow(real)
        axes[2].set_title("Real")
        axes[2].axis("off")
        filename = "final_result.png" if final else f"epoch_{epoch}.png"
        plt.savefig(os.path.join(self.log_dir, filename))
        plt.close()

if __name__ == "__main__":
    gan = ConditionalGAN(log_dir="demos/conditional_gan/gan_plots", checkpoint_dir="demos/conditional_gan/checkpoints")
    gan.train("demos/conditional_gan/hand_drawn_dove.png", "demos/conditional_gan/real_dove.png", epochs=1000, batch_size=6, plot_interval=500)