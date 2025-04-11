
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def load_generator(checkpoint_dir="demos/vanilla_gan/checkpoints"):
    model_path = os.path.join(checkpoint_dir, "generator.keras")
    return tf.keras.models.load_model(model_path)

def generate_and_plot(generator, noise_dim=2, num_samples=100, output_path="demos/vanilla_gan/gan_plots/generated_points.png"):
    noise = tf.random.normal([num_samples, noise_dim])
    generated_points = generator(noise, training=False).numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(generated_points[:, 0], generated_points[:, 1], c='green', label='Generated Points', alpha=0.7)
    plt.title("Generated Points Using Trained Vanilla GAN")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"\033[92m[INFO]\033[0m Generated plot saved to: {output_path}")

if __name__ == "__main__":
    generator = load_generator()
    generate_and_plot(generator)
