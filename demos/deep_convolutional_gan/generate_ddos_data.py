import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

NOISE_DIM = 50
NUM_POINTS = 100
NUM_FEATURES = 3
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

generator_path = os.path.join(OUTPUT_DIR, "checkpoints/generator.keras")
generator = tf.keras.models.load_model(generator_path)

def generate_synthetic_ddos(num_samples=5):
    noise = tf.random.normal([num_samples, NOISE_DIM])
    fake_data = generator(noise, training=False).numpy()
    return fake_data

def plot_sample(sample_idx, data):
    sample = data[sample_idx, :, 0, :]
    labels = ["Volume", "Packet Size", "Connections"]
    x = np.linspace(0, 10, NUM_POINTS)
    plt.figure(figsize=(10, 6))
    for i in range(NUM_FEATURES):
        plt.plot(x, sample[:, i], label=labels[i])
    plt.title(f"Synthetic DDoS Sample #{sample_idx+1}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"gan_plots/synthetic_ddos_{sample_idx+1}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# Save synthetic data to CSV
def save_to_csv(data):
    labels = ["volume", "packet_size", "connections"]
    for i in range(data.shape[0]):
        sample = data[i, :, 0, :]
        df = pd.DataFrame(sample, columns=labels)
        csv_path = os.path.join(OUTPUT_DIR, f"gan_plots/synthetic_ddos_{i+1}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

if __name__ == "__main__":
    os.makedirs(os.path.join(OUTPUT_DIR, "gan_plots"), exist_ok=True)
    synthetic_data = generate_synthetic_ddos(num_samples=5)
    for idx in range(synthetic_data.shape[0]):
        plot_sample(idx, synthetic_data)
    save_to_csv(synthetic_data)