
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_losses(csv_path='demos/vanilla_gan/gan_plots/losses.csv', save_path='demos/vanilla_gan/gan_plots/loss_plot.png'):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Loss CSV not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['d_loss'], label='Discriminator Loss', color='blue')
    plt.plot(df['epoch'], df['g_loss'], label='Generator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Loss plot saved to: {save_path}")

if __name__ == '__main__':
    plot_losses()
