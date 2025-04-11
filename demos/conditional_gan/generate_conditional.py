
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def load_image(path, size=(128, 128)):
    img = Image.open(path).resize(size)
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    return np.clip(img, 0, 1).astype(np.float32)

def generate_image(generator_path, input_path, output_path="generated_result.png"):
    generator = tf.keras.models.load_model(generator_path)
    input_img = load_image(input_path)
    input_tensor = tf.expand_dims(input_img, axis=0)

    generated_img = generator(input_tensor, training=False)[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(input_img)
    axes[0].set_title("Input Sketch")
    axes[0].axis("off")
    axes[1].imshow(generated_img)
    axes[1].set_title("Generated Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Output saved to {output_path}")

if __name__ == "__main__":
    generator_model = "demos/conditional_gan/checkpoints/generator.keras"
    sketch_image = "demos/conditional_gan/hand_drawn_dove.png"
    output_file = "demos/conditional_gan/gan_plots/generated_from_saved_model.png"
    generate_image(generator_model, sketch_image, output_file)
