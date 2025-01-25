import tensorflow as tf
from src.data.loader import CelebADataLoader
from src.models.gan import GAN
import matplotlib.pyplot as plt
import numpy as np

def generate_images(model, num_images=1, latent_dim=100):
    random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))
    generated_images = model.generator(random_latent_vectors)
    generated_images = (generated_images * 127.5 + 127.5).numpy().astype('uint8')
    
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.show()

def main():
    # Initialize data loader
    data_loader = CelebADataLoader(batch_size=32, img_size=(64, 64))
    
    # Initialize GAN
    gan = GAN(latent_dim=100, img_shape=(64, 64, 3))
    
    # Compile models
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    # Generate sample images before training
    print("Generating sample images before training...")
    generate_images(gan, num_images=5)
    
    print("GAN setup complete. Ready for training.")

if __name__ == "__main__":
    main()
