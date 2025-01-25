import tensorflow as tf
from tensorflow.keras import layers

class GAN:
    def __init__(self, latent_dim=100, img_shape=(64, 64, 3)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def build_generator(self):
        model = tf.keras.Sequential()
        
        # Foundation for 8x8 image
        model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_dim=self.latent_dim))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((8, 8, 256)))
        
        # Upsample to 16x16
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        # Upsample to 32x32
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        # Upsample to 64x64
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        # Output layer
        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        
        return model
    
    def build_discriminator(self):
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.img_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        # Downsample to 32x32
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        # Downsample to 16x16
        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        # Flatten and output
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        
        return model
