import os
import kagglehub
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

class CelebADataLoader(Sequence):
    def __init__(self, batch_size=32, img_size=(64, 64)):
        self.batch_size = batch_size
        self.img_size = img_size
        self.dataset_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
        self.image_files = self._get_image_files()
        
    def _get_image_files(self):
        img_dir = os.path.join(self.dataset_path, "img_align_celeba")
        return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_files = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        
        for file in batch_files:
            img = Image.open(file)
            img = img.resize(self.img_size)
            img = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
            batch_images.append(img)
            
        return np.array(batch_images)
    
    def load_full_dataset(self):
        return np.array([self.__getitem__(i) for i in range(len(self))])
