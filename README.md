# Face Generation using GANs

This project implements a Generative Adversarial Network (GAN) for generating realistic face images using the CelebA dataset. The implementation uses TensorFlow/Keras and includes data preprocessing, model architecture, and image generation capabilities.

## Features

- Data loading and preprocessing pipeline for CelebA dataset
- Customizable GAN architecture with generator and discriminator networks
- Image generation function for visualizing results
- Modular code structure for easy extension
- Pre-training visualization of generated images

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- NumPy
- Matplotlib
- Pillow
- KaggleHub

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/GAN-for-Image-Generation.git
cd GAN-for-Image-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the project:
```bash
python main.py
```

## Project Structure

```
GAN-for-Image-Generation/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── gan.py
├── main.py
├── requirements.txt
└── README.md
```

## Usage

The project is set up to:
1. Load and preprocess the CelebA dataset
2. Initialize the GAN model
3. Generate sample images before training
4. Prepare for training (training not started by default)

To start training, modify the main.py script to include the training loop.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License
