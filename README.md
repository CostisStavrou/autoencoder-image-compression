# Autoencoder Image Compression

A fully connected autoencoder in PyTorch that compresses 25×50 greyscale images (1,250 pixels) into a 32-number code and reconstructs them, storing 39× fewer values per image.

**Built:** March 2025, as Year 2 Machine Learning coursework for my Computer Science degree at the University of Bath.

## Model

| | Layers |
|---|---|
| Encoder | 1250 → 512 → 128 → 32 (ReLU) |
| Decoder | 32 → 128 → 512 → 1250 (ReLU, sigmoid output) |

It is trained with mean squared error and Adam (learning rate 0.001), batch size 50, for 50 epochs, on a 70/30 train/test split of 1,200 images.

## Results

| Method (32 numbers per image) | Test MSE | PSNR |
|---|---|---|
| Autoencoder (this model) | 0.0106 | 19.8 dB |
| PCA, 32 components | 0.0050 | 23.0 dB |

![Original and reconstructed test images](reconstructed_images.png)

The autoencoder keeps the overall layout and brightness of each image but loses edges and fine detail. **PCA beats it** at the same compression. With 840 training images, a fully connected network with about 1.4 million parameters doesn't learn a better code than a linear projection. Convolutional layers, which share weights across the image and exploit local structure, would be the next thing to try.

## Run it

The dataset (`dataset_LR.npy`, provided with the coursework) is not included in this repository. Place it next to the notebook, then:

```bash
pip install -r requirements.txt
jupyter notebook autoencoder.ipynb
```

The notebook's saved outputs show the results without running it.
