from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from PIL import Imagei
import os
import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Step 1: Get all image paths
image_dir = '/image/path/dir'  # Replace with your image directory
image_paths = []

for dirname, _, filenames in os.walk(image_dir):
    for filename in filenames:
        full_path = os.path.join(dirname, filename)
        image_paths.append(full_path)

# Step 2: Shuffle and split
random.seed(42)
random.shuffle(image_paths)

train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

print(f"Train size: {len(train_paths)}, Test size: {len(test_paths)}")

# Step 3: Function to load and preprocess images
def load_images(paths, target_size=(128, 128)):
    data = []
    for path in paths:
        try:
            img = Image.open(path).convert('RGB')  # or 'L' for grayscale
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # normalize
            data.append(img_array)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
    return np.array(data)

# Step 4: Load images
x_train = load_images(train_paths)
x_test = load_images(test_paths)

# If images are RGB (3 channels)
for i in range(10):
    plt.subplot(2, 5, i + 1)  # 2 rows, 5 columns
    plt.imshow(x_train[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 5: Build the autoencoder model

# Input layer
input_img = Input(shape=(128, 128, 3))  # RGB image

# Encoder
x = Conv2D(32, 3, activation='relu', padding='same')(input_img)  # 128x128x32
x = MaxPooling2D(2)(x)                                            # 64x64x32
x = Conv2D(64, 3, activation='relu', padding='same')(x)           # 64x64x64
x = MaxPooling2D(2)(x)                                            # 32x32x64
x = Conv2D(32, 3, activation='relu', padding='same')(x)           # 32x32x32
x = MaxPooling2D(2)(x)                                            # 16x16x32
latent = Conv2D(4, 3, activation='relu', padding='same')(x)       # 16x16x4

# Create encoder model
encoder = Model(inputs=input_img, outputs=latent, name='encoder')

# Latent input for decoder
latent_input = Input(shape=(16, 16, 4))

x = Conv2D(32, 3, activation='relu', padding='same')(latent_input)
x = UpSampling2D(2)(x)                      # 32x32
x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = UpSampling2D(2)(x)                      # 64x64
x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = UpSampling2D(2)(x)                      # 128x128
output_img = Conv2D(3, 3, activation='sigmoid', padding='same')(x)  # RGB output

# Create decoder model
decoder = Model(inputs=latent_input, outputs=output_img, name='decoder')

# Combine encoder and decoder into autoencoder
autoencoder = Model(inputs=input_img, outputs=decoder(encoder(input_img)))
autoencoder.compile(optimizer='adam', loss='mse')

# Step 6: Train the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test))

# Step 7: Evaluate the autoencoder
for i in range(10):
    img = x_test[i]
    sample_img_batch = np.expand_dims(img, axis=0)
    latent_vec = encoder.predict(sample_img_batch)
    reconstructed = decoder.predict(latent_vec)

    # Show
    plt.figure(figsize=(4,2))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(reconstructed[0])
    plt.title('Reconstructed')
    plt.axis('off')
    plt.show()


# Step 8: Save the models
encoder.save('encoder_model.h5')
decoder.save('decoder_model.h5')
autoencoder.save('autoencoder_model.h5')

# Step 9: Model summary (Optional)
autoencoder.summary()

