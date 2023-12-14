import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_images_from_folder(folder, width, height):
    data = []
    target = []
    for class_label in ["Event", "Soccer"]:
        class_path = os.path.join(folder, class_label)
        if os.path.isdir(class_path):
            for img_filename in os.listdir(class_path):
                img_path = os.path.join(class_path, img_filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (width, height))

                # Aggiungi le immagini e le etichette
                data.append(img)
                target.append(class_label)

    return np.array(data), np.array(target)


# Encoder
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, (3, 3), padding="same")(encoder_inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(32, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Flatten for fully connected layers
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)

# Define the parameters for the latent space
z_mean = layers.Dense(512, name="z_mean")(x)
z_log_var = layers.Dense(512, name="z_log_var")(x)


# Sampling function
def sampling(args):
    z_means, z_log_vars = args
    batch = tf.shape(z_means)[0]
    dim = tf.shape(z_means)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    z_out = z_means + tf.exp(0.5 * z_log_vars) * epsilon

    # KL Divergence regularization term
    zkl_loss = -0.5 * tf.reduce_sum(1 + z_log_vars - tf.square(z_means) - tf.exp(z_log_vars), axis=-1)

    return z_out - zkl_loss


# Sampling layer

z = layers.Lambda(sampling, output_shape=(512,), name="z")([z_mean, z_log_var])

# Decoder
decoder_inputs = layers.Input(shape=(512,))
x = layers.Dense(1024, activation="relu")(decoder_inputs)
x = layers.Reshape((2, 2, 2, 64))(x)

x = layers.UpSampling3D((2, 2, 2))(x)
x = layers.Conv2DTranspose(128, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(64, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(16, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

decoder_outputs = layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")(x)

# Define the VAE model
vae = keras.Model(encoder_inputs, decoder_outputs, name="vae")

# KL Divergence loss
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)

# Reconstruction loss
reconstruction_loss = -tf.reduce_sum(encoder_inputs * tf.math.log(decoder_outputs + 1e-10) + (1 - encoder_inputs) * tf.math.log(1 - decoder_outputs + 1e-10), axis=-1)

# Total loss
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

# Compile the model
vae.compile(optimizer='adam', loss=vae_loss, metrics=['loss'])

# Display the model summary
vae.summary()

# Train the VAE model with your data
# percorso della cartella del dataset
folder_path = "/percorso/della/tua/cartella"
# Carica le immagini e le etichette
images, labels = load_images_from_folder(folder_path, 128, 128)



# Assuming 'images' is a list or array of image data, and 'labels' are corresponding labels
images_train, images_temp, labels_train, labels_temp = train_test_split(images, labels, test_size=0.4, random_state=42)
images_val, images_test, labels_val, labels_test = train_test_split(images_temp, labels_temp, test_size=0.5, random_state=42)

# Now, images_train, labels_train are the training data, images_val, labels_val are the validation data, and images_test, labels_test are the test data


epochs = 10
batch_size = 64
x_train=None
x_val=None
x_test=None

history = vae.fit(x_train, x_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(x_val, x_val))

# Evaluate the model on the test set
eval_result = vae.evaluate(x_test, x_test, batch_size=batch_size)
print("Test Loss:", eval_result)

vae.save("models/vae/vae_model.h5")

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()







