import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
from dataset import load_soccer_images_from_folder


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

# hyperParameter of the model (change it as needed)
epochs = 20
image_reshape = (224, 224)

# Train the VAE model with your data

# path of the dataset folder
train_folder_path = "C:/Users/39392/Desktop/University/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/train"
test_folder_path = "C:/Users/39392/Desktop/University/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/test"

# load image and target
images, labels = load_soccer_images_from_folder(train_folder_path, image_reshape[0], image_reshape[1])
x_test, y_test = load_soccer_images_from_folder(test_folder_path, image_reshape[0], image_reshape[1])

# split the dataset into train, test and validation data
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.4, random_state=42)


history = vae.fit(x_train, y_train,
                  epochs=epochs,
                  validation_data=(x_val, y_val))

# Evaluate the model on the test set
eval_result = vae.evaluate(x_test, y_test)
print("Test Loss:", eval_result)

vae.save("models/vae/vae_model.h5")
now = datetime.datetime.now()

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("models/vae/fig/fine_grain_classifier_history.png".format(now))
plt.show()