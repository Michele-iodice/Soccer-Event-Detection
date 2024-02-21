import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
from dataset import load_soccer_images_from_folder
from keras.utils import to_categorical

# hyperParameter of the model (change it as needed)
epochs = 20
image_reshape = (224, 224)
num_classes = 9
learning_rate = 0.0001
clipping_value=2.0
# l2 = 0.01

# Encoder
encoder_inputs = keras.Input(shape=(224, 224, 3), name="encoder_inputs")
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
    zkl_loss = kl_divergence(z_means, z_log_vars)

    return z_out + zkl_loss


def kl_divergence(z_means, z_log_vars):
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_vars - tf.square(z_means) - tf.exp(z_log_vars), axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)
    return kl_loss


# Sampling layer
z = layers.Lambda(sampling, output_shape=(512,), name="z")([z_mean, z_log_var])
# Define encoder model
encoder_model = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
decoder_inputs = layers.Input(shape=(512,), name="decoder_inputs")
x = layers.Dense(14*14*256, activation="relu")(decoder_inputs)
x = layers.Reshape((14, 14, 256))(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

decoder_outputs = layers.Conv2DTranspose(3, (1, 1), strides=(1, 1), activation="sigmoid")(x)

# Define decoder model
decoder_model = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

# Combine encoder and decoder into VAE model
z_mean, z_log_var, z = encoder_model(encoder_inputs)
vae_outputs = decoder_model(z - kl_divergence(z_mean, z_log_var))
vae = keras.Model(encoder_inputs, [vae_outputs, z_mean, z_log_var], name="vae")


# Define custom VAE loss function
def vae_loss(y_true, y_pred):
    epsilon = 1e-10
    beta = 0.7

    y_true = tf.cast(y_true, tf.float32)
    y_true_mean = tf.reduce_mean(y_true)
    vae_output = y_pred[0]
    vae_output = tf.expand_dims(vae_output, axis=0)
    vae_output_mean = tf.reduce_mean(vae_output)
    loss_mean = y_pred[1]
    loss_log_var = y_pred[2]

    # KL Divergence regularization term
    kl_loss = kl_divergence(loss_mean, loss_log_var)

    # Reconstruction loss
    reconstruction_loss = - (y_true_mean * tf.math.log(vae_output_mean + epsilon) + (1 - y_true_mean) * tf.math.log(1 - vae_output_mean + epsilon))
    reconstruction_loss = tf.where(tf.math.is_nan(reconstruction_loss), tf.zeros_like(reconstruction_loss),
                                   reconstruction_loss)

    # Total loss
    total_loss = beta * reconstruction_loss + (1 - beta) * kl_loss
    return total_loss


# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=clipping_value)
vae.compile(optimizer=optimizer,
            loss=vae_loss,
            metrics=[vae_loss])

# Display the model summary
vae.summary()

# Train the VAE model with your data

# path of the dataset folder
train_folder_path = "C:/Users/39392/Desktop/University/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/train"
test_folder_path = "C:/Users/39392/Desktop/University/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/test"

# load image and target
images, labels = load_soccer_images_from_folder(train_folder_path, image_reshape[0], image_reshape[1])
x_test, y_test = load_soccer_images_from_folder(test_folder_path, image_reshape[0], image_reshape[1])

# split the dataset into train, test and validation data
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.4, random_state=42)

# Convert class vector to numeric values
class_to_int = {c: i for i, c in enumerate(set(y_train))}
y_train_numeric = [class_to_int[c] for c in y_train]
class_to_int = {c: i for i, c in enumerate(set(y_test))}
y_test_numeric = [class_to_int[c] for c in y_test]
class_to_int = {c: i for i, c in enumerate(set(y_val))}
y_val_numeric = [class_to_int[c] for c in y_val]

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train_numeric, num_classes)
y_test = to_categorical(y_test_numeric, num_classes)
y_val = to_categorical(y_val_numeric, num_classes)
print(f"start training")
history = vae.fit(x_train, x_train,
                  epochs=epochs,
                  validation_data=(x_val, x_val))


# Evaluate the model on the test set
print(f"start evaluation")
eval_result = vae.evaluate(x_test, x_test)
print("Test Loss:", eval_result)

vae.save("../scripts/models/vae/vae_model.keras")
now = datetime.datetime.now()

# Plot training history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, int(len(loss)) + 1)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("../scripts/models/vae/fig/vae_loss_report".format(now))
plt.show()