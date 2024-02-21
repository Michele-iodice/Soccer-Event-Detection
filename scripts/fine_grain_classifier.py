import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
from keras import layers
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, accuracy_score
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import seaborn as sns
from keras.losses import categorical_crossentropy
from dataset import load_card_images_from_folder


def mamc_loss(inputs, target):
    """Args:
                :param inputs :(tf.Tensor) feature matrix with shape (batch_size, part_num, feat_dim).
                :param target :(tf.Tensor) ground truth labels with shape (batch_size,).
                :return the mamc loss explained in equation 12 of MAMC paper Reference: Multi-Attention Multi-Class
                Constraint for Fine-grained Image Recognition
                """
    cc_inputs = inputs
    cc_target = target
    # Ottieni le dimensioni dei tensori
    batch, feature_dim = tf.shape(inputs)[0], tf.shape(inputs)[1]

    # Modifica la forma degli input per facilitare i calcoli
    inputs = tf.reshape(inputs, [batch, 1, feature_dim])
    inputs_transpose = tf.transpose(inputs, perm=[0, 2, 1])

    # Calcola il prodotto tra gli input e la loro trasposta
    dot_product = tf.matmul(inputs, inputs_transpose)
    dot_product = tf.tile(dot_product, [1, 1, 2])

    # Crea le maschere
    same_class_mask = tf.equal(tf.expand_dims(target, 0), tf.expand_dims(target, 1))
    same_atten_mask = tf.eye(batch, dtype=tf.bool)
    same_atten_mask = tf.expand_dims(same_atten_mask, axis=-1)
    same_atten_mask = tf.tile(same_atten_mask, [1, 1, 2])

    # Esegui l'operazione logica AND
    s_sasc = tf.logical_and(same_class_mask, same_atten_mask)
    s_sadc = tf.logical_and(tf.logical_not(same_class_mask), same_atten_mask)
    s_dasc = tf.logical_and(same_class_mask, tf.logical_not(same_atten_mask))
    s_dadc = tf.logical_and(tf.logical_not(same_class_mask), tf.logical_not(same_atten_mask))

    def calculate_loss(tens, mask1, mask2):
        if tf.equal(tf.size(tens), tf.size(mask1)) and tf.equal(tf.size(tens), tf.size(mask2)):
            positive_loss = -tf.math.log_sigmoid(tf.boolean_mask(tens, mask1))
            negative_loss = -tf.math.log_sigmoid(-tf.boolean_mask(tens, mask2))
            loss = tf.reduce_mean(positive_loss + negative_loss)
        else:
            loss = 0.0

        return loss

    # Calcola le perdite
    loss_sasc = calculate_loss(dot_product, s_sasc, tf.logical_or(s_sadc, tf.logical_or(s_dasc, s_dadc)))
    loss_sadc = calculate_loss(dot_product, s_sadc, s_dadc)
    loss_dasc = calculate_loss(dot_product, s_dasc, s_dadc)

    # Calcola la perdita complessiva
    loss_n_pair = tf.reduce_mean(loss_sasc + loss_sadc + loss_dasc)

    # Definisci la perdita softmax
    softmax_loss = tf.reduce_mean(categorical_crossentropy(cc_inputs, cc_target))

    # Definisci il parametro di peso λ=0.5
    lambda_weight = 0.5

    # Combina le perdite pesate
    combined_loss = loss_n_pair * lambda_weight + softmax_loss

    return combined_loss


# hyperParameter of the model (change it as needed)
input_shape=(224, 224, 3)
num_classes=2
epochs = 60
batch_size = 16
learning_rate = 0.001
image_reshape = (224, 224)
embedding_size = 64

# Define the EfficientNetB0 backbone with pretrained weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the convolutional layers of the backbone
base_model.trainable = False

# Define the input layer
input_layer = tf.keras.Input(shape=input_shape)

# Connect the input to the base model
x = base_model(input_layer)

# Global Average Pooling
y = layers.GlobalAveragePooling2D()(x)
z = layers.GlobalAveragePooling2D()(x)
# Fully Connected Layer 1
y = layers.Dense(1280)(y)
z = layers.Dense(1280)(z)
# ReLU layer
y = layers.ReLU()(y)
z = layers.ReLU()(z)
# Fully Connected Layer 2
y = layers.Dense(1280)(y)
z = layers.Dense(1280)(z)
# Sigmoid layer
y = layers.Activation('sigmoid')(y)
z = layers.Activation('sigmoid')(z)

# Attention map
attention1 = layers.Multiply()([x, y])
attention2 = layers.Multiply()([x, z])

# Flatten
fl1 = layers.Flatten()(attention1)
fl2 = layers.Flatten()(attention2)

# Fully Connected Layer 3
fc1 = layers.Dense(1024)(fl1)
fc2 = layers.Dense(1024)(fl2)


# Output layer with sigmoid activation for binary classification
output1 = layers.Dense(num_classes, activation='softmax')(fc1)
output2 = layers.Dense(num_classes, activation='softmax')(fc2)

# Create the model
fgc_model = tf.keras.Model(inputs=input_layer, outputs=[output1, output2])

# Compile the model
fgc_model.compile(optimizer='adam', loss=['categorical_crossentropy', mamc_loss], metrics=['accuracy'])

# Display the architecture of the model
fgc_model.summary()

# implement reduce_lr (to prevent over fitting)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=learning_rate)

# Train the Fine-Grain classifier model with your data

# path of the dataset folder
train_folder_path = "C:/Users/39392/Desktop/University/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/train"
test_folder_path = "C:/Users/39392/Desktop/University/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/test"

# load image and target
images, labels = load_card_images_from_folder(train_folder_path, image_reshape[0], image_reshape[1])
x_test, y_test = load_card_images_from_folder(test_folder_path, image_reshape[0], image_reshape[1])

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
history = fgc_model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=1,
                        callbacks=reduce_lr)

# Evaluate the model on the test set
print(f"start evaluation")
prediction = fgc_model.predict(x_test)
predictions = (prediction[0] + prediction[1]) / 2
y_predict = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

fgc_model.save("../scripts/models/fgc/fgc_model.keras")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_predict)

# Calculate accuracy, recall, and F1 score
accuracy = accuracy_score(y_true, y_predict)
recall = recall_score(y_true, y_predict, average='weighted')
f1 = f1_score(y_true, y_predict, average='weighted')
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Recall:", recall)
print("F1 Score: {:.2f}".format(f1))

# plot results
now = datetime.datetime.now()
acc = history.history['dense_6_accuracy']
val_acc = history.history['val_dense_6_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('model_with_MAMC accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../scripts/models/fgc/fig/fine_grain_classifier_history.png".format(now))
plt.show()

# loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('model_with_MAMC loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../scripts/models/fgc/fig/fine_grain_classifier_loss.png".format(now))
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(num_classes, num_classes))
commands = ["Red-Cards", "Yellow-Cards"]
commands = np.asarray(commands)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=commands, yticklabels=commands)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("../scripts/models/ic/fig/fine_grain_classifier_confusion_matrix.png".format(now))
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_true,
                                                        y_predict,
                                                        target_names=[str(i) for i in range(num_classes)]))