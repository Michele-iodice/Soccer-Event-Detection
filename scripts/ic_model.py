import os
import numpy as np
import cv2
import tensorflow as tf
from keras import layers
from keras.utils import to_categorical
from efficientnet.tfkeras import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def load_images_from_folder(folder, width, height):
    data = []
    target = []
    for class_label in os.listdir(folder):
        if class_label not in ["Red-Cards", "Yellow-Cards"]:
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


# hyperParameter
input_shape = (224, 224, 3)
num_classes = 9
threshold_value = 0.9

# Create the model
# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of the pre-trained model
base_model.trainable = False

# Build the classification head
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


# Apply threshold to the predictions
def thresholded_predictions(predictions, threshold=threshold_value):
    return tf.where(predictions > threshold, 1, 0)


# Wrap the threshold function as a Lambda layer
threshold_layer = layers.Lambda(thresholded_predictions)

# Add the threshold layer to the model
model.add(threshold_layer)


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# hyperParameter
epochs = 10
batch_size = 16
image_reshape = (128, 128)

# Train the VAE model with your data

# percorso della cartella del dataset
folder_path = "C:/Users/39392/Desktop/Università/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/train"
# Carica le immagini e le etichette
images, labels = load_images_from_folder(folder_path, image_reshape[0], image_reshape[1])

# split the dataset into train, test and validation data
x_train, images_temp, y_train, labels_temp = train_test_split(images, labels, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(images_temp, labels_temp, test_size=0.5, random_state=42)

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_val = to_categorical(y_val, num_classes)

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val))

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
prediction = model.predict(x_test)
y_pred = np.argmax(prediction, axis=1)
y_true = np.argmax(y_test, axis=1)


# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Calculate accuracy, recall, and F1 score
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
f1 = f1_score(y_true, y_pred, average='weighted')
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Recall:", recall)
print("F1 Score: {:.2f}".format(f1))

# Plot the confusion matrix
plt.figure(figsize=(num_classes, num_classes))
commands = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Subtitue"]
commands = np.asarray(commands)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=commands, yticklabels=commands)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred,
                                                        target_names=[str(i) for i in range(num_classes)]))

model.save("models/ic/ic_model.h5")