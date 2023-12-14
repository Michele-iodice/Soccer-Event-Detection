import os
import cv2
import numpy as np


def load_images_from_folder(folder, width, height):
    images = []
    labels = []
    for class_label in os.listdir(folder):
        class_path = os.path.join(folder, class_label)
        if os.path.isdir(class_path):
            for img_filename in os.listdir(class_path):
                img_path = os.path.join(class_path, img_filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (width, height))

                # Aggiungi le immagini e le etichette
                images.append(img)
                labels.append(class_label)

    return np.array(images), np.array(labels)


# Specifica il percorso della cartella contenente le immagini
folder_path = "/percorso/della/tua/cartella"

# Carica le immagini e le etichette
images, labels = load_images_from_folder(folder_path, 224, 224)

# Ora 'images' è un array di immagini e 'labels' è un array di etichette corrispondenti

# Predict on the test set
num_classes = 2
classes = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Subtitue"]

y_pred = vae.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(num_classes, num_classes))
commands = np.asarray(classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=commands, yticklabels=commands)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


###################################################################à


import tensorflow as tf
from tensorflow.keras import layers, models
from efficientnet.tfkeras import EfficientNetB0

def create_model(input_shape, num_classes):
    # Load pre-trained EfficientNetB0 model without the top layer (include_top=False)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Build the custom head for classification
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def load_and_preprocess_data(train_dir, validation_dir, target_size, batch_size):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator

def train_and_evaluate_model(train_dir, validation_dir, input_shape, num_classes, epochs=10, batch_size=32):
    model = create_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_generator, validation_generator = load_and_preprocess_data(train_dir, validation_dir, input_shape[:2], batch_size)

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Save the trained model
    model.save('efficientnet_model.h5')

    # Evaluate the model
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}')

# Example usage
train_dir = 'path/to/training_data'
validation_dir = 'path/to/validation_data'
input_shape = (224, 224, 3)
num_classes = 9

train_and_evaluate_model(train_dir, validation_dir, input_shape, num_classes)
