import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils import to_categorical
from efficientnet.tfkeras import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from dataset import load_soccer_images_from_folder


# hyperParameter of the model (change it as needed)
input_shape = (224, 224, 3)
num_classes = 9
threshold_value = 0.9
epochs = 20
batch_size = 16
learning_rate = 0.0001
image_reshape = (224, 224)

# Create the model
# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of the pre-trained model
base_model.trainable = False

# Build the classification head
ic_model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# Compile the model
ic_model.compile(optimizer=optimizer,
                 loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.9),
                 metrics=['accuracy'])

# Display the model summary
ic_model.summary()

# Train the image classifier model with your data

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
history = ic_model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(x_val, y_val))

print(f"start evaluation")
# Evaluate the model on the test set
prediction = ic_model.predict(x_test)
y_predict = np.argmax(prediction, axis=1)
y_true = np.argmax(y_test, axis=1)

# save the model
ic_model.save("../scripts/models/ic/ic_model.keras")


# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_predict)

# Calculate accuracy, recall, and F1 score
accuracy = accuracy_score(y_true, y_predict)
recall = recall_score(y_true, y_predict, average='weighted')
f1 = f1_score(y_true, y_predict, average='weighted')
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Recall:", recall)
print("F1 Score: {:.2f}".format(f1))

now = datetime.datetime.now()

# Plot training history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("../scripts/models/ic/fig/image_classifier_history.png".format(now))
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(num_classes, num_classes))
commands = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Subtitute"]
commands = np.asarray(commands)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=commands, yticklabels=commands)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("../scripts/models/ic/fig/image_classifier_confusion_matrix.png".format(now))
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_true,
                                                        y_predict,
                                                        target_names=[str(i) for i in range(num_classes)]))