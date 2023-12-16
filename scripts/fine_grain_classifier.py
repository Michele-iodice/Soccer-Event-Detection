import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
from keras import layers
import torch
import datetime
import matplotlib as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, accuracy_score
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import seaborn as sns
from keras.losses import categorical_crossentropy
from dataset import load_card_images_from_folder


def mamc_loss(inputs, targets):
    """
            Args:
                inputs (torch.Tensor): feature matrix with shape (batch_size, part_num, feat_dim).
                targets (torch.LongTensor): ground truth labels with shape (num_classes).
            :return the mamc loss explained in equation 12 of MAMC paper
            Reference: Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition
            """
    b, p, _ = inputs.size()
    n = b * p
    inputs = inputs.contiguous().view(n, -1)
    targets = torch.repeat_interleave(targets, p)
    parts = torch.arange(p).repeat(b)
    prod = torch.mm(inputs, inputs.t())

    same_class_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    same_atten_mask = parts.expand(n, n).eq(parts.expand(n, n).t())

    s_sasc = same_class_mask & same_atten_mask
    s_sadc = (~same_class_mask) & same_atten_mask
    s_dasc = same_class_mask & (~same_atten_mask)
    s_dadc = (~same_class_mask) & (~same_atten_mask)

    # For each anchor, compute equation (11) of paper
    loss_sasc = 0
    loss_sadc = 0
    loss_dasc = 0
    for i in range(n):
        # loss_sasc
        pos = prod[i][s_sasc[i]]
        neg = prod[i][s_sadc[i] | s_dasc[i] | s_dadc[i]]
        n_pos = pos.size(0)
        n_neg = neg.size(0)
        pos = pos.repeat(n_neg, 1).t()
        neg = neg.repeat(n_pos, 1)
        loss_sasc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))

        # loss_sadc
        pos = prod[i][s_sadc[i]]
        neg = prod[i][s_dadc[i]]
        n_pos = pos.size(0)
        n_neg = neg.size(0)
        pos = pos.repeat(n_neg, 1).t()
        neg = neg.repeat(n_pos, 1)
        loss_sadc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))

        # loss_dasc
        pos = prod[i][s_dasc[i]]
        neg = prod[i][s_dadc[i]]
        n_pos = pos.size(0)
        n_neg = neg.size(0)
        pos = pos.repeat(n_neg, 1).t()
        neg = neg.repeat(n_pos, 1)
        loss_dasc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))

    loss_n_pair=(loss_sasc + loss_sadc + loss_dasc) / n
    # Define softmax loss
    loss_softmax = categorical_crossentropy(targets, inputs)
    # Define the weight parameter λ
    lambda_param = 0.5
    # Combine the losses with the specified weight
    loss_combined = loss_softmax + lambda_param * loss_n_pair

    return loss_combined


# hyperParameter of the model (change it as needed)
input_shape=(448, 448, 3)
num_classes=2
epochs = 60
batch_size = 16
learning_rate = 0.001
image_reshape = (448, 448)
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
y = layers.Dense(2048)(y)
z = layers.Dense(2048)(z)
# ReLU layer
y = layers.ReLU()(y)
z = layers.ReLU()(z)
# Fully Connected Layer 2
y = layers.Dense(2048)(y)
z = layers.Dense(2048)(z)
# Sigmoid layer
y = layers.Activation('sigmoid')(y)
z = layers.Activation('sigmoid')(z)

# Attention map
attention1 = layers.Multiply([x, y])
attention2 = layers.Multiply([x, z])

# Flatten
fl1 = layers.Flatten()(attention1)
fl2 = layers.Flatten()(attention1)

# Fully Connected Layer 3
fc1 = layers.Dense(1024)(fl1)
fc2 = layers.Dense(1024)(fl2)


# Output layer with sigmoid activation for binary classification
output1 = layers.Dense(1, activation='softmax')(fc1)
output2 = layers.Dense(1, activation='softmax')(fc2)

# Create the model
fgc_model = tf.keras.Model(inputs=input_layer, outputs=[output1, output2])

# Compile the model
fgc_model.compile(optimizer='adam', loss=['categorical_crossentropy', mamc_loss], metrics=['accuracy'])

# Display the architecture of the model
fgc_model.summary()

# implement reduce_lr (to prevent over fitting)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0000001)

# Train the Fine-Grain classifier model with your data

# path of the dataset folder
train_folder_path = "C:/Users/39392/Desktop/Università/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/train"
test_folder_path = "C:/Users/39392/Desktop/Università/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/test"

# load image and target
images, labels = load_card_images_from_folder(train_folder_path, image_reshape[0], image_reshape[1])
x_test, y_test = load_card_images_from_folder(test_folder_path, image_reshape[0], image_reshape[1])

# split the dataset into train, test and validation data
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.4, random_state=42)

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_val = to_categorical(y_val, num_classes)

history = fgc_model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        validation_steps=64,
                        verbose=1,
                        callbacks=reduce_lr)

# Evaluate the model on the test set
prediction = fgc_model.predict(x_test)
y_predict = np.argmax(prediction, axis=1)
y_true = np.argmax(y_test, axis=1)

fgc_model.save("models/fgc/fgc_model.h5")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_predict)

# Calculate accuracy, recall, and F1 score
accuracy = accuracy_score(y_true, y_predict, average='weighted')
recall = recall_score(y_true, y_predict, average='weighted')
f1 = f1_score(y_true, y_predict, average='weighted')
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Recall:", recall)
print("F1 Score: {:.2f}".format(f1))

# plot results
now = datetime.datetime.now()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_without_MAMC accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("models/fgc/fig/fine_grain_classifier_history.png".format(now))
plt.show()

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_without_MAMC loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("models/fgc/fig/fine_grain_classifier_loss.png".format(now))
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(num_classes, num_classes))
commands = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Substitute"]
commands = np.asarray(commands)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=commands, yticklabels=commands)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("models/ic/fig/fine_grain_classifier_confusion_matrix.png".format(now))
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_true,
                                                        y_predict,
                                                        target_names=[str(i) for i in range(num_classes)]))