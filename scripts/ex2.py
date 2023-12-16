import tensorflow as tf
from tensorflow.keras import layers, models
from efficientnet.tfkeras import EfficientNetB0

def fine_grain_classification_model(input_shape, num_classes):
    # Base EfficientNetB0 model (pre-trained on ImageNet)
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')

    # Freeze the weights of the base model
    base_model.trainable = False

    # Global average pooling layer
    global_pooling = layers.GlobalAveragePooling2D()(base_model.output)

    # Fully connected layer
    fc1 = layers.Dense(512, activation='relu')(global_pooling)

    # Another fully connected layer
    fc2 = layers.Dense(256, activation='relu')(fc1)

    # Output layer with Sigmoid activation for binary classification
    output1 = layers.Dense(1, activation='sigmoid', name='output1')(fc2)

    # Attention block
    attention = layers.Attention()([base_model.output, fc2])

    # Fully connected layer after attention
    fc_attention = layers.Dense(256, activation='relu')(attention)

    # Output layer with Sigmoid activation for binary classification
    output2 = layers.Dense(1, activation='sigmoid', name='output2')(fc_attention)

    # Create the model
    model = models.Model(inputs=base_model.input, outputs=[output1, output2])

    return model

# Define input shape and number of classes
input_shape = (224, 224, 3)
num_classes = 1  # Since using Sigmoid activation for binary classification

# Create the model
model = fine_grain_classification_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
