import tensorflow as tf
from keras import layers
from efficientnet.tfkeras import EfficientNetB0

def build_efficientnetb0_model(input_shape=(224, 224, 3), num_classes=9):
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

    return model

# Create the model
input_shape = (224, 224, 3)
num_classes = 9

model = build_efficientnetb0_model(input_shape=input_shape, num_classes=num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
