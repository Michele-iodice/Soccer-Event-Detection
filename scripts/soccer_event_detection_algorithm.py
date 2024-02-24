import numpy as np
from dataset import load_images_from_folder
import pandas as pd
import tensorflow as tf


def loss_vae(y_true, y_pred):
    epsilon = 1e-10
    beta = 0.7

    vae_output = y_pred[0]
    vae_output = tf.expand_dims(vae_output, axis=0)
    loss_mean = y_pred[1]
    loss_log_var = y_pred[2]

    # KL Divergence regularization term
    kl_loss = -0.5 * tf.reduce_sum(1 + loss_log_var - tf.square(loss_mean) - tf.exp(loss_log_var), axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)

    # Reconstruction loss
    reconstruction_loss = - (y_true * tf.math.log(vae_output + epsilon) + (1 - y_true) * tf.math.log(1 - vae_output + epsilon))
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    # Total loss
    total_loss = beta * reconstruction_loss + (1 - beta) * kl_loss
    return total_loss


# hyperParameter
image_reshape = (224, 224)
threshold_ic = 0.1
threshold_vae = 144.5

# classes of the corresponding model
vae_classes = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Subtitute"]
ic_classes = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Subtitute"]
fgc_classes = ["Red-Cards", "Yellow-Cards"]

# Load the entire model
vae = tf.saved_model.load("models/vae/vae_model")
ic_model = tf.saved_model.load("models/ic/ic_model2")
fgc_model = tf.saved_model.load("../scripts/models/fgc/fgc_model2")

test_folder_path = "C:/Users/39392/Desktop/University/MAGISTRALE/Information retrieval/project_ir/soccer_dataset"

# load image and target as x_test and y_test respectively
x_test, _ = load_images_from_folder(test_folder_path, image_reshape[0], image_reshape[1])

y_predict = []
precision = []
print("Start evaluation...")
for x in x_test:
    class_predict = "Other-images"
    # Evaluate the model on the test set
    vae_reconstruction = vae(x)
    vae_predictions = loss_vae(x, vae_reconstruction)
    x_precision = vae_predictions / (1 + vae_predictions)
    if vae_predictions < threshold_vae:
        ic_predictions = ic_model(x)
        x_precision = np.max(ic_predictions)
        if x_precision > threshold_ic:
            ic_prediction = np.argmax(ic_predictions)
            class_predict = ic_classes[ic_prediction]
            if class_predict == "Cards":
                fgc_predictions = fgc_model(x)
                fgc_prediction1 = np.max(fgc_predictions[0])
                fgc_prediction2 = np.max(fgc_predictions[1])
                if fgc_prediction1 > fgc_prediction2:
                    x_precision = fgc_prediction1
                    class_predict = fgc_classes[np.argmax(fgc_predictions[0])]
                else:
                    x_precision = fgc_prediction2
                    class_predict = fgc_classes[np.argmax(fgc_predictions[1])]
            if class_predict in ["Center", "Left", "Right"]:
                class_predict = "Other-soccer-events"
        else:
            class_predict = "Other-soccer-events"
    else:
        class_predict = "Other-images"

    y_predict.append(class_predict)
    precision.append(x_precision)


print("evaluation terminated")
keys = y_predict
values = precision
columns = ['Class', 'Precision']

df = pd.DataFrame({'Class': keys, 'Precision': values}, columns=columns)
grouped_df = df.groupby('Class').mean().reset_index()

grouped_df.to_csv('../scripts/result/algorithm_precision_model3.csv', index=False)

print(grouped_df)