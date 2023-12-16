import numpy as np
from sklearn.metrics import confusion_matrix, precision_score
from keras.models import load_model
from dataset import load_images_from_folder
import matplotlib as plt
import seaborn as sns
import datetime

# hyperParameter
image_reshape = (224, 224)
threshold_value = 0.9

# classes of the corresponding model
vae_classes = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Substitute"]
ic_classes = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Substitute"]
fgc_classes = ["Red-Cards", "Yellow-Cards"]

# Load the entire model
vae = load_model("models/vae/vae_model.h5")
ic_model = load_model("models/ic/ic_model.h5")
fgc_model = load_model("models/fgc/fgc_model.h5")

test_folder_path = "C:/Users/39392/Desktop/Università/MAGISTRALE/Information retrieval/project_ir/soccer_dataset/test"

# load image and target as x_test and y_test respectively
x_test, y_test = load_images_from_folder(test_folder_path, image_reshape[0], image_reshape[1])

y_predict = []
y_true = []

for x, y in zip(x_test, y_test):

    class_predict = "Other-images"
    # Evaluate the model on the test set
    predictions = vae.predict(x)
    vae_prediction = np.max(predictions)

    if vae_prediction >= threshold_value:
        ic_predictions = ic_model.predict(x)
        if np.max(ic_predictions) > threshold_value:
            ic_prediction = np.argmax(ic_predictions)
            class_predict = ic_classes[ic_prediction]
            if class_predict == "Cards":
                fgc_predictions = fgc_model.predict(x)
                fgc_prediction1 = np.max(fgc_predictions[0])
                fgc_prediction2 = np.max(fgc_predictions[1])
                if fgc_prediction1 > fgc_prediction2:
                    class_predict = fgc_classes[np.argmax(fgc_predictions[0])]
                else:
                    class_predict = fgc_classes[np.argmax(fgc_predictions[1])]
            if class_predict in ["Center", "Left", "Right"]:
                class_predict = "Other-soccer-events"
        else:
            class_predict = "Other soccer events"

    y_predict.append(class_predict)
    y_true.append(y)


# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_predict)

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_predict, average='weighted')
print("Precision:", precision)

now = datetime.datetime.now()
# Plot precision
plt.title('Precision of Soccer Event Detection Algorithm')
plt.bar(['Precision'], [precision])
plt.ylabel('Precision')
plt.title('Precision (Sklearn)')
plt.savefig("fig/soccer_event_detection_algorithm_precision.png".format(now))
plt.show()

# Plot the confusion matrix
commands = ["Center", "Corner", "Free-Kick", "Penalty", "Red-Cards", "Tackle",
            "To-Substitute", "Yellow-Cards", "Other-soccer-events", "Other-images"]
commands = np.asarray(commands)
plt.figure(figsize=(len(commands), len(commands)))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=commands, yticklabels=commands)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("fig/soccer_event_detection_algorithm_confusion_matrix.png".format(now))
plt.show()