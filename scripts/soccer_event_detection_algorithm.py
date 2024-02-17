import numpy as np
from keras.models import load_model
from dataset import load_images_from_folder
import pandas as pd

# hyperParameter
image_reshape = (224, 224)
threshold_value = 0.9

# classes of the corresponding model
vae_classes = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Subtitute"]
ic_classes = ["Cards", "Center", "Corner", "Free-Kick", "Left", "Penalty", "Right", "Tackle", "To-Subtitute"]
fgc_classes = ["Red-Cards", "Yellow-Cards"]

# Load the entire model
vae = load_model("../scripts/models/vae/vae_model.keras")
ic_model = load_model("../scripts/models/ic/ic_model.keras")
fgc_model = load_model("../scripts/models/fgc/fgc_model.keras")

test_folder_path = "C:/Users/39392/Desktop/University/MAGISTRALE/Information retrieval/project_ir/soccer_dataset"

# load image and target as x_test and y_test respectively
x_test, y_test = load_images_from_folder(test_folder_path, image_reshape[0], image_reshape[1])

y_predict = []
precision = []

for x, y in zip(x_test, y_test):

    class_predict = "Other-images"
    # Evaluate the model on the test set
    predictions = vae.predict(x)
    vae_prediction = np.max(predictions)
    x_precision = vae_prediction
    if vae_prediction >= threshold_value:
        ic_predictions = ic_model.predict(x)
        x_precision = np.max(ic_predictions)
        if np.max(ic_predictions) > threshold_value:
            ic_prediction = np.argmax(ic_predictions)
            class_predict = ic_classes[ic_prediction]
            if class_predict == "Cards":
                fgc_predictions = fgc_model.predict(x)
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

    y_predict.append(class_predict)
    precision.append(x_precision)


keys = y_predict
values = precision
columns = ['Class', 'Precision']

df = pd.DataFrame({'Key': keys, 'Value': values}, columns=columns)
grouped_df = df.groupby('Key').mean().reset_index()

df.to_csv('../scripts/fig/algorithm_precision.csv', index=False)

print(grouped_df)