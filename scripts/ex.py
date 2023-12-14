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
