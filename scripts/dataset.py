import os
import cv2
import numpy as np
import tensorflow as tf


def load_soccer_images_from_folder(folder, width, height):
    data = []
    target = []
    print(f"dataset loading...")
    for class_label in os.listdir(folder):
        if class_label not in ["Red-Cards", "Yellow-Cards"]:
            class_path = os.path.join(folder, class_label)
            if os.path.isdir(class_path):
                for img_filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, (width, height))

                            # Add image and target
                            data.append(img)
                            target.append(class_label)
                    except cv2.error as e:
                        print(f"Error reading image {img_path}: {e}")

    print(f"dataset loading complete")
    return np.array(data), np.array(target)


def load_card_images_from_folder(folder, width, height):
    data = []
    target = []
    print(f"dataset loading...")
    for class_label in os.listdir(folder):
        if class_label in ["Red-Cards", "Yellow-Cards"]:
            class_path = os.path.join(folder, class_label)
            if os.path.isdir(class_path):
                for img_filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, (width, height))

                            # Add image and target
                            data.append(img)
                            target.append(class_label)
                    except cv2.error as e:
                        print(f"Error reading image {img_path}: {e}")

    print(f"dataset loading complete")
    return np.array(data), np.array(target)


def load_images_from_folder(folder, width, height):
    data = []
    target = []
    print(f"dataset loading...")
    for class_label in os.listdir(folder):
        if class_label in ["Event", "Other", "Soccer"]:
            class_path = os.path.join(folder, class_label)
            if os.path.isdir(class_path):
                for img_filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (width, height))
                        image_normalized = img / 255.0  # Normalizzazione nel range [0, 1]
                        # Aggiungi una dimensione batch
                        image_batch = tf.expand_dims(image_normalized, axis=0)
                        image = tf.cast(image_batch, tf.float32)

                        # Add image and target
                        data.append(image)
                        target.append(class_label)

    print(f"dataset loading complete")
    return np.array(data), np.array(target)

