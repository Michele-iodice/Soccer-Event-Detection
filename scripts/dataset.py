import os
import cv2
import numpy as np


def load_soccer_images_from_folder(folder, width, height):
    data = []
    target = []
    for class_label in os.listdir(folder):
        if class_label not in ["Red-Cards", "Yellow-Cards"]:
            class_path = os.path.join(folder, class_label)
            if os.path.isdir(class_path):
                for img_filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_filename)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (width, height))

                    # Add image and target
                    data.append(img)
                    target.append(class_label)

    return np.array(data), np.array(target)


def load_card_images_from_folder(folder, width, height):
    data = []
    target = []
    for class_label in os.listdir(folder):
        if class_label in ["Red-Cards", "Yellow-Cards"]:
            class_path = os.path.join(folder, class_label)
            if os.path.isdir(class_path):
                for img_filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_filename)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (width, height))

                    # Add image and target
                    data.append(img)
                    target.append(class_label)

    return np.array(data), np.array(target)


def load_images_from_folder(folder, width, height):
    data = []
    target = []
    for class_label in os.listdir(folder):
        if class_label in ["Event", "Other", "Soccer"]:
            class_path = os.path.join(folder, class_label)
            if os.path.isdir(class_path):
                for img_filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_filename)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (width, height))

                    # Add image and target
                    data.append(img)
                    target.append(class_label)

    return np.array(data), np.array(target)

