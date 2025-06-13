import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os
import pickle

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.applications.vgg16 import VGG16
from keras.models import Model

# Constants
SIZE_X, SIZE_Y = 128, 128
TRAIN_PATH = 'stage1_train/'

# Load folder names and split
train_ids = next(os.walk(TRAIN_PATH))[1][:5]
train_ids.sort()
train_ids_train = train_ids[:4]  # 20 for training
train_ids_val = train_ids[1:]   # 5 for validation

def load_data(ids):
    images, masks = [], []
    for id_ in ids:
        img_path = os.path.join(TRAIN_PATH, id_, "images", id_ + ".png")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

        mask = np.zeros((SIZE_Y, SIZE_X), dtype=np.uint8)
        mask_dir = os.path.join(TRAIN_PATH, id_, "masks")
        for mask_file in os.listdir(mask_dir):
            mask_path = os.path.join(mask_dir, mask_file)
            mask_ = cv2.imread(mask_path, 0)
            mask_ = cv2.resize(mask_, (SIZE_Y, SIZE_X))
            mask = np.maximum(mask, mask_)
        masks.append(mask)
    return np.array(images), np.expand_dims(np.array(masks), axis=3)

# Load data
X_train_imgs, y_train_masks = load_data(train_ids_train)
X_val_imgs, y_val_masks = load_data(train_ids_val)

# VGG16 feature extraction
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))
for layer in VGG_model.layers:
    layer.trainable = False
feature_extractor = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)

X_train_feat = feature_extractor.predict(X_train_imgs)
X_val_feat = feature_extractor.predict(X_val_imgs)

# Reshape and filter
def prepare_dataset(features, masks):
    X = features.reshape(-1, features.shape[3])
    Y = masks.reshape(-1)
    df = pd.DataFrame(X)
    df['Label'] = Y
    df = df[df['Label'] != 0]
    return df.drop('Label', axis=1), df['Label']

X_for_RF_train, Y_for_RF_train = prepare_dataset(X_train_feat, y_train_masks)
X_for_RF_val, Y_for_RF_val = prepare_dataset(X_val_feat, y_val_masks)

# Train model
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_for_RF_train, Y_for_RF_train)

# Predict on validation
y_pred = rf_model.predict(X_for_RF_val)

# Report
acc = accuracy_score(Y_for_RF_val, y_pred)
report = classification_report(Y_for_RF_val, y_pred)
print("Validation Accuracy:", acc)
print("Classification Report:\n", report)

# Visualize prediction
prediction_image = y_pred.reshape(SIZE_X, SIZE_Y)
plt.imshow(prediction_image, cmap='gray')
plt.title("Validation Prediction")
plt.axis('off')
plt.show()
