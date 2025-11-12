import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==============================
# CONFIGURATION
# ==============================

# Path to your dataset with 12 folders
BASE_DIR = "D:/Projects/waste-segregation-system/data/raw/garbage_classification"  # change this path according to your dataset location
IMG_SIZE = 128            # all images will be resized to 128x128

# Map original 12 classes into 3 main categories
CATEGORY_MAP = {
    "battery": "non_recyclable",
    "clothes": "non_recyclable",
    "shoes": "non_recyclable",
    "trash": "non_recyclable",
    "biological": "biodegradable",
    "cardboard": "recyclable",
    "paper": "recyclable",
    "plastic": "recyclable",
    "metal": "recyclable",
    "white-glass": "recyclable",
    "green-glass": "recyclable",
    "brown-glass": "recyclable"
}

FINAL_CATEGORIES = ["recyclable", "biodegradable", "non_recyclable"]

# ==============================
# LOAD ALL IMAGES
# ==============================

def load_images(base_dir, category_map, img_size=128):
    X, y = [], []
    skipped = 0
    
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Find target class
        if folder_name not in category_map:
            print(f"⚠️ Skipping folder not in map: {folder_name}")
            continue
        
        target_label = category_map[folder_name]
        label_idx = FINAL_CATEGORIES.index(target_label)
        
        print(f"📂 Loading {folder_name} → {target_label}")
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label_idx)
            except Exception as e:
                skipped += 1
                continue

    print(f"✅ Done loading. Skipped {skipped} images.")
    return np.array(X), np.array(y)


# ==============================
# EXECUTE LOADING
# ==============================

X, y = load_images(BASE_DIR, CATEGORY_MAP, IMG_SIZE)

print(f"Total images loaded: {len(X)}")
print(f"Shape of X: {X.shape}, y: {y.shape}")

# ==============================
# INSPECT SAMPLE IMAGE
# ==============================

plt.figure(figsize=(3,3))
plt.imshow(X[0])
plt.title(f"Sample - {FINAL_CATEGORIES[y[0]]}")
plt.axis('off')
plt.show()

# ==============================
# NORMALIZE AND SPLIT
# ==============================

# Normalize pixel values
X = X / 255.0

# Split dataset (80% train, 20% test)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain set: {train_X.shape}, Labels: {train_y.shape}")
print(f"Test set:  {test_X.shape}, Labels: {test_y.shape}")

# ==============================
# SAVE ARRAYS FOR LATER USE
# ==============================

os.makedirs("../data", exist_ok=True)
np.save("../data/train_X.npy", train_X)
np.save("../data/train_y.npy", train_y)
np.save("../data/test_X.npy", test_X)
np.save("../data/test_y.npy", test_y)

print("\n💾 Preprocessed data saved successfully in /data/")
