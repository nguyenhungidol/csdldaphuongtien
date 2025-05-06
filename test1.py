import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog
from skimage.color import rgb2gray


# --- Trích xuất đặc trưng màu HSV ---
def extract_color_feature(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    h, s, v = cv2.split(hsv)

    h_hist = cv2.calcHist([h], [0], None, [64], [0, 180])
    s_hist = cv2.calcHist([s], [0], None, [64], [0, 256])
    v_hist = cv2.calcHist([v], [0], None, [64], [0, 256])

    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()

    return np.concatenate([h_hist, s_hist, v_hist])


# --- Trích xuất đặc trưng HOG ---
def extract_hog_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return hog_feat


# Biên Canny
def extract_edge_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
    return edge_hist


# --- Kết hợp tất cả đặc trưng ---
def extract_combined_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return None

    # Resize ảnh về cùng kích thước nếu cần
    image = cv2.resize(image, (128, 128))

    color_feat = extract_color_feature(image)
    hog_feat = extract_hog_feature(image)
    edge_feat = extract_edge_feature(image)

    combined_feat = np.concatenate([color_feat, hog_feat, edge_feat])
    return combined_feat


# --- Quét folder để trích xuất đặc trưng ---
def extract_features_from_folder(folder_path):
    features = []
    image_names = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, filename)
            feature = extract_combined_features(path)
            if feature is not None:
                features.append(feature)
                image_names.append(filename)

    return features, image_names


# --- Main ---
if __name__ == "__main__":
    folder_path = "Final"  # Thư mục chứa ảnh
    features, image_names = extract_features_from_folder(folder_path)

    df = pd.DataFrame(features)
    df["filename"] = image_names

    df.to_csv("features.csv", index=False, encoding="utf-8")
    print("✅ Đã trích xuất đặc trưng (color + HOG + edge) và lưu vào 'features.csv'")
