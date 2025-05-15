import cv2
import os
import numpy as np
import json
from tqdm import tqdm
from skimage.feature import hog
import mysql.connector


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


def extract_edge_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
    return edge_hist


def extract_combined_features(image):
    color_feat = extract_color_feature(image)
    hog_feat = extract_hog_feature(image)
    edge_feat = extract_edge_feature(image)
    return np.concatenate([color_feat, hog_feat, edge_feat])


def insert_into_mysql(filename, feature_vector):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="hung123321@",
            database="dapt",
        )
        cursor = conn.cursor()
        vector_str = json.dumps(feature_vector.tolist())
        sql = "INSERT INTO image_features (filename, features) VALUES (%s, %s)"
        cursor.execute(sql, (filename, vector_str))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Lỗi lưu MySQL:", e)


def extract_features_from_folder(folder_path):
    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)
            if image is None:
                continue
            image = cv2.resize(image, (128, 128))
            features = extract_combined_features(image)
            insert_into_mysql(filename, features)


if __name__ == "__main__":
    folder_path = "Final"
    extract_features_from_folder(folder_path)
    print("✅ Đã lưu đặc trưng vào MySQL.")
