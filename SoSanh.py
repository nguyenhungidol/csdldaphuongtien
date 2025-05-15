import cv2
import os
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from skimage.feature import hog
import mysql.connector
from PIL import Image, ImageTk


def load_features_from_mysql():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="hung123321@",
            database="dapt",
        )
        cursor = conn.cursor()
        cursor.execute("SELECT filename, features FROM image_features")
        rows = cursor.fetchall()

        filenames, features = [], []
        for filename, feat_str in rows:
            vector = np.array(json.loads(feat_str))
            filenames.append(filename)
            features.append(vector)

        cursor.close()
        conn.close()

        return np.array(features), np.array(filenames)
    except Exception as e:
        print("❌ Lỗi đọc MySQL:", e)
        return None, None


def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    h, s, v = cv2.split(hsv)

    h_hist = cv2.calcHist([h], [0], None, [64], [0, 180])
    s_hist = cv2.calcHist([s], [0], None, [64], [0, 256])
    v_hist = cv2.calcHist([v], [0], None, [64], [0, 256])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    color_feat = np.concatenate([h_hist, s_hist, v_hist])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    edges = cv2.Canny(gray, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()

    return np.concatenate([color_feat, hog_feat, edge_hist])


def compare_new_image(new_image_path, top_n=5):
    features, filenames = load_features_from_mysql()
    if features is None:
        raise Exception("Không thể tải đặc trưng từ CSDL.")

    new_feat = extract_features(new_image_path).reshape(1, -1)

    distances = euclidean_distances(new_feat, features)[0]
    idxs_euclidean = np.argsort(distances)[:top_n]
    results_euclidean = [(filenames[i], distances[i]) for i in idxs_euclidean]

    similarities = cosine_similarity(new_feat, features)[0]
    idxs_cosine = np.argsort(similarities)[::-1][:top_n]
    results_cosine = [(filenames[i], similarities[i]) for i in idxs_cosine]

    return results_euclidean, results_cosine


def cv2_to_ImageTk(cv_img, size=(200, 200)):
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize(size)
    return ImageTk.PhotoImage(img_pil)


def run_gui():
    root = tk.Tk()
    root.title("So sánh ảnh")

    frame_top = tk.Frame(root)
    frame_top.grid(row=0, column=0, pady=10)

    label_selected_name = tk.Label(frame_top, text="Chưa chọn ảnh", font=("Arial", 14))
    label_selected_name.pack()

    label_selected_img = tk.Label(frame_top)
    label_selected_img.pack()

    frame_results = tk.Frame(root)
    frame_results.grid(row=1, column=0, pady=10)

    # Mỗi mục kết quả có ảnh + tên + điểm Euclidean + điểm Cosine
    result_widgets = []
    for i in range(3):
        frame = tk.Frame(frame_results, relief=tk.RIDGE, bd=2)
        frame.grid(row=0, column=i, padx=10, pady=5)

        img_label = tk.Label(frame)
        img_label.pack()

        name_label = tk.Label(frame, text="", font=("Arial", 10, "bold"))
        name_label.pack()

        eu_label = tk.Label(frame, text="", font=("Arial", 9))
        eu_label.pack()

        cos_label = tk.Label(frame, text="", font=("Arial", 9))
        cos_label.pack()

        result_widgets.append((img_label, name_label, eu_label, cos_label))

    status_label = tk.Label(root, text="", fg="green", font=("Arial", 10))
    status_label.grid(row=2, column=0, pady=5)

    def select_image():
        return filedialog.askopenfilename(
            title="Chọn ảnh cần so sánh",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
        )

    def on_select():
        new_image_path = select_image()
        if not new_image_path:
            status_label.config(text="Không có ảnh được chọn.")
            return

        image_name = os.path.basename(new_image_path)
        label_selected_name.config(text=f"Ảnh được chọn: {image_name}")

        img_cv = cv2.imread(new_image_path)
        if img_cv is not None:
            img_tk = cv2_to_ImageTk(img_cv)
            label_selected_img.config(image=img_tk)
            label_selected_img.image = img_tk

        try:
            results_euclidean, results_cosine = compare_new_image(new_image_path)
            # In Top 3 kết quả Euclidean
            print("\n📌 Top 3 ảnh giống nhất theo khoảng cách Euclidean:")
            for i, (filename, distance) in enumerate(results_euclidean[:3], 1):
                print(f"{i}. {filename} - Khoảng cách: {distance:.4f}")

            # In Top 3 kết quả Cosine
            print("\n📌 Top 3 ảnh giống nhất theo độ tương đồng Cosine:")
            for i, (filename, similarity) in enumerate(results_cosine[:3], 1):
                print(f"{i}. {filename} - Độ tương đồng: {similarity:.4f}")

            status_label.config(text="So sánh thành công!")

            image_folder = "Final"

            for i, ((fn_eu, score_eu), (fn_co, score_co)) in enumerate(
                zip(results_euclidean[:3], results_cosine[:3])
            ):
                path = os.path.join(image_folder, fn_eu)
                img_res = cv2.imread(path)
                if img_res is not None:
                    img_res_tk = cv2_to_ImageTk(img_res)
                    img_label, name_label, eu_label, cos_label = result_widgets[i]
                    img_label.config(image=img_res_tk)
                    img_label.image = img_res_tk
                    name_label.config(text=fn_eu)
                    eu_label.config(text=f"Euclidean: {score_eu:.4f}")
                    cos_label.config(text=f"Cosine: {score_co:.4f}")
                else:
                    img_label, name_label, eu_label, cos_label = result_widgets[i]
                    img_label.config(image="")
                    name_label.config(text=f"{fn_eu} (Không tìm thấy ảnh)")
                    eu_label.config(text=f"Euclidean: {score_eu:.4f}")
                    cos_label.config(text=f"Cosine: {score_co:.4f}")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")
            status_label.config(text="So sánh thất bại!")

    btn_select = tk.Button(root, text="Chọn ảnh", command=on_select, width=20)
    btn_select.grid(row=3, column=0, pady=10)

    root.geometry("750x700")
    root.mainloop()


if __name__ == "__main__":
    run_gui()
