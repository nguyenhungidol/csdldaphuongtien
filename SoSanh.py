import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.feature import hog
from skimage.color import rgb2gray


def extract_color_feature(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return None

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


def extract_hog_feature(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (128, 128))
    features, _ = hog(
        gray_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
    )
    return features

def extract_edge_feature(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
    return edge_hist


def extract_features(image_path):
    color_feat = extract_color_feature(image_path)
    hog_feat = extract_hog_feature(image_path)
    edge_feat = extract_edge_feature(image_path)
    
    if color_feat is None or hog_feat is None or edge_feat is None:
        raise ValueError(f"Không thể trích xuất đặc trưng từ ảnh {image_path}")
    return np.concatenate([color_feat, hog_feat, edge_feat])


def compare_new_image(new_image_path, csv_path="features.csv", top_n=5):
    df = pd.read_csv(csv_path, encoding="utf-8")
    features = df.drop(columns=["filename"]).values
    image_names = df["filename"].values

    new_feat = extract_features(new_image_path).reshape(1, -1)

    distances = euclidean_distances(new_feat, features)[0]
    idxs_euclidean = np.argsort(distances)[:top_n]
    results_euclidean = [(image_names[i], distances[i]) for i in idxs_euclidean]

    similarities = cosine_similarity(new_feat, features)[0]
    idxs_cosine = np.argsort(similarities)[::-1][:top_n]
    results_cosine = [(image_names[i], similarities[i]) for i in idxs_cosine]

    return results_euclidean, results_cosine


def select_image():
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh cần so sánh",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
    )
    return file_path


def show_top_images(results, title, image_folder, selected_name):
    plt.figure(figsize=(12, 4))
    for i, (filename, score) in enumerate(results[:3]):
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Không thể mở ảnh: {image_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"{filename}\nScore: {score:.4f}")
        plt.axis("off")
    plt.suptitle(f"{title}\nSo sánh với: {selected_name}")
    plt.tight_layout()
    plt.show(block=True)


def run_gui():
    root = tk.Tk()
    root.title("Chương trình so sánh ảnh")

    window_width = 400
    window_height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
    root.resizable(False, False)

    tk.Label(root, text="Chọn ảnh để so sánh", font=("Arial", 14)).pack(pady=10)

    selected_image_label = tk.Label(
        root, text="Chưa chọn ảnh nào", fg="blue", font=("Arial", 10)
    )
    selected_image_label.pack(pady=5)

    def on_select():
        new_image_path = select_image()
        if not new_image_path:
            messagebox.showinfo("Thông báo", "Không có ảnh nào được chọn.")
            return

        image_name = os.path.basename(new_image_path)
        selected_image_label.config(text=f"Đã chọn: {image_name}")
        print(f"\n🔍 Ảnh được chọn để so sánh: {image_name}\n")

        image_folder = "Final"
        csv_path = "features.csv"

        try:
            results_euclidean, results_cosine = compare_new_image(
                new_image_path, csv_path
            )

            print("Top 3 ảnh giống nhất (Euclidean):")
            for f, d in results_euclidean[:3]:
                print(f"{f} - Khoảng cách: {d:.4f}")

            print("\nTop 3 ảnh giống nhất (Cosine):")
            for f, s in results_cosine[:3]:
                print(f"{f} - Độ tương đồng: {s:.4f}")

            # show_top_images(
            #     results_euclidean,
            #     "Top 3 - Euclidean Distance",
            #     image_folder,
            #     image_name,
            # )
            show_top_images(
                results_cosine, "Top 3 - Cosine Similarity", image_folder, image_name
            )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")

    tk.Button(root, text="Chọn ảnh", command=on_select, width=20).pack(pady=10)
    tk.Button(root, text="Thoát", command=root.quit, width=20).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
