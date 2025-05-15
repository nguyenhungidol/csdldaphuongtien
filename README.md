MySQL - Lưu thông tin về bộ trích rút đặc trưng
create database dapt;
use dapt;

CREATE TABLE image_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255),
    features LONGTEXT  -- lưu dạng chuỗi JSON hoặc numpy tolist()
);
