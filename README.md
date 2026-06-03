# Hệ Thống Chẩn Đoán Bệnh Cây Trồng Bằng Học Sâu (Ensemble Learning) Trên Hạ Tầng Dữ Liệu Lớn (Apache Spark & HDFS)
## Kiến Trúc Hệ Thống 
<img width="617" height="316" alt="image" src="https://github.com/user-attachments/assets/8cc6079b-5368-47b6-b8b1-24d8240daf9b" />


## Cấu Trúc Thư Mục Dự Án

```text
IE212_project/
├── App_Streamlit/
│   └── app.py                  # Mã nguồn giao diện Web chẩn đoán Streamlit
├── notebooks/
│   └── IE212_project.ipynb     # Jupyter Notebook chạy toàn bộ Pipeline Spark & Training
├── data/
│   └── raw/                    # Thư mục chứa dữ liệu hình ảnh thô cục bộ 
├── hadoop_data/                # Vùng lưu trữ vật lý của Hadoop DataNodes 
├── docker-compose.yml          # Cấu hình khởi tạo cụm Docker Container (HDFS & Spark)
├── hadoop.env                  # File cấu hình môi trường cho hệ sinh thái Hadoop
├── requirements.txt            # Danh sách ghim cứng phiên bản các thư viện hệ thống
└── .gitignore                  # Bộ lọc tự động chặn các tệp tin rác và file dữ liệu nặng
```

## Hướng dẫn cấu hình và chạy streamlit
### 1. Chuẩn bị môi trường
Clone code từ git về máy:
```bash
git clone https://github.com/KuTin2005/IE212_project.git
cd IE212_project
```
### 2. Thiết lập môi trường
```bash
# Tạo môi trường ảo (venv)
python -m venv .venv
# Kích hoạt môi trường (Windows only)
.venv\Scripts\activate
# Cài đặt thư viện để chạy dự án
pip install streamlit keras torch torchvision numpy pandas pillow
```
### 3. Cấu hình và khởi động Docker HDFS
Tải docker desktop về máy tính, chạy các lệnh sau ở project đã git clone về:
```bash
docker-compose up -d
```
### 4. Khởi động web streamlit
Mở terminal và copy câu lệnh sau:
```bash
streamlit run App_Streamlit/app.py
```
