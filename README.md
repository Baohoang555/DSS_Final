# 1. Ứng dụng Phân tích Rủi ro Cổ phiếu bằng TOPSIS

Đây là một ứng dụng web được xây dựng bằng Streamlit, cho phép người dùng phân tích, đánh giá rủi ro và xếp hạng các cổ phiếu dựa trên phương pháp Ra quyết định Đa tiêu chí (MCDM) - TOPSIS.

Lưu ý: Đây là một công cụ tham khảo và phân tích, không phải là lời khuyên đầu tư trực tiếp.

(Ghi chú: Bạn hãy thay thế link ảnh trên bằng ảnh chụp màn hình của chính ứng dụng của bạn, ví dụ như các file ảnh bạn đã gửi lên.)
<img width="1535" height="549" alt="image" src="https://github.com/user-attachments/assets/8a1125a3-8054-4607-9028-886c5dc45a3e" />



# 2. Thư viện sử dụng


_ streamlit 

_ pandas 

_ numpy 

_ yfinance 

_ plotly 

_ mysql-connector-python 

_ openpyxl 


# 3. Cách Cài đặt và Chạy Code

Bạn cần có Python 3.8+ và pip được cài đặt trên máy.

# Bước 1: Clone Repository

git clone [https://github.com/Baohoang555/DSS_Final.git](https://github.com/Baohoang555/DSS_Final.git)
cd DSS_Final

# Bước 2: Tạo môi trường ảo (Khuyến nghị)

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate



# Bước 3: Cài đặt các thư viện

Sử dụng file requirements.txt trong thư mục này:

pip install -r requirements.txt


# Bước 4: Cấu hình Database SQL
Import file dữ liệu tickets_all_combined vào SQL và kết nối SQL vào python

# Bước 5: Chạy ứng dụng Streamlit
# Terminal
streamlit run DSS_Final.py


