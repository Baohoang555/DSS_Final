# 1. Ứng dụng Phân tích Rủi ro Cổ phiếu bằng TOPSIS

Đây là một ứng dụng web được xây dựng bằng Streamlit, cho phép người dùng phân tích, đánh giá rủi ro và xếp hạng các cổ phiếu dựa trên phương pháp Ra quyết định Đa tiêu chí (MCDM) - TOPSIS và 

Lưu ý: Đây là một công cụ tham khảo và phân tích, không phải là lời khuyên đầu tư trực tiếp.

<img width="1221" height="522" alt="image" src="https://github.com/user-attachments/assets/114414db-4c44-4413-a56f-a0cc3c068ef2" />



# 2. Thư viện sử dụng


_ streamlit 

_ pandas 

_ numpy 

_ yfinance 

_ plotly 

_ mysql-connector-python 

_ openpyxl 

_ itertools


# 3. Cách Cài đặt và Chạy Code

Bạn cần có Python 3.8+ và pip được cài đặt trên máy.

# Bước 1: Clone Repository

git clone [https://github.com/Baohoang555/DSS_Final.git](https://github.com/Baohoang555/DSS_Final.git)
cd ahp.py

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
streamlit run ahp.py


