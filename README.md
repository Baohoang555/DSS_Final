
⚖️ Ứng dụng Phân tích Rủi ro Cổ phiếu bằng TOPSIS

Đây là một ứng dụng web được xây dựng bằng Streamlit, cho phép người dùng phân tích, đánh giá rủi ro và xếp hạng các cổ phiếu dựa trên phương pháp Ra quyết định Đa tiêu chí (MCDM) - TOPSIS.

Lưu ý: Đây là một công cụ tham khảo và phân tích, không phải là lời khuyên đầu tư trực tiếp.

(Ghi chú: Bạn hãy thay thế link ảnh trên bằng ảnh chụp màn hình của chính ứng dụng của bạn, ví dụ như các file ảnh bạn đã gửi lên.)

🚀 Tính năng chính (Modules)

Phân tích TOPSIS: Tự động tính toán điểm số và xếp hạng các cổ phiếu từ tốt nhất (ít rủi ro nhất) đến tệ nhất (rủi ro nhất) dựa trên thuật toán TOPSIS.

Hai nguồn dữ liệu:

Yahoo Finance API: Lấy dữ liệu tài chính gần như thời gian thực (Beta, P/E, ROE, v.v.)

Database MySQL: Kết nối tới cơ sở dữ liệu tùy chỉnh của người dùng (RSI, MACD, v.v.)

Tùy chỉnh Trọng số: Giao diện thanh trượt (slider) cho phép người dùng tùy chỉnh trọng số (mức độ quan trọng) cho từng chỉ số.

Bộ trọng số mẫu: Cung cấp các bộ trọng số cài đặt sẵn (An toàn, Cân bằng, Tăng trưởng) để phân tích nhanh.

Tự động Phân tích & Gợi ý: Tự động tạo ra các đoạn văn bản phân tích điểm mạnh của cổ phiếu tốt nhất, điểm yếu của cổ phiếu rủi ro nhất và đưa ra các gợi ý hành động chung.

Trực quan hóa Dữ liệu:

Biểu đồ Cột: So sánh trực quan điểm TOPSIS giữa các cổ phiếu.

Biểu đồ Radar: So sánh chi tiết các chỉ số (đã chuẩn hóa) của top 5 cổ phiếu hàng đầu.

Xuất Báo cáo: Cho phép tải xuống báo cáo phân tích chi tiết (bao gồm dữ liệu đầu vào, trọng số đã chọn và kết quả xếp hạng) dưới dạng file Excel (.xlsx).

📦 Thư viện sử dụng

Dự án này được xây dựng bằng Python 3 và yêu cầu các thư viện sau:

streamlit - Framework chính để xây dựng web app.

pandas - Xử lý và phân tích dữ liệu.

numpy - Tính toán số học (cần cho thuật toán TOPSIS).

yfinance - Lấy dữ liệu từ API của Yahoo Finance.

plotly (plotly.express, plotly.graph_objects) - Vẽ các biểu đồ tương tác.

mysql-connector-python - Kết nối với cơ sở dữ liệu MySQL.

openpyxl - Cần thiết để pandas ghi file Excel (.xlsx).

Tất cả các thư viện này được liệt kê trong file requirements.txt.

🛠️ Cách Cài đặt và Chạy Code

Bạn cần có Python 3.8+ và pip được cài đặt trên máy.

Bước 1: Clone Repository

git clone [https://github.com/TEN-CUA-BAN/TEN-DU-AN.git](https://github.com/TEN-CUA-BAN/TEN-DU-AN.git)
cd TEN-DU-AN


(Hãy thay thế TEN-CUA-BAN/TEN-DU-AN bằng đường dẫn GitHub thực tế của bạn)

Bước 2: Tạo môi trường ảo (Khuyến nghị)

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate


Bước 3: Cài đặt các thư viện

Sử dụng file requirements.txt trong thư mục này:

pip install -r requirements.txt


Bước 4: Cấu hình Database (Tùy chọn)

Ứng dụng có thể chạy mà không cần database (chỉ dùng Yahoo Finance API). Tuy nhiên, nếu bạn muốn sử dụng luồng "Database MySQL":

Đảm bảo bạn có một server MySQL đang chạy.

Import dữ liệu của bạn vào (ví dụ: database tên tickets, table tên tickets_combined).

Quan trọng: Mở file code (ví dụ: app.py) và cập nhật thông tin kết nối trong hàm get_db_connection():

def get_db_connection():
    """Tạo kết nối tới MySQL database"""
    try:
        connection = mysql.connector.connect(
            host='DIA_CHI_HOST_CUA_BAN',     # Thay '127.0.0.1'
            user='USER_CUA_BAN',        # Thay 'root'
            password='MAT_KHAU_CUA_BAN',  # Thay '130225'
            database='TEN_DATABASE',    # Thay 'tickets'
            port=3306                   # Thay đổi nếu cần
        )
        return connection
    except Error as e:
        st.error(f"❌ Lỗi kết nối database: {e}")
        return None


Bước 5: Chạy ứng dụng Streamlit

Quay lại terminal (đã kích hoạt môi trường ảo venv), chạy lệnh sau:

streamlit run app.py


(Giả sử file Python của bạn tên là app.py)

Streamlit sẽ tự động mở một tab trên trình duyệt của bạn hiển thị ứng dụng.
