1. Ứng dụng Phân tích Rủi ro Cổ phiếu bằng TOPSIS

Đây là một ứng dụng web được xây dựng bằng Streamlit, cho phép người dùng phân tích, đánh giá rủi ro và xếp hạng các cổ phiếu dựa trên phương pháp Ra quyết định Đa tiêu chí (MCDM) - TOPSIS.

Lưu ý: Đây là một công cụ tham khảo và phân tích, không phải là lời khuyên đầu tư trực tiếp.

(Ghi chú: Bạn hãy thay thế link ảnh trên bằng ảnh chụp màn hình của chính ứng dụng của bạn, ví dụ như các file ảnh bạn đã gửi lên.)

2. Tính năng chính (Modules)

Phân tích TOPSIS: Tự động tính toán điểm số và xếp hạng các cổ phiếu từ tốt nhất (ít rủi ro nhất) đến tệ nhất (rủi ro nhất) dựa trên thuật toán TOPSIS.

Hai nguồn dữ liệu:

Yahoo Finance API: Lấy dữ liệu tài chính gần như thời gian thực (Beta, P/E, ROE, v.v.)

Database MySQL: Kết nối tới cơ sở dữ liệu tùy chỉnh của người dùng (RSI, MACD, v.v.)


Trực quan hóa Dữ liệu:

Biểu đồ Cột: So sánh trực quan điểm TOPSIS giữa các cổ phiếu.

Biểu đồ Radar: So sánh chi tiết các chỉ số (đã chuẩn hóa) của top 5 cổ phiếu hàng đầu.

Xuất Báo cáo: Cho phép tải xuống báo cáo phân tích chi tiết (bao gồm dữ liệu đầu vào, trọng số đã chọn và kết quả xếp hạng) dưới dạng file Excel (.xlsx).

3. Thư viện sử dụng

Dự án này được xây dựng bằng Python 3 và yêu cầu các thư viện sau:

_ streamlit - Framework chính để xây dựng web app.

_ pandas - Xử lý và phân tích dữ liệu.

_ numpy - Tính toán số học (cần cho thuật toán TOPSIS).

_ yfinance - Lấy dữ liệu từ API của Yahoo Finance.

_ plotly (plotly.express, plotly.graph_objects) - Vẽ các biểu đồ tương tác.

_ mysql-connector-python - Kết nối với cơ sở dữ liệu MySQL.

_ openpyxl - Cần thiết để pandas ghi file Excel (.xlsx).

Tất cả các thư viện này được liệt kê trong file requirements.txt.

4. Cách Cài đặt và Chạy Code

Bạn cần có Python 3.8+ và pip được cài đặt trên máy.

Bước 1: Clone Repository

git clone [https://github.com/Baohoang555/DSS_Final.git](https://github.com/Baohoang555/DSS_Final.git)
cd DSS_Final

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


Bước 4: Cấu hình Database SQL


Bước 5: Chạy ứng dụng Streamlit

Quay lại terminal (đã kích hoạt môi trường ảo venv), chạy lệnh sau:

streamlit run DSS_Final.py

Streamlit sẽ tự động mở một tab trên trình duyệt của bạn hiển thị ứng dụng.
