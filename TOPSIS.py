import mysql.connector
from mysql.connector import Error
import pandas as pd

db_host = '127.0.0.1'
db_user = 'root'
db_password = '130225' 
db_name = 'tickets'
db_port = 3306         
connection = None  
try:

    connection = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
        port=db_port
    )

    if connection.is_connected():
        print(f"Kết nối tới database '{db_name}' thành công!")

        # --- 2. Tạo con trỏ và truy vấn bảng 'tickets_combined' ---
        cursor = connection.cursor()
        query = "SELECT * FROM tickets_combined LIMIT 10;"  # Lấy 10 dòng đầu tiên
        
        print(f"\nĐang thực thi truy vấn: {query}")
        cursor.execute(query)

        # --- 3. Lấy và in kết quả ---
        records = cursor.fetchall()
        
        print(f"\nTìm thấy {cursor.rowcount} dòng từ bảng 'tickets_combined':")
        for row in records:
            print(row)

except Error as e:
    # In ra lỗi cụ thể để dễ dàng chẩn đoán
    print(f"\nLỗi khi thực thi: {e}")

finally:
    # --- 4. Đóng kết nối ---
    if connection and connection.is_connected():
        cursor.close()
        connection.close()
        print("\nĐã đóng kết nối MySQL.")
        
# topsis_dashboard.py

# --- 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf # Thư viện để lấy dữ liệu chứng khoán
import plotly.express as px

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="TOPSIS - Phân tích Rủi ro Cổ phiếu",
    page_icon="⚖️",
    layout="wide"
)

# --- 2. CÁC HÀM XỬ LÝ LÕI (TƯƠNG TỰ MODEL DEFINITION) ---

# Sử dụng cache để không phải tải lại dữ liệu mỗi khi thay đổi widget
@st.cache_data
def get_stock_data(tickers_list):
    """
    Hàm lấy các chỉ số tài chính quan trọng cho việc phân tích rủi ro từ Yahoo Finance.
    """
    financial_data = []
    for ticker_symbol in tickers_list:
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Lấy các chỉ số, nếu không có thì trả về None để xử lý sau
            data = {
                'Mã CP': ticker_symbol,
                'Beta': info.get('beta'),
                'P/E': info.get('trailingPE'),
                'Nợ/Vốn CSH': info.get('debtToEquity'),
                'ROE': info.get('returnOnEquity'),
                'Biên LN': info.get('profitMargins')
            }
            financial_data.append(data)
        except Exception as e:
            st.warning(f"Không thể lấy dữ liệu cho mã '{ticker_symbol}'. Lỗi: {e}")
            
    df = pd.DataFrame(financial_data).set_index('Mã CP')
    
    # Xử lý missing values bằng giá trị trung vị của cột
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    return df

def run_topsis(decision_matrix, weights, impacts):
    """
    Hàm thực thi thuật toán TOPSIS để xếp hạng các lựa chọn.
    """
    # Bước 1: Chuẩn hóa ma trận quyết định (Vector Normalization)
    norm_matrix = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))
    
    # Bước 2: Tính toán ma trận quyết định đã được chuẩn hóa và có trọng số
    weighted_matrix = norm_matrix * weights
    
    # Bước 3: Xác định giải pháp lý tưởng tốt nhất (A+) và tệ nhất (A-)
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])
    
    for i in range(len(impacts)):
        if impacts[i] == 1: # Tiêu chí Benefit (càng cao càng tốt)
            ideal_best[i] = weighted_matrix.iloc[:, i].max()
            ideal_worst[i] = weighted_matrix.iloc[:, i].min()
        else: # Tiêu chí Cost (càng thấp càng tốt)
            ideal_best[i] = weighted_matrix.iloc[:, i].min()
            ideal_worst[i] = weighted_matrix.iloc[:, i].max()
            
    # Bước 4: Tính khoảng cách Euclidean đến giải pháp lý tưởng
    dist_to_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    # Bước 5: Tính điểm TOPSIS (Performance Score)
    # Thêm một số rất nhỏ (epsilon) vào mẫu số để tránh chia cho 0
    epsilon = 1e-6
    topsis_score = dist_to_worst / (dist_to_best + dist_to_worst + epsilon)
    
    return topsis_score

# --- 3. XÂY DỰNG GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI) ---

st.title("⚖️ Ứng dụng Phân tích Rủi ro Đầu tư Cổ phiếu bằng TOPSIS")
st.info("Ứng dụng này cho phép bạn xếp hạng rủi ro của các cổ phiếu dựa trên các chỉ số tài chính và trọng số do bạn tùy chỉnh.")

# --- Sidebar: Nơi người dùng nhập liệu và cấu hình ---
with st.sidebar:
    st.header("⚙️ Bảng điều khiển")
    
    # Nhập danh sách cổ phiếu
    tickers_input = st.text_area(
        "Nhập các mã cổ phiếu (cách nhau bởi dấu phẩy hoặc xuống dòng):",
        "AAPL, MSFT, GOOGL, TSLA, NVDA"
    )
    
    # Định nghĩa các tiêu chí và tác động của chúng
    # 1: Benefit (càng cao càng tốt), -1: Cost (càng thấp càng tốt)
    criteria = {
        'Beta': -1, 'P/E': -1, 'Nợ/Vốn CSH': -1,
        'ROE': 1, 'Biên LN': 1
    }
    st.header("⚖️ Trọng số các Tiêu chí")
    st.markdown("*Tổng các trọng số nên bằng 1.0*")
    
    weights = {}
    for crit in criteria:
        weights[crit] = st.slider(f"Độ quan trọng của {crit}", 0.0, 1.0, 0.2, 0.05)
    
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ Tổng trọng số hiện tại: {total_weight:.2f}. Hãy điều chỉnh về 1.0.")
    else:
        st.success(f"✅ Tổng trọng số: {total_weight:.2f}")

# --- Nút bắt đầu phân tích ---
if st.button("🚀 Bắt đầu Phân tích", use_container_width=True):
    # Xử lý input của người dùng
    tickers_list = [t.strip().upper() for t in tickers_input.replace(',', '\n').split('\n') if t.strip()]
    
    if not tickers_list:
        st.error("Vui lòng nhập ít nhất một mã cổ phiếu.")
    else:
        # --- 4. LẤY VÀ HIỂN THỊ DỮ LIỆU (TƯƠNG TỰ DATA LOADING & EXPLORATION) ---
        with st.spinner(f"Đang tải dữ liệu cho {len(tickers_list)} cổ phiếu..."):
            raw_data = get_stock_data(tickers_list)
        
        if not raw_data.empty:
            st.header("📊 Dữ liệu Tài chính thô")
            st.dataframe(raw_data.style.format("{:.4f}"))
            
            # --- 5. THỰC THI TOPSIS VÀ HIỂN THỊ KẾT QUẢ (TƯƠNG TỰ EVALUATION) ---
            st.header("🏆 Kết quả Xếp hạng Rủi ro")
            
            # Chuẩn bị dữ liệu cho hàm TOPSIS
            decision_matrix = raw_data[list(criteria.keys())]
            weights_list = np.array([weights[crit] for crit in criteria])
            impacts_list = np.array([criteria[crit] for crit in criteria])
            
            # Chạy thuật toán
            scores = run_topsis(decision_matrix, weights_list, impacts_list)
            
            # Tạo DataFrame kết quả
            results_df = raw_data.copy()
            results_df['Điểm TOPSIS'] = scores
            results_df['Xếp hạng'] = results_df['Điểm TOPSIS'].rank(ascending=False).astype(int)
            results_df = results_df.sort_values(by='Xếp hạng')
            
            st.success("**Diễn giải:** Cổ phiếu có **Xếp hạng 1** và **Điểm TOPSIS cao nhất** được đánh giá là lựa chọn **ít rủi ro nhất** dựa trên các tiêu chí và trọng số bạn đã chọn.")
            st.dataframe(results_df.style.format("{:.4f}").background_gradient(cmap='Greens', subset=['Điểm TOPSIS']))
            
            # --- 6. TRỰC QUAN HÓA KẾT QUẢ (TƯƠNG TỰ VISUALIZATION) ---
            st.header("🎨 Trực quan hóa Kết quả")
            fig = px.bar(
                results_df,
                x=results_df.index,
                y='Điểm TOPSIS',
                color='Điểm TOPSIS',
                color_continuous_scale='Greens',
                title='So sánh Điểm TOPSIS giữa các Cổ phiếu',
                labels={'Mã CP': 'Mã Cổ phiếu', 'Điểm TOPSIS': 'Điểm TOPSIS (Càng cao càng tốt)'}
            )
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)


    
        
        
        
