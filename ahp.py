import mysql.connector
from mysql.connector import Error
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import itertools # Thư viện để tạo các cặp so sánh cho AHP
import scipy
from scipy.stats import spearmanr

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="TOPSIS & AHP - Phân tích Rủi ro Cổ phiếu",
    page_icon="⚖️",
    layout="wide"
)

# --- CÁC BIẾN CỐ ĐỊNH CHO LUỒNG DATABASE ---
DB_CRITERIA = ['RSI', 'MACD', 'trailingPE_snapshot', 'marketCap_snapshot', 'Returns']
DB_IMPACTS = {
    'RSI': 'Cost',
    'MACD': 'Benefit',
    'trailingPE_snapshot': 'Cost',
    'marketCap_snapshot': 'Benefit',
    'Returns': 'Benefit'
}

# === DICTIONARY GIẢI THÍCH CÁC CHỈ SỐ ===
CRITERIA_EXPLANATIONS = {
    # Yahoo Finance
    'Beta': "Đo lường mức độ biến động của cổ phiếu so với thị trường chung. Beta < 1 cho thấy ít biến động hơn thị trường (ít rủi ro hơn).",
    'P/E': "Tỷ lệ Giá trên Thu nhập. Chỉ số này cho biết nhà đầu tư sẵn sàng trả bao nhiêu cho một đồng lợi nhuận. P/E quá cao có thể là dấu hiệu định giá đắt.",
    'Nợ/Vốn CSH': "Tỷ lệ Nợ trên Vốn chủ sở hữu, đo lường đòn bẩy tài chính. Tỷ lệ < 1 thường được coi là an toàn.",
    'ROE': "Lợi nhuận trên Vốn chủ sở hữu. Đo lường khả năng sinh lời của công ty. ROE > 15% thường được xem là tốt.",
    'Biên LN': "Biên Lợi nhuận. Cho biết công ty tạo ra bao nhiêu lợi nhuận từ doanh thu. Biên lợi nhuận càng cao càng tốt.",
    # Database
    'RSI': "Chỉ số Sức mạnh Tương đối. Đo lường tốc độ và sự thay đổi của các biến động giá. RSI > 70 cho thấy tín hiệu 'quá mua' (có thể sớm điều chỉnh giảm).",
    'MACD': "Đường Trung bình động hội tụ/phân kỳ. Là một chỉ báo xu hướng. MACD > 0 thường báo hiệu xu hướng tăng.",
    'trailingPE_snapshot': "Tương tự P/E, tỷ lệ giá trên thu nhập trong 12 tháng gần nhất.",
    'marketCap_snapshot': "Vốn hóa thị trường. Các công ty có vốn hóa lớn thường ổn định và ít rủi ro hơn.",
    'Returns': "Tỷ suất lợi nhuận của cổ phiếu trong một khoảng thời gian."
}

# --- HÀM KẾT NỐI DATABASE ---
@st.cache_resource
def get_db_connection():
    """Tạo kết nối tới MySQL database"""
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='130225',      # <-- THAY ĐỔI MẬT KHẨU CỦA BẠN TẠI ĐÂY
            database='tickets',    # <-- THAY ĐỔI TÊN DATABASE CỦA BẠN TẠI ĐÂY
            port=3306
        )
        return connection
    except Error as e:
        st.error(f"❌ Lỗi kết nối database: {e}")
        return None

# --- HÀM LẤY DỮ LIỆU ---
@st.cache_data
def get_stock_data(tickers_list):
    """Lấy các chỉ số tài chính từ Yahoo Finance"""
    financial_data = []
    progress_bar = st.progress(0, text="Đang tải dữ liệu từ Yahoo Finance...")

    for idx, ticker_symbol in enumerate(tickers_list):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            data = {
                'Mã CP': ticker_symbol, 'Beta': info.get('beta'), 'P/E': info.get('trailingPE'),
                'Nợ/Vốn CSH': info.get('debtToEquity'), 'ROE': info.get('returnOnEquity'),
                'Biên LN': info.get('profitMargins')
            }
            financial_data.append(data)
        except Exception as e:
            st.warning(f"⚠️ Không thể lấy dữ liệu cho '{ticker_symbol}'.")
            financial_data.append({
                'Mã CP': ticker_symbol, 'Beta': None, 'P/E': None, 'Nợ/Vốn CSH': None,
                'ROE': None, 'Biên LN': None
            })
        progress_bar.progress((idx + 1) / len(tickers_list))

    progress_bar.empty()
    if not financial_data: return pd.DataFrame()

    df = pd.DataFrame(financial_data).set_index('Mã CP')
    for col in df.columns:
        if df[col].isnull().all():
            st.error(f"❌ Không có dữ liệu cho cột '{col}'")
            return pd.DataFrame()
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
    return df

@st.cache_data
def get_data_from_db(_conn):
    """Lấy dữ liệu từ MySQL database"""
    if _conn and _conn.is_connected():
        try:
            # <-- THAY ĐỔI TÊN BẢNG CỦA BẠN TẠI ĐÂY
            return pd.read_sql("SELECT * FROM tickets_combined", _conn)
        except Exception as e:
            st.error(f"❌ Lỗi truy vấn database: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# === HÀM TÍNH TOÁN CỦA AHP VÀ TOPSIS ===
def calculate_ahp_weights(comparison_matrix):
    """Tính trọng số (eigenvector) của ma trận AHP và Tỷ lệ nhất quán (CR)."""
    try:
        A = np.array(comparison_matrix)
        n = A.shape[0]

        eigenvalues, eigenvectors = np.linalg.eig(A)
        max_eigen_value = np.max(eigenvalues.real)
        
        max_eigen_vector_index = np.argmax(eigenvalues.real)
        weights = eigenvectors[:, max_eigen_vector_index].real
        weights = weights / weights.sum()
        
        CI = (max_eigen_value - n) / (n - 1) if n > 1 else 0
        
        RI_values = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        RI = RI_values.get(n, 1.49) 

        CR = CI / RI if RI != 0 else 0
        
        return weights, CR, pd.DataFrame(A)
    except Exception as e:
        st.error(f"Lỗi tính toán AHP: {e}")
        n_crit = len(comparison_matrix)
        return np.array([1/n_crit for _ in range(n_crit)]), 1.0, pd.DataFrame(np.identity(n_crit))


def run_topsis(decision_matrix, weights, impacts):
    """Thuật toán TOPSIS"""
    norm_matrix = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))
    weighted_matrix = norm_matrix * weights
    
    impacts_arr = np.array([1 if i == 'Benefit' else -1 for i in impacts])
    
    ideal_best = np.where(impacts_arr == 1, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_worst = np.where(impacts_arr == -1, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))

    dist_to_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    epsilon = 1e-6
    return dist_to_worst / (dist_to_best + dist_to_worst + epsilon)

# === CÁC HÀM TIỆN ÍCH KHÁC ===
def generate_analysis_text(results_df, weights, impacts):
    """Tạo ra các đoạn văn bản phân tích và gợi ý"""
    if results_df.empty: return "", "", ""
    
    best_stock_name = results_df.index[0]
    worst_stock_name = results_df.index[-1]
    best_stock_data = results_df.iloc[0]
    worst_stock_data = results_df.iloc[-1]
    
    impact_map = {'Benefit': 1, 'Cost': -1}
    numerical_impacts = {k: impact_map[v] for k, v in impacts.items()}

    best_reasons = []
    for crit, impact_val in numerical_impacts.items():
        if crit not in results_df.columns: continue
        val = best_stock_data[crit]
        if (impact_val == 1 and val >= results_df[crit].quantile(0.75)) or \
           (impact_val == -1 and val <= results_df[crit].quantile(0.25)):
            best_reasons.append(f"**{crit}** ({val:.2f})")

    best_analysis = f"🏆 **{best_stock_name}** được xếp hạng cao nhất. Điểm mạnh chính đến từ các chỉ số: {', '.join(best_reasons[:3])}."
    
    worst_reasons = []
    for crit, impact_val in numerical_impacts.items():
        if crit not in results_df.columns: continue
        val = worst_stock_data[crit]
        if (impact_val == 1 and val <= results_df[crit].quantile(0.25)) or \
           (impact_val == -1 and val >= results_df[crit].quantile(0.75)):
            worst_reasons.append(f"**{crit}** ({val:.2f})")
    
    worst_analysis = f"⚠️ **{worst_stock_name}** có rủi ro cao nhất trong danh sách, chủ yếu do các chỉ số chưa tốt như: {', '.join(worst_reasons[:3])}."

    actionable_advice = """
    💡 **Gợi ý hành động:**
    - **Đối với các cổ phiếu top đầu:** Đây là những ứng viên sáng giá. Hãy cân nhắc đưa vào danh sách theo dõi và **phân tích sâu hơn** về yếu tố cơ bản của doanh nghiệp.
    - **Đối với các cổ phiếu cuối bảng:** Cần thận trọng. Nếu đang nắm giữ, bạn nên **xem xét lại vị thế** và có thể đặt các biện pháp phòng ngừa rủi ro như **lệnh cắt lỗ (stop-loss)**.
    - **Lưu ý:** Kết quả này hoàn toàn dựa trên các chỉ số và trọng số bạn đã cung cấp. Đây là công cụ tham khảo, không phải lời khuyên đầu tư.
    """
    
    return best_analysis, worst_analysis, actionable_advice

def format_large_number(num):
    """Định dạng số lớn (vd: vốn hóa) thành dạng tỷ, triệu."""
    if pd.isna(num): return "N/A"
    num = float(num)
    if num >= 1e12: return f"{num / 1e12:.2f} nghìn tỷ"
    if num >= 1e9: return f"{num / 1e9:.2f} tỷ"
    if num >= 1e6: return f"{num / 1e6:.2f} triệu"
    return f"{num:,.2f}"

# --- GIAO DIỆN CHÍNH ---
st.title("⚖️ Ứng dụng Phân tích Rủi ro Đầu tư Cổ phiếu bằng TOPSIS và AHP")
st.info("🎯 Xếp hạng rủi ro cổ phiếu dựa trên các chỉ số tài chính với trọng số tùy chỉnh (có thể dùng AHP để tính trọng số)")

# --- THANH ĐIỀU KHIỂN (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Bảng điều khiển")
    data_source = st.radio("📂 Chọn nguồn dữ liệu:", ["Yahoo Finance API", "Database MySQL"], key="data_source")

    weights = {}
    CR = 1.0 
    
    if data_source == "Yahoo Finance API":
        criteria_yf = {'Beta': -1, 'P/E': -1, 'Nợ/Vốn CSH': -1, 'ROE': 1, 'Biên LN': 1}
        criteria_list = list(criteria_yf.keys())
        n_criteria = len(criteria_list)
        tickers_input = st.text_area("Nhập mã cổ phiếu (phân cách bởi dấu phẩy):", "AAPL, MSFT, GOOGL, TSLA, NVDA")
        col_impacts = {k: ('Benefit' if v == 1 else 'Cost') for k, v in criteria_yf.items()}
    else:
        criteria_list = DB_CRITERIA
        n_criteria = len(criteria_list)
        col_impacts = DB_IMPACTS

    st.header("⚖️ Thiết lập Trọng số")
    
    tab_ahp, tab_manual = st.tabs(["🔢 Tính Trọng số bằng AHP", "🛠️ Nhập Trọng số Thủ công"])

    # ----------------------------------------------------
    # TAB 1: TÍNH TRỌNG SỐ BẰNG AHP
    # ----------------------------------------------------
    with tab_ahp:
        with st.expander("⚖️ So sánh độ quan trọng của các tiêu chí (Nhập liệu)", expanded=True):
            for crit_i, crit_j in itertools.combinations(criteria_list, 2):
                impact_display_i = f" (*{col_impacts.get(crit_i, '')}*)" if data_source != "Yahoo Finance API" else ""
                impact_display_j = f" (*{col_impacts.get(crit_j, '')}*)" if data_source != "Yahoo Finance API" else ""

                comparison_value = st.select_slider(
                    f"**{crit_i}**{impact_display_i} quan trọng hơn **{crit_j}**{impact_display_j}?",
                    options=[1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    value=1.0,
                    format_func=lambda x: f"1/{round(1/x)}" if x < 1 else str(round(x)),
                    key=f"ahp_{crit_i}_{crit_j}_{data_source}",
                    help=f"So sánh {CRITERIA_EXPLANATIONS.get(crit_i)} vs {CRITERIA_EXPLANATIONS.get(crit_j)}"
                )
                st.session_state[f'ahp_comp_{crit_i}_{crit_j}'] = comparison_value

        comparison_matrix = [[0.0] * n_criteria for _ in range(n_criteria)]
        for i in range(n_criteria):
            for j in range(n_criteria):
                if i == j:
                    comparison_matrix[i][j] = 1.0
                elif i < j:
                    crit_i, crit_j = criteria_list[i], criteria_list[j]
                    comp_val = st.session_state.get(f'ahp_comp_{crit_i}_{crit_j}', 1.0)
                    comparison_matrix[i][j] = comp_val
                else:
                    comparison_matrix[i][j] = 1.0 / comparison_matrix[j][i]
                    
        ahp_weights_array, CR, AHP_Matrix_df = calculate_ahp_weights(comparison_matrix)
        
        st.session_state['weights_source'] = 'AHP'
        st.session_state['CR'] = CR
        st.session_state['AHP_Matrix_df'] = AHP_Matrix_df
        st.session_state['ahp_weights'] = {criteria_list[i]: ahp_weights_array[i] for i in range(n_criteria)}

        st.markdown("---")
        with st.expander("💡 Giải thích Tỷ lệ nhất quán (CR)"):
            st.info(
                """
                **CR (Consistency Ratio)** đo lường mức độ logic và không mâu thuẫn trong các so sánh cặp của bạn.
                - **CR ≤ 0.1 (Tốt):** Các đánh giá nhất quán, kết quả đáng tin cậy.
                - **CR > 0.1 (Cần xem lại):** Có sự mâu thuẫn (VD: A > B, B > C, nhưng C > A). 
                
                Bạn nên điều chỉnh lại các so sánh để kết quả đáng tin cậy hơn.
                """
            )

        with st.expander("📊 Kết quả AHP (Trọng số & CR)", expanded=True):
            st.markdown(f"**Tỷ lệ nhất quán (CR):**")
            st.code(f"{CR:.4f}", language=None)
            if CR > 0.1:
                st.error("⚠️ CR > 0.1. Mức độ nhất quán thấp.")
            else:
                st.success("✅ CR ≤ 0.1. Mức độ nhất quán chấp nhận được.")
                
            weights_df = pd.DataFrame(st.session_state['ahp_weights'].items(), columns=['Tiêu chí', 'Trọng số']).set_index('Tiêu chí')
            st.markdown("**Trọng số được tính toán:**")
            st.dataframe(weights_df.style.format("{:.4f}"))

        with st.expander("📋 Xem Ma trận So sánh Cặp (Đầu vào AHP)"):
            AHP_Matrix_df.columns = criteria_list
            AHP_Matrix_df.index = criteria_list
            st.dataframe(AHP_Matrix_df.style.format("{:.3f}").background_gradient(cmap='Blues', axis=None), use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: NHẬP TRỌNG SỐ THỦ CÔNG
    # ----------------------------------------------------
    with tab_manual:
        st.subheader("Nhập Trọng số Thủ công")
        
        default_weights = {k: 1.0 / n_criteria for k in criteria_list}
        
        if data_source == "Yahoo Finance API":
            preset = st.selectbox("🎚️ Chọn bộ trọng số mẫu:", ["Tùy chỉnh", "An toàn", "Cân bằng", "Tăng trưởng"], key="preset_yf_man")

            if preset == "An toàn": default_weights = {'Beta': 0.3, 'P/E': 0.2, 'Nợ/Vốn CSH': 0.3, 'ROE': 0.1, 'Biên LN': 0.1}
            elif preset == "Cân bằng": default_weights = {'Beta': 0.2, 'P/E': 0.2, 'Nợ/Vốn CSH': 0.2, 'ROE': 0.2, 'Biên LN': 0.2}
            elif preset == "Tăng trưởng": default_weights = {'Beta': 0.1, 'P/E': 0.1, 'Nợ/Vốn CSH': 0.1, 'ROE': 0.35, 'Biên LN': 0.35}
        
        manual_weights = {}
        for crit in criteria_list:
            impact_display = f" (*{col_impacts.get(crit, '')}*)" if data_source != "Yahoo Finance API" else ""
            manual_weights[crit] = st.slider(
                f"🎯 {crit}{impact_display}", 0.0, 1.0, default_weights.get(crit, 1.0/n_criteria), 0.05, 
                key=f'w_{crit}_man_slider', help=CRITERIA_EXPLANATIONS.get(crit, "")
            )
        
        st.session_state['weights_source'] = 'Manual'
        st.session_state['manual_weights'] = manual_weights
        st.session_state['CR'] = 0.0 

    # ----------------------------------------------------
    # LOGIC CHUNG SAU KHI THIẾT LẬP TRỌNG SỐ
    # ----------------------------------------------------
    if st.session_state.get('weights_source') == 'AHP':
        weights = st.session_state.get('ahp_weights', {})
        CR = st.session_state.get('CR', 1.0)
    elif st.session_state.get('weights_source') == 'Manual':
        weights = st.session_state.get('manual_weights', {})
        CR = 0.0
    
    total_weight = sum(weights.values())
    st.markdown("---")
    st.markdown("**Trạng thái Trọng số Hiện tại**")
    if abs(total_weight - 1.0) > 0.01 and st.session_state.get('weights_source') == 'Manual':
        st.warning(f"⚠️ Tổng trọng số: {total_weight:.2f} (Cần = 1.0)")
    else:
        st.success(f"✅ Tổng trọng số: {total_weight:.2f}")

# --- NÚT PHÂN TÍCH VÀ HIỂN THỊ KẾT QUẢ ---
if st.button("🚀 Bắt đầu Phân tích", use_container_width=True):
    
    CR = st.session_state.get('CR', 1.0)
    
    if CR > 0.1:
        st.error("CR > 0.1. Vui lòng điều chỉnh lại các so sánh cặp trong Tab AHP để đảm bảo tính nhất quán trước khi phân tích.")
        st.stop()
        
    if not weights or sum(weights.values()) == 0:
        st.error("❌ Trọng số không hợp lệ. Vui lòng thiết lập trọng số trong bảng điều khiển.")
        st.stop()
        
    normalized_weights = {k: v / sum(weights.values()) for k, v in weights.items()}
    
    raw_data = pd.DataFrame()
    if data_source == "Yahoo Finance API":
        tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        if not tickers_list:
            st.error("❌ Vui lòng nhập ít nhất một mã cổ phiếu.")
            st.stop()

        with st.spinner(f"⏳ Đang tải dữ liệu cho {len(tickers_list)} cổ phiếu..."):
            raw_data = get_stock_data(tickers_list)
        
        criteria = {k: ('Benefit' if v == 1 else 'Cost') for k, v in criteria_yf.items()}
    else:
        with st.spinner("⏳ Đang kết nối và xử lý dữ liệu từ database..."):
            conn = get_db_connection()
            if not conn: st.stop()
            
            df_raw = get_data_from_db(conn)
            if df_raw.empty: st.stop()

            ticker_col = 'Ticker'
            if ticker_col not in df_raw.columns:
                st.error(f"❌ Không tìm thấy cột '{ticker_col}' trong database.")
                st.stop()

            df_raw['Date'] = pd.to_datetime(df_raw.get('Date'), errors='coerce')
            df_grouped = df_raw.sort_values('Date', ascending=True).groupby(ticker_col).last().reset_index()
            
            missing_cols = [col for col in DB_CRITERIA if col not in df_grouped.columns]
            if missing_cols:
                st.error(f"❌ Database thiếu các cột: {', '.join(missing_cols)}")
                st.stop()

            st.success(f"✅ Đã xử lý {len(df_grouped)} mã cổ phiếu.")
        
        raw_data = df_grouped.set_index(ticker_col)[DB_CRITERIA].copy()
        raw_data.index.name = 'Mã CP'

        for col in raw_data.columns:
            raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce').fillna(raw_data[col].median())
        criteria = DB_IMPACTS

    if raw_data.empty:
        st.error("❌ Không có dữ liệu để phân tích.")
        st.stop()

    st.header("📊 Dữ liệu Đầu vào sau khi xử lý")
    formatters = {col: "{:,.2f}" for col in raw_data.columns}
    if 'marketCap_snapshot' in raw_data.columns:
        formatters['marketCap_snapshot'] = format_large_number
    st.dataframe(raw_data.style.format(formatters).background_gradient(cmap='YlOrRd', axis=0), use_container_width=True)

    st.header("🏆 Kết quả Xếp hạng TOPSIS")
    decision_matrix = raw_data[list(criteria.keys())]
    weights_list = np.array([normalized_weights[crit] for crit in criteria])
    impacts_list = list(criteria.values())

    scores = run_topsis(decision_matrix.values, weights_list, impacts_list)

    results_df = raw_data.copy()
    results_df['Điểm TOPSIS'] = scores
    results_df['Xếp hạng'] = results_df['Điểm TOPSIS'].rank(ascending=False).astype(int)
    results_df = results_df.sort_values(by='Xếp hạng')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🥇 Tốt nhất (Ít rủi ro nhất)", results_df.index[0], f"{results_df['Điểm TOPSIS'].iloc[0]:.4f}")
    col2.metric("🥈 Rủi ro nhất", results_df.index[-1], f"{results_df['Điểm TOPSIS'].iloc[-1]:.4f}")
    col3.metric("📊 Điểm trung bình", f"{results_df['Điểm TOPSIS'].mean():.4f}")
    
    result_formatters = {col: "{:,.4f}" for col in results_df.columns if pd.api.types.is_numeric_dtype(results_df[col]) and col not in ['Xếp hạng']}
    result_formatters['Điểm TOPSIS'] = "{:.4f}"
    if 'marketCap_snapshot' in results_df.columns:
        result_formatters['marketCap_snapshot'] = format_large_number

    st.dataframe(
        results_df.style
        .apply(lambda row: ['background-color: #2E8B57; color: white; font-weight: bold' if row['Xếp hạng'] == 1 else '' for _ in row], axis=1)
        .background_gradient(cmap='Greens', subset=['Điểm TOPSIS'])
        .format(result_formatters),
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("📝 Phân tích và Gợi ý")
    best_analysis, worst_analysis, actionable_advice = generate_analysis_text(results_df, normalized_weights, criteria)
    st.markdown(best_analysis)
    st.markdown(worst_analysis)
    st.markdown(actionable_advice)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Biểu đồ So sánh", "🎯 Biểu đồ Radar", "📥 Tải xuống"])
    with tab1:
        fig = px.bar(results_df, x=results_df.index, y='Điểm TOPSIS', color='Điểm TOPSIS',
                     color_continuous_scale='Greens', title='So sánh Điểm TOPSIS giữa các Cổ phiếu', text='Điểm TOPSIS')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(xaxis_title="Mã Cổ phiếu", yaxis_title="Điểm TOPSIS", xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        top_n = min(5, len(results_df))
        top_stocks = results_df.head(top_n)
        norm_data = (top_stocks[list(criteria.keys())] - top_stocks[list(criteria.keys())].min()) / \
                      (top_stocks[list(criteria.keys())].max() - top_stocks[list(criteria.keys())].min())
        fig_radar = go.Figure()
        for ticker in top_stocks.index:
            fig_radar.add_trace(go.Scatterpolar(r=norm_data.loc[ticker].values, theta=list(criteria.keys()), fill='toself', name=ticker))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title=f"So sánh Top {top_n} Cổ phiếu (dữ liệu đã chuẩn hóa)")
        st.plotly_chart(fig_radar, use_container_width=True)
    with tab3:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Ket_qua_TOPSIS')
            raw_data.to_excel(writer, sheet_name='Du_lieu_goc')
            pd.DataFrame({
                'Chỉ số': list(normalized_weights.keys()),
                'Trọng số': list(normalized_weights.values()),
                'Tác động': [criteria[k] for k in normalized_weights.keys()]
            }).to_excel(writer, sheet_name='Cau_hinh_phan_tich', index=False)
        
        st.download_button("📊 Tải báo cáo chi tiết (.xlsx)", output.getvalue(),
                          f"topsis_ahp_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    try:
        # 1. Tính toán kết quả TOPSIS (Độc lập)
        num_criteria = len(criteria)
        independent_weights_list = np.array([1 / num_criteria] * num_criteria)
        independent_scores = run_topsis(decision_matrix.values, independent_weights_list, impacts_list)
        
        independent_results_df = raw_data.copy()
        independent_results_df['Điểm TOPSIS (Độc lập)'] = independent_scores
        independent_results_df['Hạng (Độc lập)'] = independent_results_df['Điểm TOPSIS (Độc lập)'].rank(ascending=False).astype(int)
        independent_results_df = independent_results_df.sort_values(by='Hạng (Độc lập)')

        # 2. Lưu cả 3 kết quả vào session_state
        st.session_state['results_cr'] = CR
        st.session_state['results_topsis_independent'] = independent_results_df
        st.session_state['results_combined'] = results_df

        # 3. Đặt cờ báo hiệu đã chạy thành công
        st.session_state['analysis_run_successfully'] = True

    except Exception as e:
        st.error(f"Lỗi khi tính toán so sánh: {e}")
        st.session_state['analysis_run_successfully'] = False

st.markdown("---")
st.markdown("🔬 **TOPSIS & AHP Analysis System**")
if st.session_state.get('analysis_run_successfully', False):
    st.markdown("---")
    st.header("🏁 Đánh giá Hiệu quả 3 Giai đoạn (Định lượng)")

    # --- 1. Đánh giá AHP (Độc lập) ---
    st.subheader("1. Hiệu quả AHP (Độc lập)")
    st.info(
        "Hiệu quả của AHP được đo bằng **Tỷ lệ Nhất quán (CR)**. "
        "Chỉ số này cho biết các so sánh cặp của bạn có logic và nhất quán hay không."
    )

    cr_score = st.session_state.get('results_cr', 1.0)
    st.metric(label="Tỷ lệ Nhất quán (CR)", value=f"{cr_score:.4f}")
    if cr_score <= 0.1:
        st.success("✅ **Đánh giá: Hiệu quả.** (CR <= 0.1). Các so sánh của bạn nhất quán, bộ trọng số đáng tin cậy.")
    else:
        st.error("⚠️ **Đánh giá: Không Hiệu quả.** (CR > 0.1). Các so sánh của bạn mâu thuẫn. Cần xem lại các so sánh cặp trong tab AHP.")

    # --- 2. Đánh giá TOPSIS (Độc lập) ---
    st.subheader("2. Hiệu quả TOPSIS (Độc lập)")
    st.info(
        "Bản thân TOPSIS độc lập (với trọng số bằng nhau) không có 'điểm hiệu quả'. "
        "Vai trò của nó là tạo ra một **Xếp hạng Cơ sở (Baseline)** để làm nền so sánh."
    )
    st.markdown("Xem Xếp hạng Cơ sở trong bảng so sánh bên dưới.")

    # --- 3. Đánh giá Kết hợp (AHP + TOPSIS) ---
    st.subheader("3. Hiệu quả Kết hợp (AHP + TOPSIS)")
    st.info(
        "Hiệu quả của mô hình kết hợp được đo bằng **mức độ tác động** của AHP lên kết quả của TOPSIS. "
        "Chúng ta dùng **Hệ số Tương quan Hạng Spearman (Rho)** để đo lường điều này."
    )
    try:
        # Lấy 2 bảng xếp hạng
        df_combined = st.session_state['results_combined']['Xếp hạng']
        df_independent = st.session_state['results_topsis_independent']['Hạng (Độc lập)']
        
        # Căn chỉnh 2 bảng theo index (Mã CP) để đảm bảo so sánh đúng
        df_independent = df_independent.reindex(df_combined.index)
        
        # Tính toán tương quan hạng
        correlation, p_value = spearmanr(df_combined, df_independent)      
        st.metric(label="Hệ số Tương quan Hạng (Spearman's Rho)", value=f"{correlation:.4f}")     

        interpretation = ""
        if correlation > 0.8:
            interpretation = ("**Giống nhau (Tương quan > 0.8):** "
                            "Việc dùng AHP gần như **không làm thay đổi** kết quả xếp hạng. Điều này xảy ra khi các trọng số AHP gần bằng nhau.")
        elif correlation > 0.4:
            interpretation = ("**Khá tương đồng (Tương quan 0.4 - 0.8):** "
                            "Việc dùng AHP **có điều chỉnh** thứ hạng, nhưng xu hướng chung vẫn được giữ nguyên.")
        else:
            interpretation = ("**Rất khác biệt (Tương quan < 0.4):** "
                            "Việc dùng AHP đã **thay đổi đáng kể** kết quả, chứng tỏ trọng số của bạn có tác động lớn.")
        
        st.success(
            f"""
            **Giải thích:**
            - Gần **1.0**: Hai bảng xếp hạng y hệt nhau.
            - Gần **0.0**: Hai bảng xếp hạng không liên quan.
            **Đánh giá của bạn:** {interpretation}
            """
        )

    except Exception as e:
        st.error(f"Lỗi tính toán tương quan: {e}")