# Lab 4: Dự Báo Chuỗi Thời Gian Đa Biến (Multivariate Time Series Forecasting)

**Chủ đề:** Electricity Load + Weather (`VAR` vs `XGBoost` vs `LSTM`)  
**Đề bài tham chiếu:** [`practice 4.pdf`](../practice%204.pdf)

---

## 1. Mục Tiêu Dự Án

Xây dựng và so sánh toàn diện 3 nhóm phương pháp dự báo chuỗi thời gian đa biến trên dữ liệu phụ tải điện kết hợp các biến thời tiết:

* **Mô hình thống kê (Statistical Baseline):** `VAR` (Vector Autoregression) — kiểm định tính dừng (ADF), sai phân (Differencing) và chọn bậc trễ theo `AIC` / `BIC`.
* **Học máy (Machine Learning):** `XGBoost` — chuyển chuỗi đa biến sang dạng vector phẳng kèm Rolling Features & Load Trend, tinh chỉnh `max_depth`, `n_estimators` với `Early Stopping`.
* **Học sâu (Deep Learning):** `LSTM` (`LSTM(64) -> LSTM(32) -> Dense(1)`) — dữ liệu đầu vào Tensor 3D `(batch_samples, 24, 4)`, hàm mất mát `MSE`, bộ tối ưu `Adam`.
* **Thẩm định thực tế:** Áp dụng **Walk-forward Validation (Backtesting)** cho cả 3 mô hình và đánh giá qua 3 chỉ số: **`RMSE`**, **`MAE`**, **`MAPE`**.

---

## 2. Bài Toán & Dữ Liệu

* **Bộ dữ liệu chính:** Electricity Load Diagrams Dataset (UCI) kết hợp dữ liệu thời tiết.
* **Vector quan sát tại thời điểm $t$:**
  $$X_t = [load_t,\ temp_t,\ humidity_t,\ wind\_speed_t]$$
* **Mục tiêu dự báo:** Sử dụng cửa sổ lịch sử $L = 24$ giờ gần nhất để dự báo phụ tải điện ở giờ tiếp theo:
  $$X_{t-24:t} \Rightarrow load_{t+1}$$
* **Tỷ lệ chia dữ liệu theo thời gian (`shuffle = False`):** `70% Train` / `15% Validation` / `15% Test`.
* **Chuẩn hóa dữ liệu:** Áp dụng `StandardScaler` trên các biến (fit trên tập Train để tránh Data Leakage).
* **Bài tập nâng cao (Mở rộng):** Traffic Forecasting trên bộ dữ liệu `METR-LA` hoặc `PEMS-BAY` ($207$ trạm cảm biến tốc độ mỗi $5$ phút, dự báo đa bước $15, 30, 60$ phút: $[X_{t-12}, \dots, X_t] \Rightarrow X_{t+3}, X_{t+6}, X_{t+12}$).

---

## 3. Cấu Trúc Thư Mục

```text
Lab4_TS/
├── README.md                  # Tài liệu tổng quan, kiến trúc và lộ trình thực hiện Lab 4
├── data/
│   ├── raw/                   # Dữ liệu thô (Electricity Load + Weather; METR-LA / PEMS-BAY)
│   └── processed/             # Dữ liệu đã làm sạch, chuẩn hóa (StandardScaler) & chia Train/Val/Test (70/15/15)
├── notebooks/                 # Jupyter Notebook cho EDA, trực quan hóa & thử nghiệm các mô hình
├── scripts/                   # Mã nguồn Python module hóa (data loader, features, VAR/XGBoost/LSTM, walk-forward, CLI)
├── models/                    # Trọng số mô hình đã lưu (.joblib, .keras/.pt), scaler và model_registry.json
├── logs/                      # Log huấn luyện, bảng xếp hạng metrics (CSV/JSON) và kết quả dự báo
└── docs/
    └── images/                # Báo cáo phân tích, câu hỏi thảo luận & các biểu đồ trực quan hóa
```

---

## 4. Lộ Trình Triển Khai (Theo Đề Bài `practice 4.pdf`)

1. **Data Understanding & EDA:**
   - Kiểm tra Missing values, Outliers, Seasonality (chu kỳ ngày, chu kỳ tuần).
   - Trực quan hóa: `load` theo thời gian, `load` vs `temperature`, ma trận tương quan (Correlation Matrix).
2. **Preprocessing & Feature Engineering:**
   - Chuẩn hóa dữ liệu bằng `StandardScaler` và chia tập theo thứ tự thời gian (`70% / 15% / 15%`).
   - Tạo cửa sổ trễ $L = 24$ giờ cho cả 4 biến ($24 \times 4 = 96$ đặc trưng đầu vào).
   - Riêng cho **XGBoost**, bổ sung: `rolling_mean_6`, `rolling_std_24`, và `load_trend` ($load_t - load_{t-24}$).
3. **Huấn luyện 3 Mô Hình & Backtesting:**
   - **VAR:** Kiểm định ADF $\rightarrow$ sai phân (nếu cần) $\rightarrow$ chọn $p$ theo AIC/BIC $\rightarrow$ dự báo vector và trích cột `load`.
   - **XGBoost:** Vector phẳng $(96 + \text{rolling})$ $\rightarrow$ dự báo $load_{t+1}$ với Early Stopping trên tập Validation.
   - **LSTM:** Tensor 3D `(samples, 24, 4)` $\rightarrow$ `LSTM(64) -> LSTM(32) -> Dense(1)` (Loss: `MSE`, Optimizer: `Adam`).
   - Thực hiện **Walk-forward Validation** (`Train -> predict t+1 -> cập nhật chuỗi -> predict t+2 -> ...`).
4. **Đánh giá & Trả lời 6 Câu Hỏi Phân Tích:**
   - So sánh `RMSE`, `MAE`, `MAPE` giữa `VAR`, `XGBoost` và `LSTM`.
   - Phân tích ảnh hưởng thời tiết, trường hợp VAR thất bại, chu kỳ LSTM học được, hiện tượng Overfitting và sự khác biệt giữa ML và DL.
5. **Bài Tập Nâng Cao:**
   - Khảo sát bài toán Traffic Forecasting (`METR-LA` / `PEMS-BAY`) dự báo đa bước $15, 30, 60$ phút.
