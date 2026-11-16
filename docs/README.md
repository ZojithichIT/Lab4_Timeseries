<div align="center">

# Lab 4: Dự Báo Chuỗi Thời Gian Đa Biến (Multivariate Time Series Forecasting)
### Electricity Load + Weather · Feature Engineering, Chuẩn Hóa `StandardScaler`, Huấn Luyện `VAR` / `XGBoost` / `LSTM` & Walk-forward Validation

<p>
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter" alt="Jupyter Notebook" />
  <img src="https://img.shields.io/badge/Topic-Multivariate%20TS-brightgreen?style=for-the-badge" alt="Multivariate Time Series" />
  <img src="https://img.shields.io/badge/Model-VAR%20%7C%20XGBoost%20%7C%20LSTM-red?style=for-the-badge" alt="Models" />
  <img src="https://img.shields.io/badge/Library-Statsmodels%20%7C%20XGBoost%20%7C%20DL-purple?style=for-the-badge" alt="Libraries" />
</p>

| Thuộc tính | Chi tiết |
|---|---|
| **Đề bài tham chiếu** | [`practice 4.pdf`](../../practice%204.pdf) (*Multivariate Time Series Forecasting: Electricity Load + Weather*) |
| **Notebook chính** | `notebooks/main.ipynb` *(Dự kiến — EDA, tiền xử lý, VAR, XGBoost, LSTM & Walk-forward Validation)* |
| **Dữ liệu chính (Dataset)** | **Electricity Load Diagrams Dataset (UCI)** + Thời tiết ($X_t = [load_t,\ temp_t,\ humidity_t,\ wind\_speed_t]$) |
| **Mục tiêu dự báo** | Cửa sổ lịch sử $L = 24$ giờ gần nhất dự báo phụ tải bước kế tiếp: $X_{t-24:t} \Rightarrow load_{t+1}$ |
| **Mô hình cốt lõi** | `VAR` (Statistical Baseline), `XGBoost Regressor` (Machine Learning), `LSTM(64)->LSTM(32)->Dense(1)` (Deep Learning) |
| **Kỹ thuật thẩm định** | Time-based Train/Val/Test Split ($70\% / 15\% / 15\%$, `shuffle = False`), **Walk-forward Validation (Backtesting)** |
| **Độ đo đánh giá** | RMSE ($\text{kW/MW}$), MAE ($\text{kW/MW}$), MAPE ($\%$) |

</div>

---

## 1. Giới Thiệu Dự Án

Dự án này tập trung vào bài toán **dự báo chuỗi thời gian đa biến (Multivariate Time Series Forecasting)**, kết hợp dữ liệu phụ tải điện năng với các biến ngoại sinh thời tiết (`temperature`, `humidity`, `wind_speed`). Thay vì chỉ dựa vào lịch sử của riêng chuỗi mục tiêu như bài toán đơn biến, mô hình khai thác đồng thời tương quan chéo giữa điện năng và thời tiết để dự báo $load_{t+1}$ từ cửa sổ $24$ giờ gần nhất.

Mục tiêu chính bao gồm:
* **Khám phá & Hiểu dữ liệu đa biến (Data Understanding):** Kiểm tra missing values, outliers, tính mùa vụ (chu kỳ ngày, chu kỳ tuần) và trực quan hóa quan hệ `load` theo thời gian, `load` vs `temperature` cùng ma trận tương quan.
* **Tiền xử lý chuẩn hóa & Chia tập chống rò rỉ:** Chuẩn hóa dữ liệu bằng `StandardScaler` (giúp `LSTM` học ổn định, tránh bias khác đơn vị đo) và phân chia tuần tự theo thời gian tỷ lệ $70\% / 15\% / 15\%$.
* **Feature Engineering cho ML & DL:** Tạo cửa sổ trễ $L = 24$ giờ cho cả $4$ biến ($24 \times 4 = 96$ đặc trưng phẳng cho `XGBoost`, Tensor 3D `(batch_samples, 24, 4)` cho `LSTM`), bổ sung `rolling_mean_6`, `rolling_std_24` và `load_trend` ($load_t - load_{t-24}$) cho `XGBoost`.
* **Triển khai & So sánh 3 trường phái mô hình:** Đối chiếu trực tiếp mô hình thống kê (`VAR` với kiểm định ADF, sai phân, chọn bậc trễ AIC/BIC), Học máy (`XGBoost` với `Early Stopping`) và Học sâu (`LSTM(64) -> LSTM(32) -> Dense(1)`, Loss `MSE`, Optimizer `Adam`).
* **Backtesting & Mở rộng nâng cao:** Thực hiện **Walk-forward Validation** mô phỏng dự báo thực tế cho cả 3 mô hình, trả lời 6 câu hỏi phân tích và thử nghiệm bài toán Traffic Forecasting (`METR-LA` / `PEMS-BAY` — $207$ trạm, đa bước $15, 30, 60$ phút).

---

## 2. Cấu Trúc Thư Mục

```text
Lab4_TS/
├── data/
│   ├── raw/                   # Dữ liệu thô (Electricity Load Diagrams + Weather; METR-LA / PEMS-BAY)
│   └── processed/             # Dữ liệu đã làm sạch, chuẩn hóa (StandardScaler) & chia Train/Val/Test (70/15/15)
├── notebooks/                 # Jupyter Notebook thực hành (EDA, trực quan hóa & huấn luyện VAR/XGBoost/LSTM)
├── scripts/                   # Module Python đóng gói (data loader, preprocessing, features, models, backtesting, CLI)
├── models/                    # Trọng số mô hình đã lưu (.joblib, .keras/.pt), scaler và model_registry.json
├── logs/                      # Log huấn luyện, lịch sử thực nghiệm (JSON/CSV), leaderboard & kết quả dự báo
└── docs/
    ├── README.md              # Trang chủ tài liệu & chỉ mục dự án Lab 4
    └── images/                # Biểu đồ trực quan hóa EDA, tương quan & so sánh dự báo
```

---

## 3. Lộ Trình Thực Hành & Hạng Mục Triển Khai

* 🔍 **Phần 1 — Khám Phá & Tiền Xử Lý Dữ Liệu (`Data Understanding & Preprocessing`):**
  Kiểm tra missing values, outliers, seasonality (ngày/tuần); trực quan hóa `load` theo thời gian, `load` vs `temperature`, Correlation Matrix; chuẩn hóa `StandardScaler` và chia tập `70% / 15% / 15%` (`shuffle = False`).

* 🧩 **Phần 2 — Tạo Đặc Trưng Đa Biến (`Feature Engineering`):**
  Thiết kế cửa sổ trễ $L = 24$ giờ cho 4 biến (`load`, `temperature`, `humidity`, `wind_speed`); bổ sung `rolling_mean_6`, `rolling_std_24` và `load_trend` cho mô hình `XGBoost`; định dạng Tensor 3D `(batch_samples, 24, 4)` cho `LSTM`.

* 📊 **Phần 3 — Huấn Luyện 3 Nhóm Mô Hình & Backtesting:**
  1. **`VAR` (Statistical Baseline):** Kiểm định tính dừng `ADF`, sai phân nếu cần, chọn bậc trễ theo `AIC/BIC`, dự báo vector và trích cột `load` để đánh giá.
  2. **`XGBoost` (Machine Learning):** Huấn luyện trên vector phẳng kèm rolling/trend features, tinh chỉnh `max_depth`, `n_estimators` với `Early Stopping`.
  3. **`LSTM` (Deep Learning):** Kiến trúc `LSTM(64) -> LSTM(32) -> Dense(1)`, Loss `MSE`, Optimizer `Adam`.
  4. **Walk-forward Validation:** `Train -> predict t+1 -> thêm vào chuỗi -> predict t+2 -> ...` áp dụng đồng nhất cho cả 3 mô hình.

* 📈 **Phần 4 — Đánh Giá, 6 Câu Hỏi Phân Tích & Bài Tập Nâng Cao:**
  So sánh `RMSE`, `MAE`, `MAPE` giữa 3 mô hình; phân tích ảnh hưởng biến thời tiết, trường hợp `VAR` thất bại, chu kỳ `LSTM` học được, hiện tượng Overfitting và sự khác biệt ML vs DL; mở rộng sang bài toán Traffic Forecasting (`METR-LA` / `PEMS-BAY`, dự báo $15, 30, 60$ phút).

---

## 4. Hướng Dẫn Cài Đặt & Thực Thi

### 4.1. Thiết lập môi trường
```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn xgboost tensorflow
```

### 4.2. Khởi chạy thực nghiệm *(sau khi hoàn thiện code trong `notebooks/` và `scripts/`)*
* **Cách 1 — Chạy trực tiếp trên Jupyter Notebook:**
  ```bash
  jupyter notebook notebooks/main.ipynb
  ```
* **Cách 2 — Chạy pipeline huấn luyện & backtesting qua CLI:**
  ```bash
  python scripts/train.py
  ```
* **Cách 3 — Chạy dự báo từ mô hình đã lưu:**
  ```bash
  python scripts/predict.py
  ```
