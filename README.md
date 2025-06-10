# US Accidents Prediction Project

本專案旨在利用美國車禍數據進行預測與視覺化，協助找出高風險地區與時間，以優化事故應對資源配置與成本控制。

---

## 📅 專案時程與分工

| 日期        | 任務內容                        | 負責人 |
|-------------|-------------------------------|--------|
| 6/1 23:59   | 完成項目 0, 1, 2, 4            | 澤、廷 |
| 6/1 23:59   | 模型預測任務                   | 翔     |
| 6/2 23:59   | 地圖功能實作                   | 江     |
| 6/4 23:59   | 提交完整程式碼與論文初稿       | 全員   |
| 6/5 23:59   | 提交簡報與 Demo 影片（30 秒） | 全員   |
| 6/6 08:00   | 上台報告（10 分鐘 + QA 3 分鐘）| 全員   |

---

## 📌 任務分工與模組說明

### 0. 資料精簡策略
- **目標**：降低資料量，提升處理效率
- **策略**：
  - 僅保留 2019–2021 年資料
  - 僅保留有經緯度與完整天氣欄位的資料
  - 可針對 CA、TX、NY 等大州進行篩選
  - 精選欄位：`Severity`, `Start_Time`, `Weather_Condition`, `Temperature`, `Wind_Speed` 等
- **輸出**：`cleaned_accidents_2019_2021.csv` / `.parquet`

---

### 1. 資料處理與初步探索
- 欄位清理與命名標準化
- 時間欄位轉換（年月日、時段、星期）
- 統一類別命名（如天氣條件）
- 繪製分布圖：嚴重度、時間、州別、天氣
- **輸出**：`EDA.ipynb`

---

### 2. 特徵工程
- 數值特徵：風速、溫度、能見度標準化或分箱
- 類別特徵：One-Hot / Target Encoding
- 地理特徵：H3 格子編碼 or 城市聚合
- 平衡處理：使用 SMOTE-NC 增加嚴重事故樣本
- **輸出**：`features_ready.csv` / `.pkl`

---

### 3. 模型建立與預測
- 分類任務：Severity（或轉為 binary）
- 模型：Logistic Regression（baseline）、LightGBM（主模型）
- 驗證方式：Stratified K-Fold、F1-score、Recall
- **輸出**：
  - `model.pkl`
  - `tuned_model_results.csv`

---

### 4. 模型解釋
- 使用 SHAP 解釋特徵影響力
- SHAP Summary Plot & Dependence Plot
- 撰寫分析：哪些因素會導致嚴重事故（夜間、低能見度等）
- **輸出**：
  - `shap_summary.png`
  - `shap_dependence.png`

---

### 5. 熱點分析與空間預測
- 建立格網（H3）統計歷史事故風險
- 預測模型：特定格子是否高風險
- 建立 Choropleth 或地圖點資料
- **輸出**：
  - `risk_score_by_grid.csv`
  - `risk_map_data.geojson`

---

### 6. 視覺化展示：互動地圖
- 使用 **Folium** 搭配 **Streamlit** 建立互動式地圖
- 歷史資料展示（時間軸可滑動）
- 預測熱點展示（CircleMarker / Choropleth）
- 加入條件篩選器：
  - 時段（早/晚）
  - 天氣（晴/雨/霧）
  - 嚴重度（1～4）
  - 州別（可選擇）
- **輸出**：`map_app.py`

---

## 📦 最終提交（6/5 前）
- ✅ `code/`：完整 Python 原始碼與模型
- ✅ `slides/`：簡報投影片（含模型架構與成果）
- ✅ `paper.pdf`：5～6 頁論文報告
- ✅ `demo.mp4`：30 秒互動展示影片

---

## 🎤 報告規格（6/6）
- 發表時間：10 分鐘 + 3 分鐘 QA
- 必須對下一組進行提問

---

如需進一步協助架構程式目錄或撰寫 `streamlit` 頁面，歡迎指定我補上。
