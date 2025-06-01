# accident_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime

# 頁面配置
st.set_page_config(
    page_title="美國交通事故預測系統",
    page_icon="🚗",
    layout="wide"
)

# 標題
st.title("🚗 美國交通事故智能預測系統")
st.markdown("基於機器學習的實時事故風險評估")

# 載入模型和數據
@st.cache_resource
def load_resources():
    """載入模型和數據"""
    # 這裡載入您訓練好的模型
    # model = joblib.load('model_output/lightgbm_model.txt')
    # grid_stats = pd.read_csv('grid_statistics.csv')
    
    # 模擬數據（實際使用時替換）
    grid_stats = pd.DataFrame({
        'lat': np.random.uniform(25, 48, 100),
        'lng': np.random.uniform(-125, -70, 100),
        'accident_count': np.random.randint(10, 1000, 100),
        'avg_severity': np.random.uniform(1, 4, 100),
        'risk_score': np.random.uniform(0, 100, 100)
    })
    
    return None, grid_stats

model, grid_stats = load_resources()

# 側邊欄
st.sidebar.header("🔧 預測參數設置")

# 時間選擇
col1, col2 = st.sidebar.columns(2)
with col1:
    selected_date = st.date_input("選擇日期", datetime.now())
with col2:
    selected_hour = st.slider("選擇時間", 0, 23, 12, format="%d:00")

# 天氣條件
weather_condition = st.sidebar.selectbox(
    "天氣狀況",
    ["☀️ 晴天", "☁️ 多雲", "🌧️ 下雨", "🌨️ 下雪", "🌫️ 霧"]
)

# 地區選擇
selected_state = st.sidebar.selectbox(
    "選擇州",
    ["All States", "California", "Texas", "Florida", "New York", "Illinois"]
)

# 主要內容區
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 事故風險熱力圖")
    
    # 創建地圖
    fig_map = go.Figure()
    
    # 添加熱力圖數據
    fig_map.add_trace(go.Scattermapbox(
        lat=grid_stats['lat'],
        lon=grid_stats['lng'],
        mode='markers',
        marker=dict(
            size=np.log1p(grid_stats['accident_count']) * 5,
            color=grid_stats['risk_score'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="風險等級"),
            opacity=0.7
        ),
        text=[f"風險分數: {score:.1f}" for score in grid_stats['risk_score']],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=39.8283, lon=-98.5795),
            zoom=3
        ),
        showlegend=False,
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.subheader("📊 實時統計")
    
    # 風險指標
    avg_risk = grid_stats['risk_score'].mean()
    risk_level = "高" if avg_risk > 70 else "中" if avg_risk > 40 else "低"
    risk_color = "🔴" if avg_risk > 70 else "🟡" if avg_risk > 40 else "🟢"
    
    st.metric(
        label="當前風險等級",
        value=f"{risk_color} {risk_level}",
        delta=f"{avg_risk:.1f}%"
    )
    
    # 預測統計
    col1_2, col2_2 = st.columns(2)
    with col1_2:
        st.metric("預測事故數", f"{int(grid_stats['accident_count'].sum()):,}")
    with col2_2:
        st.metric("平均嚴重度", f"{grid_stats['avg_severity'].mean():.2f}")
    
    # 嚴重度分布圖
    severity_dist = pd.DataFrame({
        'Severity': ['輕微', '一般', '嚴重', '致命'],
        'Count': [40, 35, 20, 5]
    })
    
    fig_pie = px.pie(
        severity_dist, 
        values='Count', 
        names='Severity',
        color_discrete_map={
            '輕微': '#90EE90',
            '一般': '#FFD700',
            '嚴重': '#FF6347',
            '致命': '#8B0000'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(showlegend=False, height=300)
    
    st.plotly_chart(fig_pie, use_container_width=True)

# 詳細分析區
st.subheader("📈 時間趨勢分析")

# 創建24小時趨勢圖
hours = list(range(24))
hourly_risk = np.random.randint(20, 80, 24)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=hours,
    y=hourly_risk,
    mode='lines+markers',
    line=dict(color='red', width=2),
    marker=dict(size=8),
    fill='tozeroy',
    fillcolor='rgba(255,0,0,0.1)'
))

fig_trend.update_layout(
    xaxis_title="時間 (小時)",
    yaxis_title="風險指數",
    hovermode='x unified',
    height=300
)

st.plotly_chart(fig_trend, use_container_width=True)

# 高風險區域表格
st.subheader("⚠️ 高風險區域警示")

high_risk_df = grid_stats.nlargest(10, 'risk_score')[['lat', 'lng', 'risk_score', 'accident_count']]
high_risk_df.columns = ['緯度', '經度', '風險分數', '歷史事故數']
high_risk_df.index = range(1, len(high_risk_df) + 1)

st.dataframe(
    high_risk_df.style.background_gradient(subset=['風險分數'], cmap='Reds'),
    use_container_width=True
)

# 底部信息
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>數據更新時間：{} | 模型版本：LightGBM v1.0</p>
        <p>⚠️ 此預測僅供參考，請謹慎駕駛</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)