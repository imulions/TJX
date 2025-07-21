import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore

# cd E:\PythonEnvironments   
# Set-ExecutionPolicy RemoteSigned
# .\tf-env\Scripts\Activate.ps1  

# 加载模型和缩放器
model = load_model('ann_model.h5', compile=False)
scaler_X = joblib.load('scaler_X.joblib')
scaler_y = joblib.load('scaler_y.joblib')

# 设置页面
st.set_page_config(
    page_title="亚精胺生产预测系统",
    page_icon="🧪",
    layout="wide"
)

# 标题和说明
st.title("🧪 亚精胺生产条件优化预测系统")
st.markdown("""
使用人工神经网络模型预测干重和胡萝卜素产量
""")

# 侧边栏输入
st.sidebar.header("培养条件参数设置")

# 定义输入控件
inputs = {}
inputs['light_intensity'] = st.sidebar.slider(
    '光照强度 (umol/m²/s)', 
    0.0, 200.0, 100.0, 0.1
)

inputs['temperature'] = st.sidebar.slider(
    '温度 (℃)', 
    20.0, 40.0, 30.0, 0.1
)

inputs['hormone'] = st.sidebar.slider(
    '激素浓度 (mol/L)', 
    0.0, 1.0, 0.5, 0.01
)

inputs['bicarbonate'] = st.sidebar.slider(
    '碳酸氢盐 (mmol/L)', 
    0.0, 200.0, 100.0, 1.0
)

inputs['nitrogen_source'] = st.sidebar.slider(
    '氮源 (mmol/L)', 
    0.0, 50.0, 25.0, 0.1
)

inputs['phosphorus_source'] = st.sidebar.slider(
    '磷源 (mmol/L)', 
    0.0, 10.0, 5.0, 0.1
)

inputs['cultivation_time'] = st.sidebar.slider(
    '培养时间 (天)', 
    0.0, 30.0, 15.0, 0.5
)

inputs['nacl'] = st.sidebar.slider(
    '氯化钠 (mmol/L)', 
    0.0, 200.0, 100.0, 1.0
)

# 转换为DataFrame
input_df = pd.DataFrame([inputs])

# 显示输入参数
st.subheader("当前培养条件")
st.dataframe(input_df)

# 预测按钮
if st.button('预测产量'):
    # 缩放输入
    input_scaled = scaler_X.transform(input_df)
    
    # 预测
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    # 确保非负
    prediction = np.maximum(prediction, 0)
    
    # 显示结果
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="干重 (Dry weight)",
            value=f"{prediction[0][0]:.4f} g/L",
            help="预测的干重产量"
        )
    
    with col2:
        st.metric(
            label="胡萝卜素产量 (Carotene yield)",
            value=f"{prediction[0][1]:.4f} mg/L",
            help="预测的胡萝卜素产量"
        )
    
    # 结果可视化
    st.subheader("产量预测结果")
    results_df = pd.DataFrame({
        '指标': ['干重', '胡萝卜素'],
        '产量': [prediction[0][0], prediction[0][1]],
        '单位': ['g/L', 'mg/L']
    })
    st.bar_chart(results_df.set_index('指标')['产量'])

# 添加说明
st.sidebar.markdown("""
**使用说明:**
1. 调整左侧滑块设置培养条件
2. 点击"预测产量"按钮获取结果
3. 结果将显示在右侧主区域
""")