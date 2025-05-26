import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------
# Cài đặt giao diện
# -------------------------
st.set_page_config(page_title="Dự báo Cổ Phiếu Bằng AI", layout="wide")
st.title("📊 DỰ BÁO GIÁ CỔ PHIẾU THEO THời GIAN")
st.caption("Phát triển bởi sinh viên - 2025")

# -------------------------
# Tải file .csv
# -------------------------
st.sidebar.header("📂 Nhập dữ liệu")
file = st.sidebar.file_uploader("Chọn file .csv", type=["csv"])
forecast_days = st.sidebar.slider("Số ngày dự báo", 3, 30, 7)

if file:
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Vẽ giá của Close
    st.subheader("1. Biểu đồ giá đóng cửa")
    st.line_chart(df['Close'])

    # Chuẩn hóa dữ liệu
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    # Mô hình LSTM + Dropout
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Dự báo
    last_seq = scaled[-60:]
    preds = []
    for _ in range(forecast_days):
        inp = last_seq.reshape(1, 60, 1)
        pred = model.predict(inp, verbose=0)
        preds.append(pred[0][0])
        last_seq = np.append(last_seq, pred)[-60:]

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    # Vẽ dự báo
    st.subheader(f"2. Dự báo giá {forecast_days} ngày tiếp theo")
    fig, ax = plt.subplots()
    ax.plot(forecast, marker='o', color='orange')
    ax.set_title(f"Giá dự báo trong {forecast_days} ngày")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá")
    st.pyplot(fig)

    # Kết luận xu hướng
    chg = forecast[-1][0] - data[-1][0]
    if chg > 0:
        st.success(f"Xu hướng: Tăng ({chg:.2f})")
    else:
        st.warning(f"Xu hướng: Giảm ({chg:.2f})")

    # Thông tin chi tiết
    with st.expander("Dữ liệu cuối kèm dự báo"):
        tail_df = pd.DataFrame({"Ngày": pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days), "Giá dự báo": forecast.flatten()})
        st.dataframe(tail_df)
else:
    st.info("Hãy tải lên dữ liệu .csv có cột Date và Close")
