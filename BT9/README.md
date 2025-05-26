# BTL: Dự báo giá cổ phiếu bằng Web App AI

Đây là bài tập lớn môn Khoa học Dữ liệu. Ứng dụng Streamlit để xây dựng một Web App cho phép người dùng tải dữ liệu cổ phiếu (.csv), phân tích và dự báo giá trong tương lai bằng mô hình LSTM.

## Hướng dẫn sử dụng:
1. Tải file `stock_forecast_app.py`
2. Cài thư viện bằng lệnh: `pip install -r requirements.txt`
3. Chạy app: `streamlit run stock_forecast_app.py`
4. Giao diện web sẽ xuất hiện, chọn file `.csv` để dự báo.

## Chức năng chính:
- Vẽ biểu đồ giá đóng cửa từ dữ liệu thật
- Dự báo giá `n` ngày tới bằng AI (chọn từ 3–30 ngày)
- Kết luận xu hướng tăng/giảm
- Hiển thị dữ liệu dự báo chi tiết
