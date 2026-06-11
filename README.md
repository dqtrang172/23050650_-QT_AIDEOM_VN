# VN AIDEOM-VN — Web App 12 bài mô hình ra quyết định

Web app Streamlit cho bộ bài tập **Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI**.
Giao diện được thiết kế theo phong cách dashboard tối, hiện đại, có 12 menu tương ứng 12 bài và mỗi bài đều có khối **Tác nhân phân tích kết quả**.

## 1. Cách chạy nhanh

### Windows
Nhấp đôi file:

```bat
run_app.bat
```

### Mac/Linux
```bash
bash run_app.sh
```

### Chạy thủ công
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## 2. Cấu trúc thư mục

```text
MSV_HoTen_AIDEOM_VN/
├── app.py
├── requirements.txt
├── run_app.bat
├── run_app.sh
├── data/
    ├── vietnam_macro_2020_2025.csv
    ├── vietnam_sectors_2024.csv
    └── vietnam_regions_2024.csv



## 3. Các chức năng chính

- Trang chủ có KPI kinh tế Việt Nam và bản đồ 12 bài theo 4 cấp độ.
- Bài 1: Cobb-Douglas mở rộng, TFP, MAPE, growth accounting, dự báo 2030.
- Bài 2: LP phân bổ ngân sách số 4 hạng mục, shadow price gần đúng, phân tích độ nhạy.
- Bài 3: Chỉ số ưu tiên 10 ngành, min-max, sensitivity theo trọng số AI.
- Bài 4: LP phân bổ ngân sách theo vùng - hạng mục, ràng buộc công bằng vùng.
- Bài 5: MIP lựa chọn 15 dự án chuyển đổi số bằng duyệt tổ hợp nhị phân.
- Bài 6: TOPSIS 6 vùng, trọng số chuyên gia và Entropy.
- Bài 7: Mô phỏng Pareto đa mục tiêu và chọn nghiệm thỏa hiệp.
- Bài 8: Tối ưu động/mô phỏng liên thời gian 2026-2035.
- Bài 9: Mô hình NetJob lao động dưới tác động AI.
- Bài 10: Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định.
- Bài 11: Q-learning tabular cho chính sách thích nghi.
- Bài 12: Dashboard tích hợp AIDEOM-VN và so sánh 5 kịch bản.
