# Phân tích so sánh hiệu năng LSB vs DCT Steganography

Dự án này cung cấp một công cụ hoàn chỉnh để so sánh hiệu năng giữa hai phương pháp steganography phổ biến: **LSB (Least Significant Bit)** và **DCT (Discrete Cosine Transform)**.

## 🎯 Mục tiêu dự án

- So sánh hiệu năng giữa LSB và DCT steganography
- Đánh giá chất lượng ảnh sau khi giấu tin (PSNR, SSIM)
- Đo lường thời gian xử lý của từng phương pháp
- Cung cấp giao diện demo trực quan với ảnh minh họa
- Tạo báo cáo chi tiết về hiệu năng

## 🚀 Tính năng chính

### 1. **LSB Steganography**
- Giấu tin vào bit ít quan trọng nhất của mỗi pixel
- Xử lý nhanh, đơn giản
- Phù hợp với ảnh có nhiều chi tiết

### 2. **DCT Steganography**
- Giấu tin vào miền tần số (DCT coefficients)
- Khả năng chống nhiễu tốt hơn
- Xử lý theo block 8x8 pixel

### 3. **Phân tích hiệu năng**
- **PSNR (Peak Signal-to-Noise Ratio)**: Đo lường chất lượng ảnh
- **SSIM (Structural Similarity Index)**: Đánh giá độ tương đồng cấu trúc
- **Thời gian xử lý**: So sánh tốc độ nhúng và trích xuất
- **Độ chính xác**: Kiểm tra khả năng khôi phục tin mật

### 4. **Giao diện demo**
- GUI thân thiện với người dùng
- Hiển thị ảnh gốc và ảnh sau khi giấu tin
- So sánh trực quan giữa hai phương pháp
- Lưu và xuất kết quả

## 📁 Cấu trúc dự án

```
ComparisonLSB_DCT/
├── lsb_steganography.py      # Triển khai LSB steganography
├── dct_steganography.py      # Triển khai DCT steganography
├── performance_analyzer.py    # Phân tích và so sánh hiệu năng
├── gui_demo.py               # Giao diện người dùng
├── demo_script.py            # Script demo tự động
├── requirements.txt           # Thư viện cần thiết
└── README.md                 # Hướng dẫn sử dụng
```

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.7+
- Windows/Linux/macOS

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Thư viện chính
- `opencv-python`: Xử lý ảnh
- `numpy`: Tính toán số học
- `Pillow`: Xử lý ảnh nâng cao
- `matplotlib`: Vẽ biểu đồ
- `scikit-image`: Xử lý ảnh khoa học

## 🎮 Cách sử dụng

### 1. **Chạy GUI Demo (Khuyến nghị)**
```bash
python gui_demo.py
```

**Hướng dẫn sử dụng GUI:**
1. Nhấn "Chọn Ảnh" để chọn ảnh cần xử lý
2. Nhập tin cần giấu vào ô text
3. Nhấn "So sánh LSB vs DCT" để thực hiện phân tích
4. Xem kết quả trong phần "Kết quả phân tích"
5. Nhấn "Xem ảnh kết quả" để so sánh trực quan
6. Nhấn "Lưu kết quả" để lưu các ảnh và báo cáo

### 2. **Chạy Demo Script**
```bash
python demo_script.py
```

Script này sẽ:
- Tạo ảnh mẫu tự động
- Thực hiện so sánh LSB vs DCT
- Hiển thị kết quả chi tiết
- Tạo biểu đồ so sánh
- Lưu tất cả kết quả

### 3. **Sử dụng trực tiếp các class**
```python
from lsb_steganography import LSBSteganography
from dct_steganography import DCTSteganography
from performance_analyzer import PerformanceAnalyzer

# Khởi tạo
lsb = LSBSteganography()
dct = DCTSteganography()
analyzer = PerformanceAnalyzer()

# So sánh
lsb_results, dct_results = analyzer.compare_methods("image.jpg", "Secret text")
```

## 📊 Kết quả đầu ra

### 1. **Ảnh kết quả**
- `*_lsb.png`: Ảnh sau khi giấu tin bằng LSB
- `*_dct.png`: Ảnh sau khi giấu tin bằng DCT

### 2. **Báo cáo so sánh**
- Thời gian xử lý (nhúng + trích xuất)
- Chỉ số PSNR và SSIM
- Độ chính xác của việc khôi phục tin
- Kết luận và khuyến nghị

### 3. **Biểu đồ so sánh**
- So sánh trực quan ảnh gốc vs stego
- Biểu đồ cột so sánh PSNR, SSIM, thời gian
- Lưu dưới dạng PNG chất lượng cao

## 🔍 Giải thích kỹ thuật

### **LSB (Least Significant Bit)**
- **Nguyên lý**: Thay đổi bit ít quan trọng nhất của mỗi pixel
- **Ưu điểm**: Đơn giản, nhanh, ít thay đổi ảnh
- **Nhược điểm**: Dễ bị phát hiện, không chống nhiễu

### **DCT (Discrete Cosine Transform)**
- **Nguyên lý**: Chuyển ảnh sang miền tần số, giấu tin vào DCT coefficients
- **Ưu điểm**: Chống nhiễu tốt, khó phát hiện
- **Nhược điểm**: Phức tạp hơn, có thể làm mất một số chi tiết

### **Chỉ số đánh giá**
- **PSNR**: 
  - > 40 dB: Khó phân biệt bằng mắt thường
  - 30-40 dB: Chất lượng tốt
  - < 30 dB: Chất lượng thấp
  
- **SSIM**: 
  - 1.0: Hoàn toàn giống nhau
  - 0.9-1.0: Rất giống
  - 0.8-0.9: Giống
  - < 0.8: Khác biệt

## 📈 Kết quả mẫu

Dự án sẽ tạo ra các kết quả như:

```
=== BÁO CÁO SO SÁNH HIỆU NĂNG LSB vs DCT ===

1. THỜI GIAN XỬ LÝ:
   LSB - Nhúng: 0.0234s, Trích xuất: 0.0156s
   DCT - Nhúng: 0.1567s, Trích xuất: 0.1234s
   Tổng thời gian LSB: 0.0390s
   Tổng thời gian DCT: 0.2801s

2. CHẤT LƯỢNG ẢNH:
   LSB - PSNR: 45.67 dB, SSIM: 0.9876
   DCT - PSNR: 52.34 dB, SSIM: 0.9945

3. ĐỘ CHÍNH XÁC:
   LSB: ✓
   DCT: ✓

4. KẾT LUẬN:
   - LSB nhanh hơn DCT
   - DCT có chất lượng ảnh tốt hơn (PSNR cao hơn)
   - DCT có độ tương đồng cấu trúc tốt hơn (SSIM cao hơn)
```

## 🎨 Tùy chỉnh

### **Thay đổi tham số DCT**
```python
# Trong dct_steganography.py
dct = DCTSteganography(block_size=16)  # Thay đổi kích thước block
stego_img, time = dct.embed(image_path, text, alpha=0.05)  # Thay đổi alpha
```

### **Thêm phương pháp mới**
1. Tạo class mới kế thừa từ base class
2. Implement các method: `embed()`, `extract()`, `calculate_metrics()`
3. Thêm vào `PerformanceAnalyzer`

## 🐛 Xử lý lỗi thường gặp

### **Lỗi "Ảnh quá nhỏ"**
- Sử dụng ảnh có kích thước lớn hơn (khuyến nghị > 256x256)
- Giảm độ dài text cần giấu

### **Lỗi thư viện**
- Cài đặt lại: `pip install --upgrade -r requirements.txt`
- Kiểm tra phiên bản Python (yêu cầu 3.7+)

### **Lỗi hiển thị ảnh**
- Kiểm tra định dạng ảnh (hỗ trợ: PNG, JPG, JPEG, BMP, TIFF)
- Đảm bảo ảnh không bị hỏng

## 📚 Tài liệu tham khảo

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Steganography Techniques](https://en.wikipedia.org/wiki/Steganography)
- [DCT Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform)

## 👥 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork dự án
2. Tạo branch mới cho tính năng
3. Commit thay đổi
4. Push lên branch
5. Tạo Pull Request

## 📄 Giấy phép

Dự án này được phát hành dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo issue trên GitHub hoặc liên hệ trực tiếp.

---

**Lưu ý**: Dự án này chỉ dành cho mục đích giáo dục và nghiên cứu. Vui lòng tuân thủ luật pháp địa phương khi sử dụng.
