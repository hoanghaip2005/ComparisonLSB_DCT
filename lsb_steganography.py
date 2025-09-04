import cv2
import numpy as np
from PIL import Image
import time

class LSBSteganography:
    def __init__(self):
        self.name = "Enhanced LSB (Adaptive Least Significant Bit)"
        self.key = 42  # Khóa cho hoán vị
        
    def text_to_binary(self, text):
        """Chuyển đổi text thành binary"""
        binary = ''.join(format(ord(char), '08b') for char in text)
        return binary
    
    def binary_to_text(self, binary):
        """Chuyển đổi binary thành text"""
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                text += chr(int(byte, 2))
        return text
        
    def calculate_complexity_map(self, img):
        """Tính toán ma trận độ phức tạp sử dụng vectorization"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Tính gradient theo chiều x và y sử dụng Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Tính độ lớn gradient
        complexity = np.sqrt(grad_x**2 + grad_y**2)
        
        # Chuẩn hóa về khoảng [1,3]
        complexity = 1 + 2 * (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-10)
        
        return complexity.astype(np.int32)
        
    def generate_permutation_map(self, height, width):
        """Tạo ma trận hoán vị cho các pixel"""
        np.random.seed(self.key)
        total_pixels = height * width
        indices = np.random.permutation(total_pixels)
        return indices // width, indices % width
    
    def embed(self, image_path, secret_text):
        """Giấu tin vào ảnh sử dụng LSB thích nghi"""
        start_time = time.time()
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Không thể đọc ảnh")
            
        # Tạo bản sao để không ảnh hưởng đến ảnh gốc
        img = img.copy()
        
        # Chuyển đổi text thành binary
        text_length = len(secret_text)  # Độ dài văn bản gốc
        binary_text = self.text_to_binary(secret_text)
        binary_length = format(text_length, '032b')  # Lưu độ dài văn bản gốc
        print(f"[DEBUG] Embedding - Text length: {text_length}")
        print(f"[DEBUG] Embedding - Length in binary: {binary_length}")
        
        # Kiểm tra khả năng giấu tin
        total_pixels = img.shape[0] * img.shape[1] * 3
        total_bits_needed = len(binary_text) + 32
        if total_bits_needed > total_pixels:
            raise ValueError(f"Ảnh quá nhỏ để giấu tin này (cần {total_bits_needed} bits, có {total_pixels} bits)")
            
        # Tạo ma trận hoán vị
        perm_rows, perm_cols = self.generate_permutation_map(img.shape[0], img.shape[1])
        
        # Tính ma trận độ phức tạp
        complexity_map = self.calculate_complexity_map(img)
            
        # Nhúng độ dài text vào 32 bit đầu tiên sử dụng 1-LSB
        header_rows = perm_rows[:32]
        header_cols = perm_cols[:32]
        for i, (row, col) in enumerate(zip(header_rows, header_cols)):
            channel = i % 3
            img[row, col, channel] = (img[row, col, channel] & 0xFE) | int(binary_length[i])
        
        # Chuẩn bị dữ liệu để nhúng
        binary_data = np.array([int(b) for b in binary_text])
        data_length = len(binary_data)
        
        # Nhúng text sử dụng số bit thích nghi
        data_idx = 0
        pixels_needed = (data_length + 7) // 8  # Số pixel cần thiết, làm tròn lên
        
        for idx in range(32, min(pixels_needed + 32, len(perm_rows))):
            if data_idx >= data_length:
                break
                
            row, col = perm_rows[idx], perm_cols[idx]
            num_bits = complexity_map[row, col]
            
            for channel in range(3):
                if data_idx >= data_length:
                    break
                    
                # Số bit có thể nhúng vào pixel này
                bits_to_embed = min(num_bits, data_length - data_idx)
                if bits_to_embed <= 0:
                    break
                    
                # Tạo mặt nạ bit
                mask = (0xFF << bits_to_embed) & 0xFF
                
                # Lấy bits_to_embed bits từ dữ liệu
                bits = 0
                for k in range(bits_to_embed):
                    if data_idx < data_length:
                        bits |= binary_data[data_idx] << k
                        data_idx += 1
                
                # Nhúng bits vào pixel
                img[row, col, channel] = (img[row, col, channel] & mask) | bits
        
        processing_time = time.time() - start_time
        
        return img, processing_time
    
    def extract(self, stego_image):
        """Trích xuất tin từ ảnh stego"""
        start_time = time.time()
        
        # Tạo ma trận hoán vị
        perm_rows, perm_cols = self.generate_permutation_map(stego_image.shape[0], stego_image.shape[1])
        
        # Tính ma trận độ phức tạp
        complexity_map = self.calculate_complexity_map(stego_image)
        
        # Trích xuất độ dài text từ 32 bit đầu tiên (header)
        binary_length = ''
        for i in range(32):
            row = perm_rows[i]  # perm_rows đã là 1D array
            col = perm_cols[i]  # perm_cols đã là 1D array
            channel = i % 3
            binary_length += str(stego_image[row, col, channel] & 1)
        
        # Đây là độ dài của văn bản gốc, không phải độ dài chuỗi nhị phân
        text_length = int(binary_length, 2)
        print(f"[DEBUG] Text length extracted: {text_length}")
        print(f"[DEBUG] Binary length string: {binary_length}")
        
        # Trích xuất text (cần số bit = text_length * 8 vì mỗi ký tự = 8 bit)
        binary_text = []
        data_idx = 0
        total_bits_needed = text_length * 8
        
        # Bắt đầu từ vị trí sau header (32 bits)
        for idx in range(32, len(perm_rows)):
            if data_idx >= total_bits_needed:
                break
                
            row, col = perm_rows[idx], perm_cols[idx]
            num_bits = min(complexity_map[row, col], 3)  # Giới hạn tối đa 3 bits
            
            for channel in range(3):
                if data_idx >= total_bits_needed:
                    break
                
                # Trích xuất các bit từ pixel
                pixel_value = stego_image[row, col, channel]
                bits = pixel_value & ((1 << num_bits) - 1)  # Lấy num_bits bit thấp nhất
                
                # Thêm từng bit vào kết quả
                for k in range(num_bits):
                    if data_idx < total_bits_needed:
                        bit = (bits >> k) & 1
                        binary_text.append('1' if bit else '0')
                        data_idx += 1
        
        # Chuyển đổi list các bit thành chuỗi nhị phân
        binary_string = ''.join(binary_text)
        extracted_text = self.binary_to_text(binary_string)
        processing_time = time.time() - start_time
        
        return extracted_text, processing_time
    
    def calculate_metrics(self, original_img, stego_img):
        """Tính toán PSNR và SSIM"""
        # Chuyển đổi sang grayscale nếu cần
        if len(original_img.shape) == 3:
            original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_img
            
        if len(stego_img.shape) == 3:
            stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
        else:
            stego_gray = stego_img
        
        # Tính PSNR
        mse = np.mean((original_gray.astype(float) - stego_gray.astype(float)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Tính SSIM đơn giản
        mu_x = np.mean(original_gray)
        mu_y = np.mean(stego_gray)
        sigma_x = np.std(original_gray)
        sigma_y = np.std(stego_gray)
        sigma_xy = np.mean((original_gray - mu_x) * (stego_gray - mu_y))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'mse': mse
        }
