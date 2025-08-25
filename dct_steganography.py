import cv2
import numpy as np
from PIL import Image
import time
from scipy.fftpack import dct, idct

class DCTSteganography:
    def __init__(self, block_size=8):
        self.name = "DCT (Discrete Cosine Transform)"
        self.block_size = block_size
        
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
    
    def embed(self, image_path, secret_text, alpha=0.1):
        """Giấu tin vào ảnh sử dụng DCT"""
        start_time = time.time()
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Không thể đọc ảnh")
        
        # Chuyển đổi sang YUV để xử lý kênh Y (luminance)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:, :, 0].astype(np.float64)
        
        # Chuyển đổi text thành binary
        binary_text = self.text_to_binary(secret_text)
        binary_length = format(len(binary_text), '032b')
        full_binary = binary_length + binary_text
        
        # Kiểm tra khả năng giấu tin
        total_blocks = (y_channel.shape[0] // self.block_size) * (y_channel.shape[1] // self.block_size)
        if len(full_binary) > total_blocks:
            raise ValueError("Ảnh quá nhỏ để giấu tin này")
        
        # Nhúng tin vào các block DCT
        bit_idx = 0
        for i in range(0, y_channel.shape[0] - self.block_size + 1, self.block_size):
            for j in range(0, y_channel.shape[1] - self.block_size + 1, self.block_size):
                if bit_idx >= len(full_binary):
                    break
                    
                # Lấy block
                block = y_channel[i:i+self.block_size, j:j+self.block_size]
                
                # Áp dụng DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Nhúng bit vào hệ số DCT (vị trí [1,1] - tần số thấp)
                if full_binary[bit_idx] == '1':
                    dct_block[1, 1] = abs(dct_block[1, 1]) + alpha * abs(dct_block[1, 1])
                else:
                    dct_block[1, 1] = abs(dct_block[1, 1]) - alpha * abs(dct_block[1, 1])
                
                # Áp dụng IDCT
                idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                
                # Cập nhật block
                y_channel[i:i+self.block_size, j:j+self.block_size] = idct_block
                
                bit_idx += 1
        
        # Cập nhật kênh Y
        img_yuv[:, :, 0] = np.clip(y_channel, 0, 255).astype(np.uint8)
        
        # Chuyển về BGR
        stego_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        processing_time = time.time() - start_time
        
        return stego_img, processing_time
    
    def extract(self, stego_image, alpha=0.1):
        """Trích xuất tin từ ảnh stego"""
        start_time = time.time()
        
        # Chuyển đổi sang YUV
        img_yuv = cv2.cvtColor(stego_image, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:, :, 0].astype(np.float64)
        
        # Trích xuất tin từ các block DCT
        extracted_bits = []
        bit_count = 0
        
        for i in range(0, y_channel.shape[0] - self.block_size + 1, self.block_size):
            for j in range(0, y_channel.shape[1] - self.block_size + 1, self.block_size):
                # Lấy block
                block = y_channel[i:i+self.block_size, j:j+self.block_size]
                
                # Áp dụng DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Trích xuất bit từ hệ số DCT [1,1]
                if dct_block[1, 1] > 0:
                    extracted_bits.append('1')
                else:
                    extracted_bits.append('0')
                
                bit_count += 1
                
                # Kiểm tra xem đã đủ bit chưa
                if bit_count >= 32:  # Độ dài text
                    binary_length = ''.join(extracted_bits[:32])
                    text_length = int(binary_length, 2)
                    
                    if bit_count >= 32 + text_length * 8:
                        break
        
        # Trích xuất text
        if len(extracted_bits) >= 32:
            binary_length = ''.join(extracted_bits[:32])
            text_length = int(binary_length, 2)
            
            if len(extracted_bits) >= 32 + text_length * 8:
                binary_text = ''.join(extracted_bits[32:32 + text_length * 8])
                extracted_text = self.binary_to_text(binary_text)
            else:
                extracted_text = "Không thể trích xuất đầy đủ text"
        else:
            extracted_text = "Không thể trích xuất text"
        
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
