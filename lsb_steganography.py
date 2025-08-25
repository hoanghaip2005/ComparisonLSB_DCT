import cv2
import numpy as np
from PIL import Image
import time

class LSBSteganography:
    def __init__(self):
        self.name = "LSB (Least Significant Bit)"
        
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
    
    def embed(self, image_path, secret_text):
        """Giấu tin vào ảnh sử dụng LSB"""
        start_time = time.time()
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Không thể đọc ảnh")
        
        # Chuyển đổi text thành binary
        binary_text = self.text_to_binary(secret_text)
        binary_length = format(len(binary_text), '032b')
        
        # Kiểm tra khả năng giấu tin
        total_pixels = img.shape[0] * img.shape[1] * 3
        if len(binary_text) + 32 > total_pixels:
            raise ValueError("Ảnh quá nhỏ để giấu tin này")
        
        # Nhúng độ dài text vào 32 bit đầu tiên
        idx = 0
        for i in range(32):
            img[0, i//3, i%3] = (img[0, i//3, i%3] & 0xFE) | int(binary_length[i])
        
        # Nhúng text vào các pixel tiếp theo
        pixel_idx = 0
        for char in binary_text:
            for channel in range(3):
                if pixel_idx < total_pixels:
                    row = (pixel_idx // 3) // img.shape[1]
                    col = (pixel_idx // 3) % img.shape[1]
                    channel_idx = pixel_idx % 3
                    
                    if row < img.shape[0] and col < img.shape[1]:
                        img[row, col, channel_idx] = (img[row, col, channel_idx] & 0xFE) | int(char)
                    pixel_idx += 1
        
        processing_time = time.time() - start_time
        
        return img, processing_time
    
    def extract(self, stego_image):
        """Trích xuất tin từ ảnh stego"""
        start_time = time.time()
        
        # Trích xuất độ dài text từ 32 bit đầu tiên
        binary_length = ''
        for i in range(32):
            binary_length += str(stego_image[0, i//3, i%3] & 1)
        
        text_length = int(binary_length, 2)
        
        # Trích xuất text
        binary_text = ''
        pixel_idx = 0
        total_pixels = stego_image.shape[0] * stego_image.shape[1] * 3
        for i in range(text_length):
            for channel in range(3):
                if pixel_idx < total_pixels:
                    row = (pixel_idx // 3) // stego_image.shape[1]
                    col = (pixel_idx // 3) % stego_image.shape[1]
                    channel_idx = pixel_idx % 3
                    
                    if row < stego_image.shape[0] and col < stego_image.shape[1]:
                        binary_text += str(stego_image[row, col, channel_idx] & 1)
                    pixel_idx += 1
        
        extracted_text = self.binary_to_text(binary_text)
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
