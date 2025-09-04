import cv2
import numpy as np
from PIL import Image
import time
from scipy.fftpack import dct, idct
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor
import threading

class DCTSteganography:
    def __init__(self, block_size=8):
        self.name = "Enhanced DCT (Discrete Cosine Transform)"
        self.block_size = block_size
        self.thread_lock = threading.Lock()
        
    def get_block_complexity(self, block):
        """Tính độ phức tạp của block dựa vào gradient"""
        gradient_x = np.abs(np.diff(block, axis=1)).mean()
        gradient_y = np.abs(np.diff(block, axis=0)).mean()
        return (gradient_x + gradient_y) / 2
        
    def get_adaptive_alpha(self, block, base_alpha=0.1):
        """Tính hệ số alpha thích nghi dựa vào đặc điểm block"""
        complexity = self.get_block_complexity(block)
        # Điều chỉnh alpha dựa vào độ phức tạp
        # Block phức tạp hơn có thể chịu được alpha lớn hơn
        return base_alpha * (1 + complexity / 255)
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def apply_dct_to_blocks(channel, block_size):
        """Áp dụng DCT cho tất cả các block song song"""
        height, width = channel.shape
        result = np.zeros_like(channel, dtype=np.float64)
        
        for i in prange(0, height - block_size + 1, block_size):
            for j in range(0, width - block_size + 1, block_size):
                block = channel[i:i+block_size, j:j+block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                result[i:i+block_size, j:j+block_size] = dct_block
        
        return result
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def apply_idct_to_blocks(channel, block_size):
        """Áp dụng IDCT cho tất cả các block song song"""
        height, width = channel.shape
        result = np.zeros_like(channel, dtype=np.float64)
        
        for i in prange(0, height - block_size + 1, block_size):
            for j in range(0, width - block_size + 1, block_size):
                block = channel[i:i+block_size, j:j+block_size]
                idct_block = idct(idct(block.T, norm='ortho').T, norm='ortho')
                result[i:i+block_size, j:j+block_size] = idct_block
        
        return result
        
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
    
    def embed(self, image_path, secret_text, base_alpha=0.1):
        """Giấu tin vào ảnh sử dụng DCT cải tiến"""
        start_time = time.time()
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Không thể đọc ảnh")
            
        img = img.copy()  # Tạo bản sao để tránh thay đổi ảnh gốc
        
        # Chuyển đổi sang YUV để xử lý kênh Y (luminance)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:, :, 0].astype(np.float64)
        
        # Chuyển đổi text thành binary
        binary_text = self.text_to_binary(secret_text)
        binary_length = format(len(binary_text), '032b')
        print(f"[DEBUG] Message length (binary): {binary_length}")
        full_binary = binary_length + binary_text
        
        # Kiểm tra khả năng giấu tin
        total_blocks = (y_channel.shape[0] // self.block_size) * (y_channel.shape[1] // self.block_size)
        bits_needed = len(full_binary)
        print(f"[DEBUG] Total blocks available: {total_blocks}, Bits needed: {bits_needed}")
        if bits_needed > total_blocks * 2:  # Mỗi block có thể giấu 2 bit
            raise ValueError(f"Ảnh quá nhỏ để giấu tin này (cần {bits_needed} bits, có {total_blocks * 2} bits)")
        
        # Nhúng độ dài tin nhắn (32 bits đầu tiên)
        bit_idx = 0
        for i in range(0, min(self.block_size, y_channel.shape[0])):
            for j in range(0, y_channel.shape[1] - self.block_size + 1, self.block_size):
                if bit_idx >= 32:
                    break
                    
                block = y_channel[i:i+self.block_size, j:j+self.block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                alpha = self.get_adaptive_alpha(block, base_alpha)
                
                # Nhúng bit vào hệ số DCT đầu tiên
                if binary_length[bit_idx] == '1':
                    dct_block[1, 1] = abs(dct_block[1, 1]) + alpha * abs(dct_block[1, 1])
                else:
                    dct_block[1, 1] = abs(dct_block[1, 1]) - alpha * abs(dct_block[1, 1])
                    
                # Áp dụng IDCT và cập nhật block
                block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                y_channel[i:i+self.block_size, j:j+self.block_size] = block
                
                bit_idx += 1
                
            if bit_idx >= 32:
                break
                
        # Nhúng nội dung tin nhắn
        bit_idx = 0  # Reset lại để bắt đầu nhúng nội dung
        dct_positions = [(1,1), (2,2)]  # Sử dụng nhiều vị trí hệ số DCT
        
        # Nhúng nội dung tin nhắn vào các block còn lại
        for i in range(self.block_size, y_channel.shape[0] - self.block_size + 1, self.block_size):
            for j in range(0, y_channel.shape[1] - self.block_size + 1, self.block_size):
                if bit_idx >= len(binary_text):
                    break
                    
                block = y_channel[i:i+self.block_size, j:j+self.block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                alpha = self.get_adaptive_alpha(block, base_alpha)
                
                # Nhúng bit vào cả hai vị trí DCT
                for pos in dct_positions:
                    if bit_idx < len(binary_text):
                        if binary_text[bit_idx] == '1':
                            dct_block[pos] = abs(dct_block[pos]) + alpha * abs(dct_block[pos])
                        else:
                            dct_block[pos] = abs(dct_block[pos]) - alpha * abs(dct_block[pos])
                        bit_idx += 1
                
                # Áp dụng IDCT và cập nhật block
                block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                y_channel[i:i+self.block_size, j:j+self.block_size] = block
                
            if bit_idx >= len(binary_text):
                break
                
        # Chuyển về dạng uint8 và cập nhật kênh Y
        y_channel = np.clip(y_channel, 0, 255).astype(np.uint8)
        img_yuv[:, :, 0] = y_channel
        
        # Chuyển về BGR
        stego_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        processing_time = time.time() - start_time
        
        return stego_img, processing_time
        
        for i in range(0, y_channel.shape[0] - self.block_size + 1, self.block_size):
            for j in range(0, y_channel.shape[1] - self.block_size + 1, self.block_size):
                if bit_idx >= len(full_binary):
                    break
                    
                block = dct_blocks[i:i+self.block_size, j:j+self.block_size]
                alpha = self.get_adaptive_alpha(y_channel[i:i+self.block_size, j:j+self.block_size], base_alpha)
                
                # Nhúng bit vào nhiều hệ số DCT
                for pos in dct_positions:
                    if bit_idx >= len(full_binary):
                        break
                        
                    if full_binary[bit_idx] == '1':
                        block[pos] = abs(block[pos]) + alpha * abs(block[pos])
                    else:
                        block[pos] = abs(block[pos]) - alpha * abs(block[pos])
                    
                    bit_idx += 1
                
                dct_blocks[i:i+self.block_size, j:j+self.block_size] = block
        
        # Cập nhật kênh Y
        img_yuv[:, :, 0] = np.clip(y_channel, 0, 255).astype(np.uint8)
        
        # Chuyển về BGR
        stego_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        processing_time = time.time() - start_time
        
        return stego_img, processing_time
    
    def extract(self, stego_image, base_alpha=0.1):
        """Trích xuất tin từ ảnh stego sử dụng DCT cải tiến"""
        start_time = time.time()
        
        # Chuyển đổi sang YUV
        img_yuv = cv2.cvtColor(stego_image, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:, :, 0].astype(np.float64)
        
        # Áp dụng DCT cho từng block
        extracted_bits = []
        bit_count = 0
        dct_positions = [(1,1), (2,2)]  # Cùng vị trí với lúc nhúng
        
        # Trích xuất độ dài tin nhắn trước (32 bits đầu tiên)
        for i in range(0, min(self.block_size, y_channel.shape[0])):
            for j in range(0, y_channel.shape[1] - self.block_size + 1, self.block_size):
                if bit_count >= 32:
                    break
                    
                block = y_channel[i:i+self.block_size, j:j+self.block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                alpha = self.get_adaptive_alpha(block, base_alpha)
                
                # Trích xuất bit từ vị trí DCT đầu tiên
                coef = dct_block[1, 1]
                extracted_bits.append('1' if abs(coef) > (abs(coef) - alpha * abs(coef)) else '0')
                bit_count += 1
                
            if bit_count >= 32:
                break
                
        # Chuyển đổi 32 bit đầu thành độ dài tin nhắn
        binary_length = ''.join(extracted_bits[:32])
        try:
            message_length = int(binary_length, 2)
            print(f"[DEBUG] Extracted message length: {message_length} bits")
        except ValueError:
            print(f"[ERROR] Invalid binary length: {binary_length}")
            return '', time.time() - start_time
            
        # Reset lại cho phần trích xuất tin nhắn
        extracted_bits = []
        bit_count = 0
        total_bits_needed = message_length * 8
        
        # Trích xuất nội dung tin nhắn
        for i in range(0, y_channel.shape[0] - self.block_size + 1, self.block_size):
            for j in range(0, y_channel.shape[1] - self.block_size + 1, self.block_size):
                if bit_count >= total_bits_needed:
                    break
                    
                block = y_channel[i:i+self.block_size, j:j+self.block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                alpha = self.get_adaptive_alpha(block, base_alpha)
                
                # Trích xuất từ cả hai vị trí DCT
                for pos in dct_positions:
                    if bit_count >= total_bits_needed:
                        break
                    coef = dct_block[pos]
                    extracted_bits.append('1' if abs(coef) > (abs(coef) - alpha * abs(coef)) else '0')
                    bit_count += 1
                    
            if bit_count >= total_bits_needed:
                break
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
