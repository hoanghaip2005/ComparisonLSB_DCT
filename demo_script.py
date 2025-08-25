#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script để so sánh hiệu năng LSB vs DCT Steganography
Tạo ảnh mẫu và thực hiện phân tích so sánh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from lsb_steganography import LSBSteganography
from dct_steganography import DCTSteganography
from performance_analyzer import PerformanceAnalyzer
import os

def create_sample_image(size=(512, 512), save_path="sample_image.png"):
    """Tạo ảnh mẫu để test"""
    print("Đang tạo ảnh mẫu...")
    
    # Tạo ảnh gradient với nhiều màu sắc
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Tạo gradient màu
    for i in range(size[0]):
        for j in range(size[1]):
            # Red channel - gradient từ trái sang phải
            img[i, j, 0] = int(255 * j / size[1])
            # Green channel - gradient từ trên xuống dưới
            img[i, j, 1] = int(255 * i / size[0])
            # Blue channel - pattern
            img[i, j, 2] = int(128 + 127 * np.sin(i/50) * np.cos(j/50))
    
    # Thêm một số hình dạng để tạo texture
    # Hình tròn ở giữa
    center_x, center_y = size[0]//2, size[1]//2
    radius = 100
    cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)
    
    # Hình chữ nhật ở góc
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
    
    # Thêm text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'SAMPLE', (200, 100), font, 2, (0, 0, 255), 3)
    
    # Lưu ảnh
    cv2.imwrite(save_path, img)
    print(f"Đã tạo ảnh mẫu: {save_path}")
    
    return save_path

def run_demo_comparison():
    """Chạy demo so sánh LSB vs DCT"""
    print("=== DEMO SO SÁNH LSB vs DCT STEGANOGRAPHY ===\n")
    
    # Tạo ảnh mẫu nếu chưa có
    sample_image_path = "sample_image.png"
    if not os.path.exists(sample_image_path):
        sample_image_path = create_sample_image()
    
    # Text cần giấu
    secret_text = "Đây là tin mật được giấu bằng steganography! LSB vs DCT comparison demo."
    print(f"Text cần giấu: {secret_text}")
    print(f"Độ dài: {len(secret_text)} ký tự")
    print()
    
    # Khởi tạo analyzer
    analyzer = PerformanceAnalyzer()
    
    # Thực hiện so sánh
    print("Đang thực hiện phân tích so sánh...")
    lsb_results, dct_results = analyzer.compare_methods(sample_image_path, secret_text)
    
    if not lsb_results or not dct_results:
        print("Có lỗi trong quá trình phân tích!")
        return
    
    # Hiển thị kết quả
    print("\n" + "="*60)
    print("KẾT QUẢ SO SÁNH")
    print("="*60)
    
    # Báo cáo chi tiết
    report = analyzer.generate_comparison_report(lsb_results, dct_results)
    print(report)
    
    # Lưu kết quả
    print("\nĐang lưu kết quả...")
    analyzer.save_results(lsb_results, dct_results, "demo_results")
    
    # Tạo so sánh trực quan
    print("Đang tạo biểu đồ so sánh...")
    original_img = cv2.imread(sample_image_path)
    analyzer.create_visual_comparison(original_img, lsb_results, dct_results, "demo_comparison.png")
    
    print("\nDemo hoàn tất! Kiểm tra các file kết quả:")
    print("- demo_results_lsb.png: Ảnh sau khi giấu tin bằng LSB")
    print("- demo_results_dct.png: Ảnh sau khi giấu tin bằng DCT")
    print("- demo_comparison.png: Biểu đồ so sánh trực quan")
    print("- demo_results_lsb.png và demo_results_dct.png: Ảnh kết quả")

def run_individual_tests():
    """Chạy test riêng lẻ cho từng phương pháp"""
    print("\n=== TEST RIÊNG LẺ ===\n")
    
    # Tạo ảnh mẫu
    sample_image_path = "sample_image.png"
    if not os.path.exists(sample_image_path):
        sample_image_path = create_sample_image()
    
    # Test LSB
    print("Testing LSB Steganography...")
    lsb = LSBSteganography()
    secret_text = "LSB test message"
    
    try:
        stego_img, embed_time = lsb.embed(sample_image_path, secret_text)
        extracted_text, extract_time = lsb.extract(stego_img)
        
        print(f"✓ LSB - Nhúng: {embed_time:.4f}s, Trích xuất: {extract_time:.4f}s")
        print(f"  Text gốc: {secret_text}")
        print(f"  Text trích xuất: {extracted_text}")
        print(f"  Độ chính xác: {'✓' if extracted_text == secret_text else '✗'}")
        
        # Lưu ảnh test
        cv2.imwrite("lsb_test_result.png", stego_img)
        
    except Exception as e:
        print(f"✗ LSB error: {str(e)}")
    
    # Test DCT
    print("\nTesting DCT Steganography...")
    dct = DCTSteganography()
    
    try:
        stego_img, embed_time = dct.embed(sample_image_path, secret_text)
        extracted_text, extract_time = dct.extract(stego_img)
        
        print(f"✓ DCT - Nhúng: {embed_time:.4f}s, Trích xuất: {extract_time:.4f}s")
        print(f"  Text gốc: {secret_text}")
        print(f"  Text trích xuất: {extracted_text}")
        print(f"  Độ chính xác: {'✓' if extracted_text == secret_text else '✗'}")
        
        # Lưu ảnh test
        cv2.imwrite("dct_test_result.png", stego_img)
        
    except Exception as e:
        print(f"✗ DCT error: {str(e)}")

def main():
    """Hàm chính"""
    print("Chọn chế độ demo:")
    print("1. So sánh đầy đủ LSB vs DCT")
    print("2. Test riêng lẻ từng phương pháp")
    print("3. Cả hai")
    
    choice = input("\nNhập lựa chọn (1/2/3): ").strip()
    
    if choice == "1":
        run_demo_comparison()
    elif choice == "2":
        run_individual_tests()
    elif choice == "3":
        run_individual_tests()
        run_demo_comparison()
    else:
        print("Lựa chọn không hợp lệ, chạy demo đầy đủ...")
        run_demo_comparison()

if __name__ == "__main__":
    main()
