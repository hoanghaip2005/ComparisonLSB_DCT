#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script test đơn giản để kiểm tra dự án
"""

import os
import sys

def test_imports():
    """Kiểm tra import các module"""
    print("🔍 Kiểm tra import các module...")
    
    try:
        from lsb_steganography import LSBSteganography
        print("✓ LSB module imported successfully")
    except ImportError as e:
        print(f"✗ LSB module import failed: {e}")
        return False
    
    try:
        from dct_steganography import DCTSteganography
        print("✓ DCT module imported successfully")
    except ImportError as e:
        print(f"✗ DCT module import failed: {e}")
        return False
    
    try:
        from performance_analyzer import PerformanceAnalyzer
        print("✓ Performance Analyzer module imported successfully")
    except ImportError as e:
        print(f"✗ Performance Analyzer module import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Kiểm tra chức năng cơ bản"""
    print("\n🔍 Kiểm tra chức năng cơ bản...")
    
    try:
        # Test LSB
        lsb = LSBSteganography()
        print("✓ LSB object created successfully")
        
        # Test DCT
        dct = DCTSteganography()
        print("✓ DCT object created successfully")
        
        # Test Analyzer
        analyzer = PerformanceAnalyzer()
        print("✓ Performance Analyzer object created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_image_creation():
    """Kiểm tra tạo ảnh mẫu"""
    print("\n🔍 Kiểm tra tạo ảnh mẫu...")
    
    try:
        import cv2
        import numpy as np
        
        # Tạo ảnh mẫu đơn giản
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 128  # Red channel
        img[:, :, 1] = 64   # Green channel
        img[:, :, 2] = 192  # Blue channel
        
        # Lưu ảnh
        cv2.imwrite("test_image.png", img)
        
        if os.path.exists("test_image.png"):
            print("✓ Test image created successfully")
            # Xóa ảnh test
            os.remove("test_image.png")
            return True
        else:
            print("✗ Test image creation failed")
            return False
            
    except Exception as e:
        print(f"✗ Image creation test failed: {e}")
        return False

def test_requirements():
    """Kiểm tra requirements"""
    print("\n🔍 Kiểm tra requirements...")
    
    required_packages = [
        'cv2',
        'numpy', 
        'PIL',
        'matplotlib',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Hàm chính"""
    print("🚀 Bắt đầu kiểm tra dự án...\n")
    
    tests = [
        ("Import modules", test_imports),
        ("Basic functionality", test_basic_functionality),
        ("Image creation", test_image_creation),
        ("Requirements", test_requirements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print("="*50)
    print(f"📊 KẾT QUẢ KIỂM TRA: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Tất cả tests đều thành công! Dự án sẵn sàng sử dụng.")
        print("\nĐể chạy demo:")
        print("1. python demo_script.py")
        print("2. python gui_demo.py")
    else:
        print("⚠️  Một số tests thất bại. Vui lòng kiểm tra và sửa lỗi.")
        print("\nĐể cài đặt dependencies:")
        print("pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
