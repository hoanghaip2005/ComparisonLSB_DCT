#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Demo đơn giản để test dự án LSB vs DCT Steganography
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from lsb_steganography import LSBSteganography
from dct_steganography import DCTSteganography
from performance_analyzer import PerformanceAnalyzer

class SimpleSteganographyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LSB vs DCT Steganography - Demo")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Khởi tạo các đối tượng
        self.lsb = LSBSteganography()
        self.dct = DCTSteganography()
        self.analyzer = PerformanceAnalyzer()
        
        # Biến lưu trữ
        self.original_image_path = None
        self.original_image = None
        self.secret_text = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        """Thiết lập giao diện đơn giản"""
        # Title
        title_label = tk.Label(self.root, text="LSB vs DCT STEGANOGRAPHY DEMO", 
                              font=('Arial', 16, 'bold'), 
                              bg='#2c3e50', fg='white')
        title_label.pack(fill='x', pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Image selection
        img_frame = tk.Frame(main_frame, bg='#f0f0f0')
        img_frame.pack(fill='x', pady=10)
        
        tk.Button(img_frame, text="Chọn Ảnh", command=self.select_image,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=(0, 10))
        
        self.image_path_label = tk.Label(img_frame, text="Chưa chọn ảnh", 
                                        bg='#f0f0f0', font=('Arial', 9))
        self.image_path_label.pack(side='left')
        
        # Text input
        text_frame = tk.Frame(main_frame, bg='#f0f0f0')
        text_frame.pack(fill='x', pady=10)
        
        tk.Label(text_frame, text="Nhập tin cần giấu:", font=('Arial', 10, 'bold'),
                bg='#f0f0f0').pack(anchor='w')
        
        self.text_input = scrolledtext.ScrolledText(text_frame, height=3, width=50,
                                                   font=('Arial', 10))
        self.text_input.pack(fill='x', pady=5)
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=20)
        
        tk.Button(button_frame, text="So sánh LSB vs DCT", command=self.compare_methods,
                 bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                 height=2, width=20).pack(pady=10)
        
        # Results display
        results_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        results_frame.pack(fill='both', expand=True, pady=10)
        
        tk.Label(results_frame, text="KẾT QUẢ PHÂN TÍCH", font=('Arial', 12, 'bold'),
                bg='white').pack(pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=70,
                                                     font=('Consolas', 10))
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Sẵn sàng", 
                                     bg='#2c3e50', fg='white', font=('Arial', 9))
        self.status_label.pack(fill='x', side='bottom')
        
    def select_image(self):
        """Chọn ảnh"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            self.original_image_path = file_path
            self.image_path_label.config(text=f"Đã chọn: {os.path.basename(file_path)}")
            self.status_label.config(text="Đã chọn ảnh")
            
            # Load ảnh
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is not None:
                    self.status_label.config(text=f"Đã tải ảnh: {self.original_image.shape}")
                else:
                    messagebox.showerror("Lỗi", "Không thể tải ảnh!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi tải ảnh: {str(e)}")
    
    def compare_methods(self):
        """So sánh LSB vs DCT"""
        if self.original_image_path is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        
        self.secret_text = self.text_input.get("1.0", tk.END).strip()
        if not self.secret_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập tin cần giấu!")
            return
        
        try:
            self.status_label.config(text="Đang phân tích...")
            self.root.update()
            
            # Thực hiện so sánh
            lsb_results, dct_results = self.analyzer.compare_methods(
                self.original_image_path, self.secret_text
            )
            
            if lsb_results and dct_results:
                # Hiển thị kết quả
                result_text = f"""
=== BÁO CÁO SO SÁNH HIỆU NĂNG LSB vs DCT ===

1. THỜI GIAN XỬ LÝ:
   LSB - Nhúng: {lsb_results['embed_time']:.4f}s, Trích xuất: {lsb_results['extract_time']:.4f}s
   DCT - Nhúng: {dct_results['embed_time']:.4f}s, Trích xuất: {dct_results['extract_time']:.4f}s
   Tổng thời gian LSB: {lsb_results['embed_time'] + lsb_results['extract_time']:.4f}s
   Tổng thời gian DCT: {dct_results['embed_time'] + dct_results['extract_time']:.4f}s

2. CHẤT LƯỢNG ẢNH:
   LSB - PSNR: {lsb_results['psnr']:.2f} dB, SSIM: {lsb_results['ssim']:.4f}
   DCT - PSNR: {dct_results['psnr']:.2f} dB, SSIM: {dct_results['ssim']:.4f}

3. ĐỘ CHÍNH XÁC:
   LSB: {'✓' if lsb_results['accuracy'] else '✗'}
   DCT: {'✓' if dct_results['accuracy'] else '✗'}

4. KẾT LUẬN:
   - LSB {'nhanh hơn' if (lsb_results['embed_time'] + lsb_results['extract_time']) < (dct_results['embed_time'] + dct_results['extract_time']) else 'chậm hơn'} DCT
   - DCT {'có chất lượng ảnh tốt hơn' if dct_results['psnr'] > lsb_results['psnr'] else 'có chất lượng ảnh thấp hơn'} (PSNR cao hơn)
   - DCT {'có độ tương đồng cấu trúc tốt hơn' if dct_results['ssim'] > lsb_results['ssim'] else 'có độ tương đồng cấu trúc thấp hơn'} (SSIM cao hơn)

=== PHÂN TÍCH HOÀN THÀNH ===
                """
                
                self.results_text.delete("1.0", tk.END)
                self.results_text.insert("1.0", result_text)
                self.status_label.config(text="Phân tích hoàn thành!")
                
                # Hiển thị thông báo thành công
                messagebox.showinfo("Thành công", "Phân tích LSB vs DCT đã hoàn thành!\nXem kết quả trong phần hiển thị bên dưới.")
                
            else:
                messagebox.showerror("Lỗi", "Không thể thực hiện phân tích!")
                self.status_label.config(text="Có lỗi xảy ra")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong quá trình phân tích: {str(e)}")
            self.status_label.config(text="Có lỗi xảy ra")

def main():
    root = tk.Tk()
    app = SimpleSteganographyGUI(root)
    print("GUI đang khởi tạo...")
    root.mainloop()
    print("GUI đã đóng.")

if __name__ == "__main__":
    main()
