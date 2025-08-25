import sys
print("[DEBUG] Python version:", sys.version)
try:
    import tkinter as tk
    print("[DEBUG] Tkinter imported successfully.")
except Exception as e:
    print("[ERROR] Tkinter import failed:", e)
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from lsb_steganography import LSBSteganography
from dct_steganography import DCTSteganography
from performance_analyzer import PerformanceAnalyzer
import sys
print("Python version:", sys.version)
print("Tkinter test:", __import__('tkinter'))
class SteganographyComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("So sánh hiệu năng LSB vs DCT Steganography")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Khởi tạo các đối tượng steganography
        self.lsb = LSBSteganography()
        self.dct = DCTSteganography()
        self.analyzer = PerformanceAnalyzer()
        
        # Biến lưu trữ
        self.original_image_path = None
        self.original_image = None
        self.lsb_stego_image = None
        self.dct_stego_image = None
        self.secret_text = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        """Thiết lập giao diện người dùng"""
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="PHÂN TÍCH SO SÁNH HIỆU NĂNG LSB vs DCT STEGANOGRAPHY", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=20)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Image display
        left_panel = tk.Frame(main_frame, bg='#f0f0f0')
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Image selection
        img_select_frame = tk.Frame(left_panel, bg='#f0f0f0')
        img_select_frame.pack(fill='x', pady=(0, 10))
        
        tk.Button(img_select_frame, text="Chọn Ảnh", command=self.select_image, 
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=(0, 10))
        
        self.image_path_label = tk.Label(img_select_frame, text="Chưa chọn ảnh", 
                                        bg='#f0f0f0', font=('Arial', 9))
        self.image_path_label.pack(side='left')
        
        # Image display area
        img_display_frame = tk.Frame(left_panel, bg='white', relief='raised', bd=2)
        img_display_frame.pack(fill='both', expand=True)
        
        # Original image
        tk.Label(img_display_frame, text="Ảnh Gốc", font=('Arial', 12, 'bold'), 
                bg='white').pack(pady=5)
        
        self.original_img_label = tk.Label(img_display_frame, text="Chưa có ảnh", 
                                          bg='white', width=40, height=15)
        self.original_img_label.pack(pady=5)
        
        # Right panel - Controls and results
        right_panel = tk.Frame(main_frame, bg='#f0f0f0')
        right_panel.pack(side='right', fill='both', padx=(5, 0))
        
        # Text input
        text_frame = tk.Frame(right_panel, bg='#f0f0f0')
        text_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(text_frame, text="Nhập tin cần giấu:", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0').pack(anchor='w')
        
        self.text_input = scrolledtext.ScrolledText(text_frame, height=4, width=40, 
                                                   font=('Arial', 10))
        self.text_input.pack(fill='x', pady=5)
        
        # Buttons
        button_frame = tk.Frame(right_panel, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=(0, 10))
        
        tk.Button(button_frame, text="So sánh LSB vs DCT", command=self.compare_methods,
                 bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'), height=2).pack(fill='x', pady=2)
        
        tk.Button(button_frame, text="Xem ảnh kết quả", command=self.show_results,
                 bg='#f39c12', fg='white', font=('Arial', 11, 'bold'), height=2).pack(fill='x', pady=2)
        
        tk.Button(button_frame, text="Lưu kết quả", command=self.save_results,
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'), height=2).pack(fill='x', pady=2)
        
        # Results display
        results_frame = tk.Frame(right_panel, bg='white', relief='raised', bd=2)
        results_frame.pack(fill='both', expand=True)
        
        tk.Label(results_frame, text="Kết quả phân tích:", font=('Arial', 12, 'bold'), 
                bg='white').pack(pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=50, 
                                                     font=('Consolas', 9))
        self.results_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Sẵn sàng", fg='white', bg='#34495e')
        self.status_label.pack(side='left', padx=10, pady=5)
        
    def select_image(self):
        """Chọn ảnh để xử lý"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.original_image_path = file_path
                self.original_image = cv2.imread(file_path)
                
                # Hiển thị ảnh
                self.display_image(self.original_image, self.original_img_label)
                
                # Cập nhật label
                filename = os.path.basename(file_path)
                self.image_path_label.config(text=f"Đã chọn: {filename}")
                
                self.status_label.config(text=f"Đã tải ảnh: {filename}")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")
    
    def display_image(self, image, label, max_size=(300, 300)):
        """Hiển thị ảnh trong label"""
        if image is None:
            label.config(text="Không có ảnh")
            return
        
        # Resize ảnh để vừa với label
        height, width = image.shape[:2]
        if height > max_size[0] or width > max_size[1]:
            scale = min(max_size[0]/height, max_size[1]/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Chuyển đổi BGR sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Chuyển đổi sang PIL Image
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Cập nhật label
        label.config(image=photo, text="")
        label.image = photo  # Giữ reference
    
    def compare_methods(self):
        """So sánh hiệu năng LSB vs DCT"""
        if not self.original_image_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        
        # Lấy text cần giấu
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
            
            # Lưu kết quả
            self.lsb_stego_image = lsb_results['stego_image'] if lsb_results else None
            self.dct_stego_image = dct_results['stego_image'] if dct_results else None
            
            # Tạo báo cáo
            report = self.analyzer.generate_comparison_report(lsb_results, dct_results)
            
            # Hiển thị kết quả
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert("1.0", report)
            
            self.status_label.config(text="Phân tích hoàn tất!")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong quá trình phân tích: {str(e)}")
            self.status_label.config(text="Có lỗi xảy ra")
    
    def show_results(self):
        """Hiển thị ảnh kết quả"""
        if self.lsb_stego_image is None or self.dct_stego_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng thực hiện phân tích trước!")
            return
        
        # Tạo cửa sổ mới để hiển thị kết quả
        results_window = tk.Toplevel(self.root)
        results_window.title("Kết quả so sánh LSB vs DCT")
        results_window.geometry("1000x600")
        
        # Tạo frame chính
        main_frame = tk.Frame(results_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Ảnh gốc
        tk.Label(main_frame, text="Ảnh Gốc", font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=5)
        original_label = tk.Label(main_frame)
        original_label.grid(row=1, column=0, padx=5, pady=5)
        self.display_image(self.original_image, original_label, (200, 200))
        
        # Ảnh LSB
        tk.Label(main_frame, text="LSB Stego", font=('Arial', 12, 'bold')).grid(row=0, column=1, pady=5)
        lsb_label = tk.Label(main_frame)
        lsb_label.grid(row=1, column=1, padx=5, pady=5)
        self.display_image(self.lsb_stego_image, lsb_label, (200, 200))
        
        # Ảnh DCT
        tk.Label(main_frame, text="DCT Stego", font=('Arial', 12, 'bold')).grid(row=0, column=2, pady=5)
        dct_label = tk.Label(main_frame)
        dct_label.grid(row=1, column=2, padx=5, pady=5)
        self.display_image(self.dct_stego_image, dct_label, (200, 200))
        
        # Thông tin chi tiết
        info_frame = tk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        # Lấy thông tin từ analyzer
        lsb_results, dct_results = self.analyzer.compare_methods(
            self.original_image_path, self.secret_text
        )
        
        if lsb_results and dct_results:
            info_text = f"""
            THÔNG TIN CHI TIẾT:
            
            LSB:
            - Thời gian nhúng: {lsb_results['embed_time']:.4f}s
            - Thời gian trích xuất: {lsb_results['extract_time']:.4f}s
            - PSNR: {lsb_results['psnr']:.2f} dB
            - SSIM: {lsb_results['ssim']:.4f}
            
            DCT:
            - Thời gian nhúng: {dct_results['embed_time']:.4f}s
            - Thời gian trích xuất: {dct_results['extract_time']:.4f}s
            - PSNR: {dct_results['psnr']:.2f} dB
            - SSIM: {dct_results['ssim']:.4f}
            """
            
            info_label = tk.Label(info_frame, text=info_text, font=('Consolas', 10), 
                                 justify='left', bg='#f8f9fa', relief='raised', bd=2)
            info_label.pack(pady=10)
    
    def save_results(self):
        """Lưu kết quả"""
        if self.lsb_stego_image is None or self.dct_stego_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng thực hiện phân tích trước!")
            return
        
        try:
            # Chọn thư mục lưu
            save_dir = filedialog.askdirectory(title="Chọn thư mục lưu kết quả")
            if save_dir:
                # Lưu ảnh
                cv2.imwrite(os.path.join(save_dir, "lsb_stego.png"), self.lsb_stego_image)
                cv2.imwrite(os.path.join(save_dir, "dct_stego.png"), self.dct_stego_image)
                
                # Lưu báo cáo
                report_path = os.path.join(save_dir, "comparison_report.txt")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get("1.0", tk.END))
                
                messagebox.showinfo("Thành công", f"Đã lưu kết quả vào:\n{save_dir}")
                self.status_label.config(text="Đã lưu kết quả")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu kết quả: {str(e)}")

def main():
    print("[DEBUG] Khởi tạo Tkinter window...")
    root = tk.Tk()
    print("[DEBUG] Tkinter window created.")
    app = SteganographyComparisonGUI(root)
    print("[DEBUG] SteganographyComparisonGUI initialized.")
    root.mainloop()
    print("[DEBUG] Tkinter mainloop exited.")

if __name__ == "__main__":
    main()
