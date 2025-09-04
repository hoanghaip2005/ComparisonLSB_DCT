print("[INFO] Starting GUI application...")

import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import os

print("[INFO] Importing steganography modules...")
from lsb_steganography import LSBSteganography
from dct_steganography import DCTSteganography
from performance_analyzer import PerformanceAnalyzer

class SteganographyGUI:
    def __init__(self, root):
        """Khởi tạo giao diện chính"""
        print("[INFO] Initializing GUI...")
        self.root = root
        self.setup_window()
        self.init_steganography()
        self.create_gui()
        print("[INFO] GUI initialized successfully")
        
    def setup_window(self):
        """Thiết lập cửa sổ chính"""
        self.root.title("So sánh phương pháp giấu tin LSB và DCT")
        self.root.geometry("1200x800")
        
        # Căn giữa cửa sổ
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 1200) // 2
        y = (screen_height - 800) // 2
        self.root.geometry(f"1200x800+{x}+{y}")
        print("[INFO] Window setup completed")
        
    def init_steganography(self):
        """Khởi tạo đối tượng steganography"""
        print("[INFO] Initializing steganography objects...")
        self.lsb = LSBSteganography()
        self.dct = DCTSteganography()
        self.analyzer = PerformanceAnalyzer()
        
        # Biến lưu trữ cho mỗi phương pháp
        self.lsb_data = {
            'image_path': None,
            'original_image': None,
            'stego_image': None,
            'secret_text': None,
            'text_widget': None,
            'original_label': None,
            'stego_label': None
        }
        
        self.dct_data = {
            'image_path': None,
            'original_image': None,
            'stego_image': None,
            'secret_text': None,
            'text_widget': None,
            'original_label': None,
            'stego_label': None
        }
        print("[INFO] Steganography initialization completed")
        
    def create_gui(self):
        """Tạo giao diện người dùng"""
        print("[INFO] Creating GUI elements...")
        # Tạo header
        self.create_header()
        
        # Tạo notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tạo các tab
        self.create_method_tab("LSB", self.lsb_data)
        self.create_method_tab("DCT", self.dct_data)
        print("[INFO] GUI creation completed")
        
    def create_header(self):
        """Tạo phần header của giao diện"""
        header = ttk.Frame(self.root)
        header.pack(fill='x', padx=10, pady=5)
        
        title = ttk.Label(header, 
                         text="SO SÁNH PHƯƠNG PHÁP GIẤU TIN LSB VÀ DCT",
                         font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
    def create_method_tab(self, method_name, data_dict):
        """Tạo tab cho một phương pháp steganography"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=f'{method_name} Steganography')
        
        # Chia layout thành 2 phần
        left_frame = ttk.Frame(tab)
        left_frame.pack(side='left', fill='y', padx=10, pady=5)
        
        right_frame = ttk.Frame(tab)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=5)
        
        # Tạo các thành phần trong tab
        self.create_input_section(left_frame, method_name, data_dict)
        self.create_image_section(right_frame, data_dict)
        
    def create_input_section(self, parent, method_name, data_dict):
        """Tạo phần nhập liệu"""
        # Frame cho text input
        text_frame = ttk.LabelFrame(parent, text="Nội dung cần giấu")
        text_frame.pack(fill='x', padx=5, pady=5)
        
        text_widget = scrolledtext.ScrolledText(text_frame, height=5, width=40)
        text_widget.pack(padx=5, pady=5)
        data_dict['text_widget'] = text_widget
        
        # Frame cho các nút điều khiển chính
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, 
                  text="Chọn ảnh", 
                  command=lambda: self.choose_image(method_name, data_dict)
                  ).pack(side='left', padx=5)
        
        ttk.Button(control_frame, 
                  text="Giấu tin", 
                  command=lambda: self.hide_data(method_name, data_dict)
                  ).pack(side='left', padx=5)
                  
        # Frame cho các nút điều khiển phụ
        extra_control_frame = ttk.Frame(parent)
        extra_control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(extra_control_frame,
                  text="Lưu ảnh",
                  command=lambda: self.save_stego_image(method_name, data_dict)
                  ).pack(side='left', padx=5)
                  
        ttk.Button(extra_control_frame,
                  text="Trích xuất tin",
                  command=lambda: self.extract_data(method_name, data_dict)
                  ).pack(side='left', padx=5)
                  
        ttk.Button(extra_control_frame,
                  text="So sánh biểu đồ",
                  command=lambda: self.show_comparison_charts(method_name, data_dict)
                  ).pack(side='left', padx=5)
                  
        # Frame cho thông tin dung lượng
        capacity_frame = ttk.LabelFrame(parent, text="Thông tin dung lượng")
        capacity_frame.pack(fill='x', padx=5, pady=5)
        
        capacity_label = ttk.Label(capacity_frame, text="Chưa có thông tin")
        capacity_label.pack(padx=5, pady=5)
        data_dict['capacity_label'] = capacity_label
                  
        # Frame cho kết quả phân tích
        analysis_frame = ttk.LabelFrame(parent, text="Kết quả phân tích")
        analysis_frame.pack(fill='x', padx=5, pady=5)
        
        # Text widget để hiển thị kết quả phân tích
        analysis_text = scrolledtext.ScrolledText(analysis_frame, height=8, width=40)
        analysis_text.pack(padx=5, pady=5)
        data_dict['analysis_text'] = analysis_text
        
    def create_image_section(self, parent, data_dict):
        """Tạo phần hiển thị ảnh"""
        # Frame cho ảnh gốc
        original_frame = ttk.LabelFrame(parent, text="Ảnh gốc")
        original_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        original_label = ttk.Label(original_frame)
        original_label.pack(fill='both', expand=True, padx=5, pady=5)
        data_dict['original_label'] = original_label
        
        # Frame cho ảnh kết quả
        stego_frame = ttk.LabelFrame(parent, text="Ảnh sau khi giấu tin")
        stego_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        stego_label = ttk.Label(stego_frame)
        stego_label.pack(fill='both', expand=True, padx=5, pady=5)
        data_dict['stego_label'] = stego_label
        
    def choose_image(self, method_name, data_dict):
        """Xử lý chọn ảnh"""
        print(f"[INFO] Choosing image for {method_name}...")
        file_path = filedialog.askopenfilename(
            title=f"Chọn ảnh cho {method_name}",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Đọc và lưu ảnh
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Không thể đọc ảnh")
                
                data_dict['image_path'] = file_path
                data_dict['original_image'] = image.copy()
                
                # Hiển thị ảnh
                self.display_image(image, data_dict['original_label'])
                print(f"[INFO] Image loaded successfully for {method_name}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load image: {str(e)}")
                messagebox.showerror("Lỗi", f"Không thể mở ảnh: {str(e)}")
                
    def hide_data(self, method_name, data_dict):
        """Xử lý giấu tin"""
        print(f"[INFO] Starting data hiding process for {method_name}...")
        if not data_dict['image_path']:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
            
        secret_text = data_dict['text_widget'].get("1.0", "end-1c").strip()
        if not secret_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập nội dung cần giấu!")
            return
            
        try:
            # Thực hiện giấu tin
            steganography = self.lsb if method_name == "LSB" else self.dct
            print(f"[INFO] Embedding data using {method_name}...")
            
            # Phân tích hiệu năng
            stego_image, metrics = self.analyze_performance(
                method_name, 
                steganography, 
                data_dict['image_path'], 
                secret_text
            )
            
            # Lưu và hiển thị kết quả
            data_dict['stego_image'] = stego_image
            self.display_image(stego_image, data_dict['stego_label'])
            
            # Hiển thị kết quả phân tích
            self.display_analysis_results(data_dict['analysis_text'], metrics)
            
            # Thông báo thành công
            print(f"[INFO] Data hiding successful for {method_name}")
            messagebox.showinfo("Thành công", 
                              f"Đã giấu tin thành công bằng phương pháp {method_name}!")
            
        except Exception as e:
            print(f"[ERROR] Data hiding failed: {str(e)}")
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra khi giấu tin: {str(e)}")

    def analyze_performance(self, method_name, steganography, image_path, secret_text):
        """Phân tích hiệu năng của phương pháp giấu tin"""
        import time
        start_time = time.time()
        
        # Thực hiện giấu tin và ghi nhận thời gian
        stego_image = steganography.embed(image_path, secret_text)
        processing_time = time.time() - start_time
        
        # Tính toán PSNR và SSIM
        metrics = self.analyzer.calculate_metrics(
            cv2.imread(image_path),
            stego_image[0] if isinstance(stego_image, tuple) else stego_image
        )
        
        # Thêm thời gian xử lý vào metrics
        metrics['processing_time'] = processing_time
        metrics['method'] = method_name
        
        return stego_image, metrics
        
    def display_analysis_results(self, text_widget, metrics):
        """Hiển thị kết quả phân tích"""
        result_text = f"""Kết quả phân tích {metrics['method']}:

1. Chất lượng ảnh:
   - PSNR: {metrics['psnr']:.2f} dB
   - SSIM: {metrics['ssim']:.4f}
   
2. Hiệu năng:
   - Thời gian xử lý: {metrics['processing_time']*1000:.2f} ms

3. Đánh giá:
   - Chất lượng: {"Tốt" if metrics['psnr'] > 40 else "Trung bình" if metrics['psnr'] > 30 else "Kém"}
   - Độ bền: {"Cao" if metrics['ssim'] > 0.95 else "Trung bình" if metrics['ssim'] > 0.9 else "Thấp"}
   - Tốc độ: {"Nhanh" if metrics['processing_time'] < 0.1 else "Trung bình" if metrics['processing_time'] < 0.5 else "Chậm"}
"""
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", result_text)

    def save_stego_image(self, method_name, data_dict):
        """Lưu ảnh đã giấu tin"""
        if data_dict['stego_image'] is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh để lưu!")
            return
        
        # Mở dialog chọn vị trí lưu file
        dialog_title = f"Lưu ảnh {method_name}"
        filetypes = [("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        file_path = filedialog.asksaveasfilename(initialfile="stego_image.png",
                                                title=dialog_title,
                                                defaultextension=".png",
                                                filetypes=filetypes)
        
        if not file_path:
            return
            
        try:
            # Xử lý trường hợp ảnh được trả về dạng tuple
            stego_image = data_dict['stego_image']
            if isinstance(stego_image, tuple):
                stego_image = stego_image[0]
                
            # Lưu ảnh
            cv2.imwrite(file_path, stego_image)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {str(e)}")
                
    def copy_to_clipboard(self, dialog, text):
        """Copy text vào clipboard"""
        dialog.clipboard_clear()
        dialog.clipboard_append(text)
        messagebox.showinfo("Thông báo", "Đã copy vào clipboard!")
        
    def extract_data(self, method_name, data_dict):
        """Trích xuất tin đã giấu"""
        if data_dict['stego_image'] is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh để trích xuất!")
            return
            
        try:
            # Xử lý trường hợp stego_image là tuple
            stego_image = data_dict['stego_image']
            if isinstance(stego_image, tuple):
                stego_image = stego_image[0]
                
            # Trích xuất tin
            steganography = self.lsb if method_name == "LSB" else self.dct
            extracted_text = steganography.extract(stego_image)
            
            if not extracted_text:
                messagebox.showwarning("Cảnh báo", "Không tìm thấy tin giấu trong ảnh!")
                return
            
            # Tạo dialog hiển thị tin đã trích xuất
            dialog = tk.Toplevel(self.root)
            dialog.title(f"Tin đã trích xuất - {method_name}")
            dialog.geometry("400x300")
            
            # Tạo text area cho việc hiển thị và cho phép copy
            text_area = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, width=45, height=15)
            text_area.pack(padx=10, pady=10, expand=True, fill='both')
            text_area.insert('1.0', extracted_text)
            text_area.configure(state='disabled')  # Chỉ cho phép đọc
            
            # Thêm các nút chức năng
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=5)
            
            copy_button = ttk.Button(button_frame, text="Copy", 
                                   command=lambda: self.copy_to_clipboard(dialog, extracted_text))
            copy_button.pack(side='left', padx=5)
            
            close_button = ttk.Button(button_frame, text="Đóng", 
                                    command=dialog.destroy)
            close_button.pack(side='left', padx=5)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể trích xuất tin: {str(e)}")
            
    def show_comparison_charts(self, method_name, data_dict):
        """Hiển thị biểu đồ so sánh"""
        if data_dict['original_image'] is None or data_dict['stego_image'] is None:
            messagebox.showwarning("Cảnh báo", "Chưa có đủ dữ liệu để so sánh!")
            return
            
        try:
            # Xử lý trường hợp stego_image là tuple
            original_image = data_dict['original_image']
            stego_image = data_dict['stego_image']
            if isinstance(stego_image, tuple):
                stego_image = stego_image[0]
            
            # Tính toán các thông số
            mse = np.mean((original_image - stego_image) ** 2)
            metrics = self.analyzer.calculate_metrics(original_image, stego_image)
            psnr = metrics['psnr']
            ssim = metrics['ssim']
            max_diff = np.max(cv2.absdiff(original_image, stego_image))
            
            # Tạo cửa sổ mới cho biểu đồ
            plt.figure(figsize=(15, 10))
            
            # Tiêu đề chính với thông số
            plt.suptitle(f'Phân tích ảnh - Phương pháp {method_name}\n' + 
                        f'PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f} | MSE: {mse:.4f}', 
                        fontsize=12, y=0.98)
            
            # Histogram ảnh gốc
            plt.subplot(2, 2, 1)
            for i, color, name in zip(range(3), ['b', 'g', 'r'], ['Blue', 'Green', 'Red']):
                channel_hist = cv2.calcHist([original_image], [i], None, [256], [0, 256])
                plt.plot(channel_hist, color=color, alpha=0.7, label=name)
            plt.title('Histogram ảnh gốc')
            plt.xlabel('Giá trị pixel')
            plt.ylabel('Số lượng')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Histogram ảnh đã giấu tin
            plt.subplot(2, 2, 2)
            for i, color, name in zip(range(3), ['b', 'g', 'r'], ['Blue', 'Green', 'Red']):
                channel_hist = cv2.calcHist([stego_image], [i], None, [256], [0, 256])
                plt.plot(channel_hist, color=color, alpha=0.7, label=name)
            plt.title('Histogram ảnh đã giấu tin')
            plt.xlabel('Giá trị pixel')
            plt.ylabel('Số lượng')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Độ khác biệt giữa hai ảnh
            plt.subplot(2, 2, 3)
            difference = cv2.absdiff(original_image, stego_image)
            plt.imshow(difference, cmap='hot')
            plt.colorbar(label='Độ khác biệt')
            plt.title('Bản đồ độ khác biệt')
            
            # Thông số chi tiết
            plt.subplot(2, 2, 4)
            plt.axis('off')
            info_text = (
                "THÔNG SỐ CHI TIẾT\n\n"
                f"1. Chất lượng hình ảnh:\n"
                f"   • PSNR: {psnr:.2f} dB\n"
                f"   • SSIM: {ssim:.4f}\n"
                f"   • MSE: {mse:.4f}\n\n"
                f"2. Phân tích nhiễu:\n"
                f"   • Độ khác biệt tối đa: {max_diff}\n"
                f"   • Trung bình nhiễu: {np.mean(difference):.4f}\n"
                f"   • Độ lệch chuẩn: {np.std(difference):.4f}\n\n"
                f"3. Đánh giá:\n"
                f"   • Chất lượng: {'Tốt' if psnr > 40 else 'Trung bình' if psnr > 30 else 'Kém'}\n"
                f"   • Độ bền: {'Cao' if ssim > 0.95 else 'Trung bình' if ssim > 0.9 else 'Thấp'}\n"
                f"   • % Pixel thay đổi: {(np.count_nonzero(difference) / difference.size * 100):.2f}%"
            )
            plt.text(0.05, 0.95, info_text, fontsize=10, va='top', 
                    fontfamily='monospace', linespacing=1.5)

            plt.tight_layout()
            plt.show()
            # Histogram ảnh gốc
            plt.subplot(221)
            plt.title("Histogram ảnh gốc")
            for i, color in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([data_dict['original_image']], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
            
            # Histogram ảnh đã giấu tin
            plt.subplot(222)
            plt.title("Histogram ảnh đã giấu tin")
            stego_img = data_dict['stego_image']
            if isinstance(stego_img, tuple):
                stego_img = stego_img[0]
            for i, color in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([stego_img], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
            
            # Độ khác biệt giữa hai ảnh
            plt.subplot(223)
            plt.title("Độ khác biệt")
            diff = cv2.absdiff(data_dict['original_image'], stego_img)
            plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
            
            # Hiển thị
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tạo biểu đồ: {str(e)}")
            
    def update_capacity_info(self, method_name, data_dict):
        """Cập nhật thông tin dung lượng"""
        if data_dict['image_path'] is None:
            return
            
        try:
            lsb_capacity, dct_capacity = self.analyzer.estimate_capacity_bits(data_dict['image_path'])
            capacity = lsb_capacity if method_name == "LSB" else dct_capacity
            
            # Chuyển đổi sang đơn vị dễ đọc
            if capacity >= 8000:
                capacity_text = f"{capacity/8/1024:.2f} KB"
            else:
                capacity_text = f"{capacity/8:.0f} bytes"
                
            data_dict['capacity_label'].configure(
                text=f"Dung lượng tối đa: {capacity_text}\n"
                     f"(tương đương {capacity:,} bits)"
            )
        except Exception as e:
            print(f"[ERROR] Failed to update capacity info: {str(e)}")
            
    def display_image(self, image, label, size=(300, 300)):
        """Hiển thị ảnh lên label widget"""
        if image is None:
            print("[WARNING] Attempted to display None image")
            return
            
        try:
            # Nếu image là tuple (từ kết quả embed), lấy phần tử đầu tiên
            if isinstance(image, tuple):
                print("[INFO] Converting tuple to image array")
                image = image[0]

            # Chuyển đổi array sang uint8 nếu cần
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
                
            # Chuyển đổi màu từ BGR sang RGB
            if len(image.shape) == 3:  # Ảnh màu
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:  # Ảnh grayscale
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize ảnh
            image_resized = cv2.resize(image_rgb, size)
            
            # Chuyển đổi sang định dạng PhotoImage
            image_pil = Image.fromarray(image_resized)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Hiển thị ảnh
            label.configure(image=image_tk)
            label.image = image_tk  # Giữ tham chiếu
            print("[INFO] Image displayed successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to display image: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh: {str(e)}")

def main():
    try:
        print("[INFO] Starting main program...")
        root = tk.Tk()
        app = SteganographyGUI(root)
        print("[INFO] Entering main loop")
        root.mainloop()
    except Exception as e:
        print(f"[ERROR] Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
