import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from lsb_steganography import LSBSteganography
from dct_steganography import DCTSteganography

class PerformanceAnalyzer:
    def __init__(self):
        self.lsb = LSBSteganography()
        self.dct = DCTSteganography()
        
    def estimate_capacity_bits(self, image_path):
        """Ước lượng dung lượng giấu của LSB và DCT (đơn vị: bit)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Không thể đọc ảnh để ước lượng dung lượng")
        height, width = img.shape[:2]
        # LSB: 1 bit cho mỗi kênh màu
        lsb_capacity_bits = height * width * 3  # không tính header
        # DCT: 1 bit cho mỗi block 8x8 ở kênh Y
        blocks_h = height // self.dct.block_size
        blocks_w = width // self.dct.block_size
        dct_capacity_bits = blocks_h * blocks_w  # mỗi block 1 bit
        # Trừ header 32 bit cho cả hai
        lsb_capacity_payload_bits = max(0, lsb_capacity_bits - 32)
        dct_capacity_payload_bits = max(0, dct_capacity_bits - 32)
        return lsb_capacity_payload_bits, dct_capacity_payload_bits

    def analyze_single_method(self, method, image_path, secret_text, method_name):
        """Phân tích hiệu năng của một phương pháp"""
        try:
            # Nhúng tin
            stego_img, embed_time = method.embed(image_path, secret_text)
            
            # Trích xuất tin
            extracted_text, extract_time = method.extract(stego_img)
            
            # Đọc ảnh gốc để tính metrics
            original_img = cv2.imread(image_path)
            
            # Tính toán metrics
            metrics = method.calculate_metrics(original_img, stego_img)
            
            # Kiểm tra độ chính xác
            accuracy = 1.0 if extracted_text == secret_text else 0.0
            
            return {
                'method': method_name,
                'embed_time': embed_time,
                'extract_time': extract_time,
                'total_time': embed_time + extract_time,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'mse': metrics['mse'],
                'accuracy': accuracy,
                'stego_image': stego_img,
                'extracted_text': extracted_text
            }
            
        except Exception as e:
            print(f"Lỗi khi phân tích {method_name}: {str(e)}")
            return None
    
    def compare_methods(self, image_path, secret_text):
        """So sánh hiệu năng giữa LSB và DCT"""
        print("Đang phân tích hiệu năng LSB...")
        lsb_results = self.analyze_single_method(self.lsb, image_path, secret_text, "LSB")
        
        print("Đang phân tích hiệu năng DCT...")
        dct_results = self.analyze_single_method(self.dct, image_path, secret_text, "DCT")
        
        # Ước lượng dung lượng
        try:
            lsb_cap_bits, dct_cap_bits = self.estimate_capacity_bits(image_path)
        except Exception:
            lsb_cap_bits, dct_cap_bits = 0, 0
        if lsb_results is not None:
            lsb_results['capacity_bits'] = lsb_cap_bits
        if dct_results is not None:
            dct_results['capacity_bits'] = dct_cap_bits
        return lsb_results, dct_results
    
    def generate_comparison_report(self, lsb_results, dct_results):
        """Tạo báo cáo so sánh"""
        if not lsb_results or not dct_results:
            return "Không thể tạo báo cáo do lỗi trong quá trình phân tích"
        
        report = "=== BÁO CÁO SO SÁNH HIỆU NĂNG LSB vs DCT ===\n\n"
        
        # So sánh thời gian xử lý
        report += "1. THỜI GIAN XỬ LÝ:\n"
        report += f"   LSB - Nhúng: {lsb_results['embed_time']:.4f}s, Trích xuất: {lsb_results['extract_time']:.4f}s\n"
        report += f"   DCT - Nhúng: {dct_results['embed_time']:.4f}s, Trích xuất: {dct_results['extract_time']:.4f}s\n"
        report += f"   Tổng thời gian LSB: {lsb_results['total_time']:.4f}s\n"
        report += f"   Tổng thời gian DCT: {dct_results['total_time']:.4f}s\n\n"
        
        # So sánh chất lượng ảnh
        report += "2. CHẤT LƯỢNG ẢNH:\n"
        report += f"   LSB - PSNR: {lsb_results['psnr']:.2f} dB, SSIM: {lsb_results['ssim']:.4f}\n"
        report += f"   DCT - PSNR: {dct_results['psnr']:.2f} dB, SSIM: {dct_results['ssim']:.4f}\n\n"
        # Nhận xét nhìn bằng mắt thường dựa trên ngưỡng PSNR/SSIM
        def subjective_comment(psnr, ssim):
            if psnr == float('inf') or (psnr >= 40 and ssim >= 0.98):
                return "Khó phân biệt bằng mắt thường"
            if psnr >= 35 and ssim >= 0.95:
                return "Rất khó phân biệt, khác biệt rất nhỏ"
            if psnr >= 30 and ssim >= 0.90:
                return "Có thể phân biệt khi phóng to/khuếch đại"
            return "Khác biệt có thể thấy rõ"
        report += f"   Nhìn bằng mắt thường (LSB): {subjective_comment(lsb_results['psnr'], lsb_results['ssim'])}\n"
        report += f"   Nhìn bằng mắt thường (DCT): {subjective_comment(dct_results['psnr'], dct_results['ssim'])}\n\n"
        
        # So sánh độ chính xác
        report += "3. ĐỘ CHÍNH XÁC:\n"
        report += f"   LSB: {'✓' if lsb_results['accuracy'] == 1.0 else '✗'}\n"
        report += f"   DCT: {'✓' if dct_results['accuracy'] == 1.0 else '✗'}\n\n"
        
        # Kết luận
        report += "4. KẾT LUẬN:\n"
        
        # So sánh tốc độ
        if lsb_results['total_time'] < dct_results['total_time']:
            report += "   - LSB nhanh hơn DCT\n"
        else:
            report += "   - DCT nhanh hơn LSB\n"
        
        # So sánh chất lượng
        if lsb_results['psnr'] > dct_results['psnr']:
            report += "   - LSB có chất lượng ảnh tốt hơn (PSNR cao hơn)\n"
        else:
            report += "   - DCT có chất lượng ảnh tốt hơn (PSNR cao hơn)\n"
        
        if lsb_results['ssim'] > dct_results['ssim']:
            report += "   - LSB có độ tương đồng cấu trúc tốt hơn (SSIM cao hơn)\n"
        else:
            report += "   - DCT có độ tương đồng cấu trúc tốt hơn (SSIM cao hơn)\n"
        # Dung lượng
        if 'capacity_bits' in lsb_results and 'capacity_bits' in dct_results:
            report += "\n5. DUNG LƯỢNG GIẤU:\n"
            report += f"   LSB: ~{lsb_results['capacity_bits']//8:,} byte ({lsb_results['capacity_bits']:,} bit)\n"
            report += f"   DCT: ~{dct_results['capacity_bits']//8:,} byte ({dct_results['capacity_bits']:,} bit)\n"
            if lsb_results['capacity_bits'] > dct_results['capacity_bits']:
                report += "   → LSB có dung lượng giấu lớn hơn.\n"
            elif lsb_results['capacity_bits'] < dct_results['capacity_bits']:
                report += "   → DCT có dung lượng giấu lớn hơn.\n"
            else:
                report += "   → Hai phương pháp có dung lượng tương đương trên ảnh này.\n"
        
        return report

    def show_metric_charts(self, lsb_results, dct_results):
        """Hiển thị biểu đồ PSNR/SSIM, thời gian và dung lượng"""
        if not lsb_results or not dct_results:
            print("Không thể hiển thị biểu đồ do thiếu kết quả")
            return
        methods = ['LSB', 'DCT']
        psnr_values = [lsb_results['psnr'], dct_results['psnr']]
        ssim_values = [lsb_results['ssim'], dct_results['ssim']]
        # Tránh giá trị bằng 0 làm biến mất cột khi dùng log
        time_values = [max(lsb_results['total_time'], 1e-6), max(dct_results['total_time'], 1e-6)]
        cap_values = [lsb_results.get('capacity_bits', 0)/8, dct_results.get('capacity_bits', 0)/8]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        # PSNR/SSIM
        x = np.arange(len(methods))
        width = 0.35
        axes[0].bar(x - width/2, psnr_values, width, label='PSNR (dB)')
        axes[0].bar(x + width/2, ssim_values, width, label='SSIM')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods)
        axes[0].set_title('PSNR / SSIM')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Hiển thị giá trị trên đầu cột
        for rects in axes[0].containers:
            axes[0].bar_label(rects, fmt='%.2f', padding=2, fontsize=8)
        # Time
        bars_time = axes[1].bar(methods, time_values, color=['#e67e22', '#9b59b6'])
        axes[1].set_title('Tổng thời gian (s)')
        axes[1].grid(True, alpha=0.3)
        # Dùng thang log nếu chênh lệch quá lớn để nhìn rõ cột nhỏ
        if max(time_values) / max(min(time_values), 1e-6) > 50:
            axes[1].set_yscale('log')
        axes[1].bar_label(bars_time, fmt='%.4g', padding=2, fontsize=8)
        # Capacity
        bars_cap = axes[2].bar(methods, cap_values, color=['#27ae60', '#2980b9'])
        axes[2].set_title('Dung lượng ước lượng (byte)')
        axes[2].grid(True, alpha=0.3)
        # Dùng log cho dung lượng để thấy rõ chênh lệch lớn
        if max(cap_values) / max(min([v for v in cap_values if v > 0] + [1]), 1) > 50:
            axes[2].set_yscale('log')
        axes[2].bar_label(bars_cap, fmt='%.4g', padding=2, fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def create_visual_comparison(self, original_img, lsb_results, dct_results, save_path="comparison_results.png"):
        """Tạo so sánh trực quan"""
        if not lsb_results or not dct_results:
            print("Không thể tạo so sánh trực quan")
            return
        
        # Tạo figure với 4 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('So sánh LSB vs DCT Steganography', fontsize=16, fontweight='bold')
        
        # Ảnh gốc
        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Ảnh Gốc', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ảnh LSB
        axes[0, 1].imshow(cv2.cvtColor(lsb_results['stego_image'], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'LSB Stego (PSNR: {lsb_results["psnr"]:.2f}dB)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Ảnh DCT
        axes[1, 0].imshow(cv2.cvtColor(dct_results['stego_image'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'DCT Stego (PSNR: {dct_results["psnr"]:.2f}dB)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Biểu đồ so sánh
        methods = ['LSB', 'DCT']
        psnr_values = [lsb_results['psnr'], dct_results['psnr']]
        ssim_values = [lsb_results['ssim'], dct_results['ssim']]
        time_values = [lsb_results['total_time'], dct_results['total_time']]
        
        x = np.arange(len(methods))
        width = 0.25
        
        # Chuẩn hóa giá trị để vẽ trên cùng một biểu đồ
        psnr_norm = [p/100 for p in psnr_values]  # PSNR thường > 30dB
        ssim_norm = ssim_values  # SSIM từ -1 đến 1
        time_norm = [t/max(time_values) for t in time_values]  # Thời gian
        
        axes[1, 1].bar(x - width, psnr_norm, width, label='PSNR (norm)', alpha=0.8)
        axes[1, 1].bar(x, ssim_norm, width, label='SSIM', alpha=0.8)
        axes[1, 1].bar(x + width, time_norm, width, label='Thời gian (norm)', alpha=0.8)
        
        axes[1, 1].set_xlabel('Phương pháp')
        axes[1, 1].set_ylabel('Giá trị chuẩn hóa')
        axes[1, 1].set_title('So sánh hiệu năng')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(methods)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def save_results(self, lsb_results, dct_results, save_path="stego_results"):
        """Lưu kết quả ảnh"""
        if lsb_results:
            cv2.imwrite(f"{save_path}_lsb.png", lsb_results['stego_image'])
            print(f"Đã lưu ảnh LSB: {save_path}_lsb.png")
        
        if dct_results:
            cv2.imwrite(f"{save_path}_dct.png", dct_results['stego_image'])
            print(f"Đã lưu ảnh DCT: {save_path}_dct.png")
