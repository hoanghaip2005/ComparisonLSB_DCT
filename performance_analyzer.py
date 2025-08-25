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
        
        return report
    
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
