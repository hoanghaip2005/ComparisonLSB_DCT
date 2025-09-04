import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from lsb_steganography import LSBSteganography
from dct_steganography import DCTSteganography

class PerformanceAnalyzer:
    def __init__(self):
        self.lsb = LSBSteganography()
        self.dct = DCTSteganography()
        
    def calculate_metrics(self, original_img, stego_img):
        """Tính toán các chỉ số đánh giá chất lượng"""
        if original_img is None or stego_img is None:
            raise ValueError("Ảnh đầu vào không được là None")
            
        # Chuyển đổi sang grayscale nếu cần
        if len(original_img.shape) == 3:
            original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_img.copy()
            
        if len(stego_img.shape) == 3:
            stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
        else:
            stego_gray = stego_img.copy()
            
        # Tính PSNR
        try:
            psnr_value = psnr(original_gray, stego_gray, data_range=255)
        except Exception as e:
            # Fallback nếu có lỗi
            mse = np.mean((original_gray.astype(float) - stego_gray.astype(float)) ** 2)
            psnr_value = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else 100
            
        # Tính SSIM
        try:
            ssim_value = ssim(original_gray, stego_gray, data_range=255)
        except Exception as e:
            ssim_value = 0
            
        return {
            'psnr': psnr_value,
            'ssim': ssim_value
        }
        
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

    def calculate_advanced_metrics(self, original_img, stego_img):
        """Tính toán các chỉ số nâng cao"""
        # Chuyển đổi sang grayscale nếu cần
        if len(original_img.shape) == 3:
            original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_img
            
        if len(stego_img.shape) == 3:
            stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
        else:
            stego_gray = stego_img
        
        # Tính PSNR sử dụng scikit-image
        try:
            psnr_value = psnr(original_gray, stego_gray, data_range=255)
        except:
            # Fallback nếu có lỗi
            mse = np.mean((original_gray.astype(float) - stego_gray.astype(float)) ** 2)
            psnr_value = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Tính SSIM sử dụng scikit-image
        try:
            ssim_value = ssim(original_gray, stego_gray, data_range=255)
        except:
            # Fallback calculation
            ssim_value = self._calculate_ssim_fallback(original_gray, stego_gray)
        
        # Tính MSE (Mean Squared Error)
        mse = np.mean((original_gray.astype(float) - stego_gray.astype(float)) ** 2)
        
        # Tính SME (Squared Mean Error) - bình phương của mean error
        mean_error = np.mean(original_gray.astype(float) - stego_gray.astype(float))
        sme = mean_error ** 2
        
        # Tính MAE (Mean Absolute Error)
        mae = np.mean(np.abs(original_gray.astype(float) - stego_gray.astype(float)))
        
        # Tính RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        
        # Tính Normalized Cross Correlation (NCC)
        ncc = np.corrcoef(original_gray.flatten(), stego_gray.flatten())[0, 1]
        if np.isnan(ncc):
            ncc = 0.0
        
        return {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'mse': mse,
            'sme': sme,
            'mae': mae,
            'rmse': rmse,
            'ncc': ncc
        }
    
    def _calculate_ssim_fallback(self, img1, img2):
        """Fallback SSIM calculation"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim_value = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                    ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
        
        return ssim_value

    def test_robustness(self, method, image_path, secret_text, method_name):
        """Test độ bền vững (robustness) của phương pháp"""
        try:
            # Nhúng tin
            stego_img, _ = method.embed(image_path, secret_text)
            
            # Test với các loại nhiễu khác nhau
            robustness_tests = {}
            
            # 1. Gaussian noise
            noise = np.random.normal(0, 5, stego_img.shape).astype(np.uint8)
            noisy_img = cv2.add(stego_img, noise)
            try:
                extracted_text, _ = method.extract(noisy_img)
                robustness_tests['gaussian_noise'] = 1.0 if extracted_text == secret_text else 0.0
            except:
                robustness_tests['gaussian_noise'] = 0.0
            
            # 2. Salt and pepper noise
            noisy_img_sp = stego_img.copy()
            noise_mask = np.random.random(stego_img.shape[:2]) < 0.01
            noisy_img_sp[noise_mask] = 255
            noisy_img_sp[noise_mask] = 0
            try:
                extracted_text, _ = method.extract(noisy_img_sp)
                robustness_tests['salt_pepper'] = 1.0 if extracted_text == secret_text else 0.0
            except:
                robustness_tests['salt_pepper'] = 0.0
            
            # 3. JPEG compression simulation
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, encimg = cv2.imencode('.jpg', stego_img, encode_param)
            compressed_img = cv2.imdecode(encimg, 1)
            try:
                extracted_text, _ = method.extract(compressed_img)
                robustness_tests['jpeg_compression'] = 1.0 if extracted_text == secret_text else 0.0
            except:
                robustness_tests['jpeg_compression'] = 0.0
            
            # 4. Slight blur
            blurred_img = cv2.GaussianBlur(stego_img, (3, 3), 0)
            try:
                extracted_text, _ = method.extract(blurred_img)
                robustness_tests['blur'] = 1.0 if extracted_text == secret_text else 0.0
            except:
                robustness_tests['blur'] = 0.0
            
            # Tính điểm robustness tổng thể
            robustness_score = np.mean(list(robustness_tests.values()))
            
            return {
                'robustness_score': robustness_score,
                'robustness_tests': robustness_tests
            }
            
        except Exception as e:
            print(f"Lỗi khi test robustness cho {method_name}: {str(e)}")
            return {
                'robustness_score': 0.0,
                'robustness_tests': {}
            }

    def analyze_single_method(self, method, image_path, secret_text, method_name):
        """Phân tích hiệu năng của một phương pháp"""
        try:
            # Nhúng tin
            stego_img, embed_time = method.embed(image_path, secret_text)
            
            # Trích xuất tin
            extracted_text, extract_time = method.extract(stego_img)
            
            # Đọc ảnh gốc để tính metrics
            original_img = cv2.imread(image_path)
            
            # Tính toán metrics cơ bản
            basic_metrics = method.calculate_metrics(original_img, stego_img)
            
            # Tính toán metrics nâng cao
            advanced_metrics = self.calculate_advanced_metrics(original_img, stego_img)
            
            # Test robustness
            robustness_results = self.test_robustness(method, image_path, secret_text, method_name)
            
            # Kiểm tra độ chính xác
            accuracy = 1.0 if extracted_text == secret_text else 0.0
            
            # Tính capacity utilization
            text_bits = len(secret_text) * 8
            capacity_bits = basic_metrics.get('capacity_bits', 0)
            capacity_utilization = (text_bits / capacity_bits * 100) if capacity_bits > 0 else 0
            
            return {
                'method': method_name,
                'embed_time': embed_time,
                'extract_time': extract_time,
                'total_time': embed_time + extract_time,
                'accuracy': accuracy,
                'stego_image': stego_img,
                'extracted_text': extracted_text,
                'capacity_utilization': capacity_utilization,
                'text_bits': text_bits,
                'capacity_bits': capacity_bits,
                # Basic metrics
                'psnr': advanced_metrics['psnr'],
                'ssim': advanced_metrics['ssim'],
                'mse': advanced_metrics['mse'],
                # Advanced metrics
                'sme': advanced_metrics['sme'],
                'mae': advanced_metrics['mae'],
                'rmse': advanced_metrics['rmse'],
                'ncc': advanced_metrics['ncc'],
                # Robustness
                'robustness_score': robustness_results['robustness_score'],
                'robustness_tests': robustness_results['robustness_tests']
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
        """Tạo báo cáo so sánh chi tiết"""
        if not lsb_results or not dct_results:
            return "Không thể tạo báo cáo do lỗi trong quá trình phân tích"
        
        report = "=== BÁO CÁO SO SÁNH HIỆU NĂNG LSB vs DCT STEGANOGRAPHY ===\n\n"
        
        # 1. Thông tin cơ bản
        report += "1. THÔNG TIN CƠ BẢN:\n"
        report += f"   Phương pháp: LSB vs DCT Steganography\n"
        report += f"   Kích thước text: {lsb_results.get('text_bits', 0)} bit\n"
        report += f"   Thời gian phân tích: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 2. Thời gian xử lý
        report += "2. THỜI GIAN XỬ LÝ:\n"
        report += f"   LSB - Nhúng: {lsb_results['embed_time']:.4f}s, Trích xuất: {lsb_results['extract_time']:.4f}s\n"
        report += f"   DCT - Nhúng: {dct_results['embed_time']:.4f}s, Trích xuất: {dct_results['extract_time']:.4f}s\n"
        report += f"   Tổng thời gian LSB: {lsb_results['total_time']:.4f}s\n"
        report += f"   Tổng thời gian DCT: {dct_results['total_time']:.4f}s\n"
        
        # So sánh tốc độ
        speed_ratio = dct_results['total_time'] / lsb_results['total_time']
        if speed_ratio > 1.5:
            report += f"   → LSB nhanh hơn DCT {speed_ratio:.1f} lần\n"
        elif speed_ratio < 0.67:
            report += f"   → DCT nhanh hơn LSB {1/speed_ratio:.1f} lần\n"
        else:
            report += f"   → Tốc độ tương đương\n"
        report += "\n"
        
        # 3. Chất lượng ảnh (PSNR, SSIM)
        report += "3. CHẤT LƯỢNG ẢNH:\n"
        report += f"   LSB - PSNR: {lsb_results['psnr']:.2f} dB, SSIM: {lsb_results['ssim']:.4f}\n"
        report += f"   DCT - PSNR: {dct_results['psnr']:.2f} dB, SSIM: {dct_results['ssim']:.4f}\n"
        
        # Đánh giá chất lượng
        def quality_rating(psnr, ssim):
            if psnr >= 50 and ssim >= 0.99:
                return "Xuất sắc"
            elif psnr >= 40 and ssim >= 0.95:
                return "Rất tốt"
            elif psnr >= 30 and ssim >= 0.90:
                return "Tốt"
            elif psnr >= 20 and ssim >= 0.80:
                return "Trung bình"
            else:
                return "Kém"
        
        report += f"   Đánh giá LSB: {quality_rating(lsb_results['psnr'], lsb_results['ssim'])}\n"
        report += f"   Đánh giá DCT: {quality_rating(dct_results['psnr'], dct_results['ssim'])}\n\n"
        
        # 4. Các chỉ số lỗi chi tiết
        report += "4. CÁC CHỈ SỐ LỖI CHI TIẾT:\n"
        report += f"   MSE (Mean Squared Error):\n"
        report += f"     LSB: {lsb_results['mse']:.6f}\n"
        report += f"     DCT: {dct_results['mse']:.6f}\n"
        report += f"   SME (Squared Mean Error):\n"
        report += f"     LSB: {lsb_results['sme']:.6f}\n"
        report += f"     DCT: {dct_results['sme']:.6f}\n"
        report += f"   MAE (Mean Absolute Error):\n"
        report += f"     LSB: {lsb_results['mae']:.6f}\n"
        report += f"     DCT: {dct_results['mae']:.6f}\n"
        report += f"   RMSE (Root Mean Squared Error):\n"
        report += f"     LSB: {lsb_results['rmse']:.6f}\n"
        report += f"     DCT: {dct_results['rmse']:.6f}\n"
        report += f"   NCC (Normalized Cross Correlation):\n"
        report += f"     LSB: {lsb_results['ncc']:.6f}\n"
        report += f"     DCT: {dct_results['ncc']:.6f}\n\n"
        
        # 5. Dung lượng và hiệu quả
        report += "5. DUNG LƯỢNG VÀ HIỆU QUẢ:\n"
        report += f"   Dung lượng tối đa:\n"
        report += f"     LSB: {lsb_results.get('capacity_bits', 0):,} bit ({lsb_results.get('capacity_bits', 0)//8:,} byte)\n"
        report += f"     DCT: {dct_results.get('capacity_bits', 0):,} bit ({dct_results.get('capacity_bits', 0)//8:,} byte)\n"
        report += f"   Tỷ lệ sử dụng dung lượng:\n"
        report += f"     LSB: {lsb_results.get('capacity_utilization', 0):.2f}%\n"
        report += f"     DCT: {dct_results.get('capacity_utilization', 0):.2f}%\n"
        
        # So sánh dung lượng
        if lsb_results.get('capacity_bits', 0) > dct_results.get('capacity_bits', 0):
            capacity_ratio = lsb_results['capacity_bits'] / dct_results['capacity_bits']
            report += f"   → LSB có dung lượng lớn hơn DCT {capacity_ratio:.1f} lần\n"
        elif dct_results.get('capacity_bits', 0) > lsb_results.get('capacity_bits', 0):
            capacity_ratio = dct_results['capacity_bits'] / lsb_results['capacity_bits']
            report += f"   → DCT có dung lượng lớn hơn LSB {capacity_ratio:.1f} lần\n"
        else:
            report += f"   → Dung lượng tương đương\n"
        report += "\n"
        
        # 6. Độ bền vững (Robustness)
        report += "6. ĐỘ BỀN VỮNG (ROBUSTNESS):\n"
        report += f"   Điểm tổng thể:\n"
        report += f"     LSB: {lsb_results['robustness_score']:.2f}/1.0\n"
        report += f"     DCT: {dct_results['robustness_score']:.2f}/1.0\n"
        
        # Chi tiết các test robustness
        report += f"   Chi tiết các test:\n"
        robustness_tests = ['gaussian_noise', 'salt_pepper', 'jpeg_compression', 'blur']
        test_names = ['Nhiễu Gaussian', 'Nhiễu Salt-Pepper', 'Nén JPEG', 'Làm mờ']
        
        for test, name in zip(robustness_tests, test_names):
            lsb_test = lsb_results['robustness_tests'].get(test, 0)
            dct_test = dct_results['robustness_tests'].get(test, 0)
            report += f"     {name}:\n"
            report += f"       LSB: {'✓' if lsb_test > 0.5 else '✗'} ({lsb_test:.2f})\n"
            report += f"       DCT: {'✓' if dct_test > 0.5 else '✗'} ({dct_test:.2f})\n"
        report += "\n"
        
        # 7. Độ chính xác
        report += "7. ĐỘ CHÍNH XÁC:\n"
        report += f"   LSB: {'✓ Hoàn toàn chính xác' if lsb_results['accuracy'] == 1.0 else '✗ Có lỗi'}\n"
        report += f"   DCT: {'✓ Hoàn toàn chính xác' if dct_results['accuracy'] == 1.0 else '✗ Có lỗi'}\n\n"
        
        # 8. Kết luận tổng thể
        report += "8. KẾT LUẬN TỔNG THỂ:\n"
        
        # Điểm số tổng hợp
        def calculate_overall_score(results):
            # Trọng số: PSNR (30%), SSIM (25%), Robustness (25%), Speed (20%)
            psnr_score = min(results['psnr'] / 50, 1.0) if results['psnr'] != float('inf') else 1.0
            ssim_score = results['ssim']
            robustness_score = results['robustness_score']
            speed_score = 1.0 / (1.0 + results['total_time'])  # Càng nhanh càng tốt
            
            overall = (psnr_score * 0.3 + ssim_score * 0.25 + robustness_score * 0.25 + speed_score * 0.2)
            return overall
        
        lsb_score = calculate_overall_score(lsb_results)
        dct_score = calculate_overall_score(dct_results)
        
        report += f"   Điểm tổng hợp (0-1):\n"
        report += f"     LSB: {lsb_score:.3f}\n"
        report += f"     DCT: {dct_score:.3f}\n"
        
        if lsb_score > dct_score:
            report += f"   → LSB có hiệu năng tổng thể tốt hơn\n"
        elif dct_score > lsb_score:
            report += f"   → DCT có hiệu năng tổng thể tốt hơn\n"
        else:
            report += f"   → Hiệu năng tổng thể tương đương\n"
        
        # Khuyến nghị
        report += f"\n   KHUYẾN NGHỊ:\n"
        if lsb_score > dct_score:
            report += f"   - LSB phù hợp cho ứng dụng cần tốc độ cao và dung lượng lớn\n"
            report += f"   - DCT phù hợp cho ứng dụng cần độ bền vững cao\n"
        else:
            report += f"   - DCT phù hợp cho ứng dụng cần chất lượng và độ bền vững cao\n"
            report += f"   - LSB phù hợp cho ứng dụng cần tốc độ và đơn giản\n"
        
        return report

    def show_metric_charts(self, lsb_results, dct_results):
        """Hiển thị biểu đồ đầy đủ các chỉ số"""
        if not lsb_results or not dct_results:
            print("Không thể hiển thị biểu đồ do thiếu kết quả")
            return
        
        methods = ['LSB', 'DCT']
        
        # Tạo figure với nhiều subplot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('So sánh chi tiết LSB vs DCT Steganography', fontsize=16, fontweight='bold')
        
        # 1. PSNR/SSIM
        x = np.arange(len(methods))
        width = 0.35
        
        psnr_values = [lsb_results['psnr'], dct_results['psnr']]
        ssim_values = [lsb_results['ssim'], dct_results['ssim']]
        
        axes[0, 0].bar(x - width/2, psnr_values, width, label='PSNR (dB)', alpha=0.8, color='#3498db')
        axes[0, 0].bar(x + width/2, ssim_values, width, label='SSIM', alpha=0.8, color='#e74c3c')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods)
        axes[0, 0].set_title('Chất lượng ảnh (PSNR/SSIM)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Hiển thị giá trị trên cột
        for i, (psnr, ssim) in enumerate(zip(psnr_values, ssim_values)):
            axes[0, 0].text(i - width/2, psnr + 0.5, f'{psnr:.1f}', ha='center', va='bottom', fontsize=9)
            axes[0, 0].text(i + width/2, ssim + 0.01, f'{ssim:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Các chỉ số lỗi
        mse_values = [lsb_results['mse'], dct_results['mse']]
        mae_values = [lsb_results['mae'], dct_results['mae']]
        rmse_values = [lsb_results['rmse'], dct_results['rmse']]
        
        x_pos = np.arange(len(methods))
        width = 0.25
        
        axes[0, 1].bar(x_pos - width, mse_values, width, label='MSE', alpha=0.8, color='#f39c12')
        axes[0, 1].bar(x_pos, mae_values, width, label='MAE', alpha=0.8, color='#9b59b6')
        axes[0, 1].bar(x_pos + width, rmse_values, width, label='RMSE', alpha=0.8, color='#e67e22')
        
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(methods)
        axes[0, 1].set_title('Các chỉ số lỗi')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')  # Dùng log scale cho các chỉ số lỗi
        
        # 3. Thời gian xử lý
        embed_times = [lsb_results['embed_time'], dct_results['embed_time']]
        extract_times = [lsb_results['extract_time'], dct_results['extract_time']]
        
        axes[0, 2].bar(x - width/2, embed_times, width, label='Nhúng', alpha=0.8, color='#2ecc71')
        axes[0, 2].bar(x + width/2, extract_times, width, label='Trích xuất', alpha=0.8, color='#e74c3c')
        
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(methods)
        axes[0, 2].set_title('Thời gian xử lý (s)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Hiển thị giá trị
        for i, (emb, ext) in enumerate(zip(embed_times, extract_times)):
            axes[0, 2].text(i - width/2, emb + 0.001, f'{emb:.3f}', ha='center', va='bottom', fontsize=9)
            axes[0, 2].text(i + width/2, ext + 0.001, f'{ext:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Dung lượng
        capacity_values = [lsb_results.get('capacity_bits', 0)/8, dct_results.get('capacity_bits', 0)/8]
        utilization_values = [lsb_results.get('capacity_utilization', 0), dct_results.get('capacity_utilization', 0)]
        
        ax2 = axes[1, 0].twinx()
        bars1 = axes[1, 0].bar(x, capacity_values, alpha=0.7, color='#3498db', label='Dung lượng (byte)')
        bars2 = ax2.bar(x + 0.4, utilization_values, alpha=0.7, color='#e74c3c', label='Tỷ lệ sử dụng (%)')
        
        axes[1, 0].set_xticks(x + 0.2)
        axes[1, 0].set_xticklabels(methods)
        axes[1, 0].set_title('Dung lượng và hiệu quả')
        axes[1, 0].set_ylabel('Dung lượng (byte)', color='#3498db')
        ax2.set_ylabel('Tỷ lệ sử dụng (%)', color='#e74c3c')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Hiển thị giá trị
        for i, (cap, util) in enumerate(zip(capacity_values, utilization_values)):
            axes[1, 0].text(i, cap + max(capacity_values)*0.01, f'{cap:.0f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i + 0.4, util + max(utilization_values)*0.01, f'{util:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 5. Robustness
        robustness_scores = [lsb_results['robustness_score'], dct_results['robustness_score']]
        robustness_tests = ['gaussian_noise', 'salt_pepper', 'jpeg_compression', 'blur']
        test_names = ['Gaussian', 'Salt-Pepper', 'JPEG', 'Blur']
        
        # Tạo biểu đồ radar cho robustness
        angles = np.linspace(0, 2 * np.pi, len(robustness_tests), endpoint=False).tolist()
        angles += angles[:1]  # Đóng vòng tròn
        
        lsb_robustness = [lsb_results['robustness_tests'].get(test, 0) for test in robustness_tests] + [lsb_results['robustness_tests'].get(robustness_tests[0], 0)]
        dct_robustness = [dct_results['robustness_tests'].get(test, 0) for test in robustness_tests] + [dct_results['robustness_tests'].get(robustness_tests[0], 0)]
        
        axes[1, 1].plot(angles, lsb_robustness, 'o-', linewidth=2, label='LSB', color='#3498db')
        axes[1, 1].fill(angles, lsb_robustness, alpha=0.25, color='#3498db')
        axes[1, 1].plot(angles, dct_robustness, 'o-', linewidth=2, label='DCT', color='#e74c3c')
        axes[1, 1].fill(angles, dct_robustness, alpha=0.25, color='#e74c3c')
        
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(test_names)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Độ bền vững (Robustness)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Tổng hợp điểm số
        def calculate_overall_score(results):
            psnr_score = min(results['psnr'] / 50, 1.0) if results['psnr'] != float('inf') else 1.0
            ssim_score = results['ssim']
            robustness_score = results['robustness_score']
            speed_score = 1.0 / (1.0 + results['total_time'])
            return (psnr_score * 0.3 + ssim_score * 0.25 + robustness_score * 0.25 + speed_score * 0.2)
        
        overall_scores = [calculate_overall_score(lsb_results), calculate_overall_score(dct_results)]
        
        bars = axes[1, 2].bar(methods, overall_scores, color=['#f39c12', '#9b59b6'], alpha=0.8)
        axes[1, 2].set_title('Điểm tổng hợp (0-1)')
        axes[1, 2].set_ylabel('Điểm số')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Hiển thị giá trị trên cột
        for bar, score in zip(bars, overall_scores):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
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
