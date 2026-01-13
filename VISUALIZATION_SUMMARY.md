# Tóm Tắt Các Biểu Đồ Đã Tạo

## Thông tin chung
- **Số lượng biểu đồ**: 7 biểu đồ chất lượng cao
- **Độ phân giải**: 300 DPI (publication-ready)
- **Thư mục lưu trữ**: `figures/`
- **Định dạng**: PNG

---

## 1. Best Accuracy Matrix (Transposed) ✅
**File**: `figures/01_best_accuracy_matrix_transposed.png`

### Mô tả:
- **Ma trận nhiệt** hiển thị độ chính xác tốt nhất (Best Accuracy) của từng thuật toán trên từng phân phối dữ liệu
- **Trục Y (Rows)**: 6 Thuật toán (FedAdagrad, FedAdam, FedAvg, FedAvgM, FedProx, FedYogi)
- **Trục X (Columns)**: 9 Phân phối dữ liệu (homo, Dir10.0, Dir1.0, Dir0.5, C5, C4, Dir0.1, C3, C2)
- **Màu sắc**: Từ vàng (thấp) đến đỏ (cao) - Gradient RdYlGn
- **Highlight**: Ô xanh viền đậm = thuật toán thắng trên phân phối đó

### Insights chính:
- FedProx và FedAvg thống trị với nhiều chiến thắng
- Phân phối homo và Dir10.0 cho kết quả tốt nhất
- Phân phối C2 khó nhất cho tất cả thuật toán

---

## 2. Final Round Accuracy Matrix (Transposed) ✅
**File**: `figures/02_final_accuracy_matrix_transposed.png`

### Mô tả:
- **Ma trận nhiệt** hiển thị độ chính xác ở round cuối cùng (round 500)
- **Trục Y (Rows)**: 6 Thuật toán
- **Trục X (Columns)**: 9 Phân phối dữ liệu
- **Màu sắc**: RdYlGn gradient (42-60%)
- **Highlight**: Ô xanh viền đậm = thuật toán hội tụ tốt nhất

### Insights chính:
- FedAdagrad hội tụ tốt hơn dự đoán (5/9 distributions)
- Gap giữa Best và Final cho thấy tính ổn định của thuật toán
- FedAvg có độ hội tụ ổn định nhất

---

## 3. Convergence Speed Heatmap ✅
**File**: `figures/03_convergence_speed_heatmap.png`

### Mô tả:
- **Ma trận nhiệt** hiển thị số round đạt được best accuracy
- **Trục Y (Rows)**: 6 Thuật toán
- **Trục X (Columns)**: 9 Phân phối dữ liệu
- **Màu sắc**: Đỏ-Vàng-Xanh (Càng xanh = hội tụ càng nhanh)
- **Số hiển thị**: Số round (màu xanh = <50, màu đỏ = >300)

### Insights chính:
- C2 distribution cực kỳ chậm hội tụ (369-500 rounds)
- FedAvgM nhanh nhất (trung bình 96 rounds)
- FedAdagrad chậm nhưng tiếp tục cải thiện muộn

---

## 4. Algorithm Performance Comparison ✅
**File**: `figures/04_algorithm_comparison.png`

### Mô tả:
3 biểu đồ cột ngang so sánh hiệu suất thuật toán:
1. **Average Best Accuracy** - Độ chính xác tốt nhất trung bình
2. **Average Final Accuracy** - Độ chính xác round cuối trung bình
3. **Maximum Accuracy** - Độ chính xác đỉnh cao nhất

### Màu sắc:
- 🟢 Xanh: Hiệu suất tốt
- 🟠 Cam: Hiệu suất trung bình
- 🔴 Đỏ: Hiệu suất thấp

### Insights chính:
- FedProx: Highest average (58.78%) và peak (64.77%)
- FedAvg: Best final convergence (53.48%)
- FedAdagrad: Third best overall (56.47%)

---

## 5. Distribution Difficulty Ranking ✅
**File**: `figures/05_distribution_difficulty.png`

### Mô tả:
2 biểu đồ phân tích độ khó của phân phối dữ liệu:
1. **Difficulty Ranking** - Xếp hạng từ dễ đến khó
2. **Best vs Final Comparison** - So sánh gap giữa best và final

### Độ khó:
- 🟢 **Easy**: Dir10.0 (60.76%), homo (59.94%), Dir1.0 (59.62%)
- 🟡 **Medium**: Dir0.5, C5, C4, Dir0.1
- 🔴 **Hard**: C3 (51.22%), C2 (48.90%)

### Insights chính:
- Dir10.0 dễ nhất với độ phân tán dữ liệu thấp
- C2 khó nhất (chỉ 2 class/client)
- Gap lớn = huấn luyện không ổn định

---

## 6. Top 10 Combinations ✅
**File**: `figures/06_top10_combinations.png`

### Mô tả:
- **Biểu đồ cột ngang** hiển thị 10 kết hợp thuật toán-phân phối tốt nhất
- **Màu sắc**: Mỗi thuật toán có màu riêng
- **Top 3**: Đánh dấu #1 (vàng), #2 (bạc), #3 (đồng)

### Top 3:
1. 🥇 **FedProx + homo**: 64.77%
2. 🥈 **FedAvg + homo**: 63.64%
3. 🥉 **FedAvg + Dir10.0**: 63.24%

### Insights chính:
- FedProx và FedAvg thống trị top 10
- Phân phối Dirichlet (Dir*) xuất hiện nhiều
- homo distribution cho kết quả tốt nhất

---

## 7. Comprehensive Algorithm vs Distribution Overview ✅
**File**: `figures/07_algorithm_vs_distribution_overview.png`

### Mô tả:
- **Biểu đồ cột nhóm** hiển thị so sánh 6 thuật toán trên 9 phân phối
- **Trục X**: 9 Phân phối (sắp xếp từ dễ → khó)
- **Trục Y**: Best Accuracy (%)
- **Vùng màu nền**: Xanh (Easy), Vàng (Medium), Đỏ (Hard)

### Insights chính:
- Xu hướng giảm rõ ràng từ trái sang phải (dễ → khó)
- FedProx và FedAvg luôn ở top
- Tất cả thuật toán đều thấp trên C2

---

## Hướng dẫn sử dụng

### Xem biểu đồ:
```bash
# Mở thư mục figures
cd c:\Users\DESKSTOP_003\Desktop\flowerAI\figures

# Các file PNG có thể xem bằng bất kỳ trình xem ảnh nào
```

### Tái tạo biểu đồ:
```bash
# Chạy lại script
python visualize_accuracy_report.py
```

### Chỉnh sửa biểu đồ:
Mở file [visualize_accuracy_report.py](visualize_accuracy_report.py) và chỉnh sửa:
- **Màu sắc**: Thay đổi `cmap` parameter
- **Kích thước**: Thay đổi `figsize`
- **Độ phân giải**: Thay đổi `dpi`
- **Font**: Thay đổi `plt.rcParams['font.size']`

---

## Khuyến nghị cho báo cáo

### Cho phần Abstract/Summary:
- Sử dụng **Figure 4** (Algorithm Comparison) để tóm tắt kết quả tổng quan
- Sử dụng **Figure 6** (Top 10) để highlight best results

### Cho phần Detailed Analysis:
- Sử dụng **Figure 1 & 2** (Heatmaps) để so sánh chi tiết
- Sử dụng **Figure 3** (Convergence) để phân tích tốc độ hội tụ
- Sử dụng **Figure 5** (Distribution) để phân tích độ khó

### Cho phần Conclusion:
- Sử dụng **Figure 7** (Overview) để tổng kết xu hướng
- Sử dụng **Figure 6** (Top 10) để khuyến nghị

---

## Thống kê kỹ thuật

### Dữ liệu đầu vào:
- **Số thuật toán**: 6
- **Số phân phối**: 9
- **Tổng số thí nghiệm**: 54
- **Số rounds mỗi thí nghiệm**: 500
- **Tổng số rounds**: 27,000

### Độ phân giải biểu đồ:
- **DPI**: 300 (publication-ready)
- **Format**: PNG
- **Color**: RGB
- **Style**: Seaborn whitegrid

### Thư viện sử dụng:
- `matplotlib`: Visualization
- `seaborn`: Statistical plotting
- `numpy`: Numerical computation
- `pandas`: Data manipulation

---

## Tóm tắt kết quả chính

### Thuật toán thắng cuộc:
1. 🥇 **FedProx**: 58.78% avg, 64.77% peak
2. 🥈 **FedAvg**: 58.70% avg, best convergence
3. 🥉 **FedAdagrad**: 56.47% avg, best late-game

### Phân phối dễ nhất:
1. Dir10.0: 60.76%
2. homo: 59.94%
3. Dir1.0: 59.62%

### Phân phối khó nhất:
1. C2: 48.90%
2. C3: 51.22%
3. Dir0.1: 51.84%

### Kết hợp tốt nhất:
**FedProx + homo = 64.77%**

---

**Ngày tạo**: 2026-01-13
**Script**: `visualize_accuracy_report.py`
**Dữ liệu nguồn**: `FULL_ACCURACY_MATRIX_REPORT.md`
