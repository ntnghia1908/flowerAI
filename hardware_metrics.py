# Code thường nằm trong file trainer.py
# 1. IMPORT (Cần cài: pip install psutil pynvml)
import psutil   # Để đo CPU, RAM
import pynvml   # Để đo GPU (NVIDIA Management Library)

# 2. TRONG CLASS TRAINER (__init__)
# Code khởi tạo: 
def __init__(self, ...):
        super().__init__()
        # ... các code khởi tạo khác ...
        # --- KHỞI TẠO HARDWARE MONITOR ---
        # Lấy process ID hiện tại để đo RAM/CPU của riêng tiến trình này
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent() # Gọi lần đầu để làm mốc (tránh trả về 0.0)
        # Thử khởi tạo thư viện quản lý GPU NVIDIA
        try:
            pynvml.nvmlInit()
            self.has_gpu_mon = True
        except: 
            # Nếu máy không có GPU NVIDIA hoặc chưa cài driver
            self.has_gpu_mon = False

# 3. HÀM TÍNH TOÁN (Thêm vào class Trainer)
def _get_system_metrics(self):
    # """
    # Trả về: gpu_temp, gpu_util, vram_gb, ram_gb, cpu_util
    # """
    # 1. Đo CPU & RAM (Chỉ của Process này)
    ram_gb = self.process.memory_info().rss / (1024**3) # Đổi Byte -> GB
    cpu_util = self.process.cpu_percent()
    # 2. Đo GPU (Toàn hệ thống)
    gpu_temp, gpu_util, vram_gb = 0, 0, 0
    if self.has_gpu_mon:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Lấy GPU đầu tiên
            # Lấy Util, Temp, VRAM Used
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util, gpu_temp = util.gpu, pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            vram_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)
        except: pass
    # Fallback cho VRAM nếu pynvml lỗi (nhưng chỉ đo được vram của process)
    elif torch.cuda.is_available():
        vram_gb = torch.cuda.memory_reserved(self.device) / (1024**3)
    return gpu_temp, gpu_util, vram_gb, ram_gb, cpu_util

# 4. CÁCH GỌI & GHI LOG (Trong vòng lặp train)
# Lấy số liệu
gpu_temp, gpu_util, vram_gb, ram_gb, cpu_util = self._get_system_metrics()
# Ghi vào CSV
w.writerow([
    # ... các cột khác ...
    f"{cpu_util:.1f}", f"{ram_gb:.2f}", f"{gpu_temp}", f"{gpu_util}", f"{vram_gb:.2f}"
])