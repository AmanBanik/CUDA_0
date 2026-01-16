import os
import sys
import math
import platform
import subprocess
import time
import json
import datetime
import warnings
import ctypes
import colorsys
import numpy as np
from numba import cuda, jit

# 1. SUPPRESS AVX2 WARNING
# This warning is harmless on your system; suppressing to keep CLI clean.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")
import pygame

# -----------------------------
# 2. HARDWARE & RESOLUTION DETECTION
# -----------------------------
def get_screen_resolution_windows():
    """Get the actual physical screen resolution on Windows."""
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        return w, h
    except:
        return 1920, 1080 # Fallback

def get_system_refresh_rate():
    """Detect actual system refresh rate."""
    refresh_rate = 60
    system = platform.system()
    try:
        if system == "Windows":
            try:
                import win32api, win32con
                device = win32api.EnumDisplayDevices()
                settings = win32api.EnumDisplaySettings(device.DeviceName, win32con.ENUM_CURRENT_SETTINGS)
                if settings.DisplayFrequency > 0:
                    refresh_rate = settings.DisplayFrequency
            except ImportError:
                cmd = '(Get-CimInstance Win32_VideoController | Select-Object -First 1).CurrentRefreshRate'
                result = subprocess.run(['powershell', '-Command', cmd], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip().isdigit():
                    refresh_rate = int(result.stdout.strip())
    except:
        pass
    return max(30, min(refresh_rate, 240))

# -----------------------------
# 3. USER INPUTS
# -----------------------------
def ask_user_mode():
    print("\n" + "="*50)
    print(" GARGANTUA: CUDA V2")
    print("="*50)
    print(">> Choose your mode:")
    print("   1. Default (Auto-detect Resolution & Refresh Rate)")
    print("   2. Custom (Choose your own resolution and target fps)")
    
    while True:
        choice = input("\nSelection [1/2]: ").strip()
        if choice in ['1', '2']: return choice
        print("Invalid selection.")

def ask_resolution():
    print("\n>> Select Resolution:")
    print("   1. 720p  (1280 x 720)")
    print("   2. 1080p (1920 x 1080)")
    print("   3. 1200p (1920 x 1200)")
    print("   4. 1440p (2560 x 1440)")
    print("   5. 1600p (2560 x 1600)")
    print("   6. 2160p (3840 x 2160)")
    resolutions = {'1':(1280,720), '2':(1920,1080), '3':(1920,1200),
                   '4':(2560,1440), '5':(2560,1600), '6':(3840,2160)}
    while True:
        c = input("Selection [1-6]: ").strip()
        if c in resolutions: return resolutions[c]
        print("Invalid choice.")

def ask_fps():
    print("\n>> Select Target FPS:")
    print("   1. 30 FPS")
    print("   2. 60 FPS")
    print("   3. 90 FPS")
    print("   4. 144 FPS")
    print("   5. 165 FPS")
    print("   6. 240 FPS")
    options = {'1':30, '2':60, '3':90, '4':144, '5':165, '6':240}
    while True:
        c = input("Selection [1-6]: ").strip()
        if c in options: return options[c]
        print("Invalid choice.")

def get_configuration():
    mode = ask_user_mode()
    if mode == '2':
        width, height = ask_resolution()
        target_fps = ask_fps()
    else:
        # Default Mode: Smart Auto-detect
        target_fps = get_system_refresh_rate()
        if platform.system() == "Windows":
            width, height = get_screen_resolution_windows()
        else:
            width, height = 1920, 1080
            
        print(f"\n[INFO] Auto-detected: {width}x{height} @ {target_fps}Hz")
        
    return width, height, target_fps

# -----------------------------
# 4. MONITORING & LOGGING
# -----------------------------
def log_performance(json_path, log_buffer):
    """Appends buffered logs to the JSON file."""
    try:
        # Load existing or create new
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
            
        data.extend(log_buffer)
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    except Exception as e:
        print(f"[LOG ERROR] Could not write to JSON: {e}")

# -----------------------------
# 5. CUDA KERNEL (UPDATED PHYSICS)
# -----------------------------
@cuda.jit
def compute_points_kernel(A, B, disk_inner, disk_outer, points_out, colors_out, 
                          phi_steps, theta_steps, lensing_steps):
    idx = cuda.grid(1)
    total_disk_points = phi_steps * theta_steps
    total_lensing_points = lensing_steps
    total_points = total_disk_points + total_lensing_points

    if idx >= total_points:
        return

    # Larger Black Hole
    schwarzschild_radius = 3.5 
    
    x, y, z = 0.0, 0.0, 0.0
    lum = 0.0
    hue = 0.0
    
    # --- DISK PARTICLES ---
    if idx < total_disk_points:
        phi_idx = idx // theta_steps
        theta_idx = idx % theta_steps
        
        phi_raw = phi_idx * (6.28318 / phi_steps)
        theta_raw = theta_idx * (6.28318 / theta_steps)

        # Geometry
        radius = disk_inner + (disk_outer - disk_inner) * (phi_idx / phi_steps)
        x = radius * math.cos(theta_raw)
        y = 0.2 * math.sin(theta_raw * 3) * math.sin(phi_raw * 2)
        z = radius * math.sin(theta_raw)
        
        velocity = math.sin(theta_raw) * math.cos(A)
        dist_center = radius
        
        # Rotate
        cos_A, sin_A = math.cos(A), math.sin(A)
        x_new = x * cos_A - z * sin_A
        z_new = x * sin_A + z * cos_A
        x, z = x_new, z_new
        
        cos_B, sin_B = math.cos(B), math.sin(B)
        y_new = y * cos_B - z * sin_B
        z_new = y * sin_B + z * cos_B
        y, z = y_new, z_new
        
        if dist_center < schwarzschild_radius * 1.1:
            lum = 0.0
        else:
            base_lum = 1.0 - (dist_center - disk_inner) / (disk_outer - disk_inner)
            if base_lum < 0: base_lum = 0
            
            doppler = 1.0 + velocity * 0.4
            lum = base_lum * doppler
            
            # COLOR LOGIC 
            # We just pass base luminance, we will handle color mapping in Python pre-render
            # But we can shift hue slightly here to separate "hot" inner from "cold" outer
            hue = 0.05 + (1.0 - base_lum) * 0.1 # 0.05 (Orange) -> 0.15 (Yellowish)

    # --- LENSING RING ---
    else:
        l_idx = idx - total_disk_points
        angle = l_idx * (6.28318 / lensing_steps)
        
        ring_r = schwarzschild_radius * 1.6
        x = ring_r * math.cos(angle)
        y = schwarzschild_radius * 0.25 * math.sin(angle * 2)
        z = ring_r * math.sin(angle)
        
        cos_A, sin_A = math.cos(A), math.sin(A)
        x_new = x * cos_A - z * sin_A
        z_new = x * sin_A + z * cos_A
        x, z = x_new, z_new
        
        lum = 1.0
        hue = 0.0 # Red/White ring

    points_out[idx, 0] = x
    points_out[idx, 1] = y
    points_out[idx, 2] = z
    colors_out[idx, 0] = hue
    colors_out[idx, 1] = lum 

# -----------------------------
# 6. CPU RASTERIZER
# -----------------------------
@jit(nopython=True)
def rasterize_points(points, colors, rows, cols, x_off, y_off, chars_len):
    screen_indices = np.full((rows, cols), -1, dtype=np.int32)
    z_buffer = np.zeros((rows, cols), dtype=np.float32)
    num_points = points.shape[0]
    
    for i in range(num_points):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        dist = z + 8.0 # Increased camera distance slightly for bigger BH fit
        if dist > 0:
            D = 1.0 / dist
            sx = int(x_off + 30 * D * x)
            sy = int(y_off + 20 * D * y)
            if 0 <= sx < cols and 0 <= sy < rows:
                if D > z_buffer[sy, sx]:
                    z_buffer[sy, sx] = D
                    lum = colors[i, 1]
                    c_idx = int(lum * chars_len)
                    if c_idx >= chars_len: c_idx = chars_len - 1
                    if c_idx < 0: c_idx = 0
                    screen_indices[sy, sx] = c_idx
    return screen_indices

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    WIDTH, HEIGHT, TARGET_FPS = get_configuration()
    pygame.init()
    
    # Font & Grid
    font_size = 14
    x_separator = 10 
    y_separator = 18 
    rows = HEIGHT // y_separator
    columns = WIDTH // x_separator
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Gargantua CUDA v2 - {WIDTH}x{HEIGHT}')
    font = pygame.font.SysFont('Courier New', font_size, bold=True)
    
    # UPDATED: Color Palette (Vibrant Orange/Gold)
    disk_chars = ".,-~:;=!*#$@%&"
    char_surfaces = []
    for i, char in enumerate(disk_chars):
        # Normalized intensity (0.0 to 1.0)
        norm_i = (i + 1) / len(disk_chars)
        
        # Logic: 
        # Hue: 0.02 (Red-Orange) -> 0.12 (Yellow-Gold)
        # Sat: High (0.95) -> Drops slightly for brightest (0.6) to simulate white-hot
        # Val: Increases linearly
        
        h = 0.02 + (norm_i * 0.08) 
        s = 1.0 - (norm_i * 0.4)
        v = 0.5 + (norm_i * 0.5) # Boosted base brightness
        
        rgb = colorsys.hsv_to_rgb(h, s, v)
        color = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        char_surfaces.append(font.render(char, True, color))

    # CUDA Setup
    phi_points = 350   
    theta_points = 120 
    lensing_points = 400
    total_points = (phi_points * theta_points) + lensing_points
    
    d_points = cuda.device_array((total_points, 3), dtype=np.float32)
    d_colors = cuda.device_array((total_points, 2), dtype=np.float32)
    
    threads_per_block = 256
    blocks = (total_points + (threads_per_block - 1)) // threads_per_block

    # State variables
    A, B = 0.0, 0.0
    fps_factor = 60 / TARGET_FPS
    h_speed = 0.01 * fps_factor
    v_speed = 0.005 * fps_factor

    clock = pygame.time.Clock()
    running = True

    # Logging setup
    json_filename = "session_log.json"
    log_buffer = []
    last_report_time = time.time()
    report_interval = 4.0 # Seconds

    print(f"\n[GPU] Simulation started. Monitoring active (every {report_interval}s).")
    print(f"[LOG] Writing metrics to {json_filename}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # 1. Compute
        compute_points_kernel[blocks, threads_per_block](
            A, B, 3.5, 9.0, d_points, d_colors, phi_points, theta_points, lensing_points
        )
        h_points = d_points.copy_to_host()
        h_colors = d_colors.copy_to_host()

        # 2. Rasterize
        grid_indices = rasterize_points(
            h_points, h_colors, rows, columns, 
            columns/2, rows/2, len(disk_chars)
        )

        # 3. Draw
        screen.fill((0, 0, 0))
        for r in range(rows):
            for c in range(columns):
                char_idx = grid_indices[r, c]
                if char_idx != -1:
                    screen.blit(char_surfaces[char_idx], (c * x_separator, r * y_separator))

        pygame.display.flip()
        
        # 4. Updates & Monitoring
        A += h_speed
        B += v_speed
        clock.tick(TARGET_FPS)
        
        # Periodic Reporting
        current_time = time.time()
        if current_time - last_report_time >= report_interval:
            actual_fps = clock.get_fps()
            # Log Data
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "target_fps": TARGET_FPS,
                "actual_fps": round(actual_fps, 2),
                "resolution": f"{WIDTH}x{HEIGHT}",
                "particles": total_points
            }
            log_buffer.append(entry)
            log_performance(json_filename, log_buffer)
            log_buffer = [] # Clear buffer after write
            
            # Console Output
            print(f"[MONITOR] FPS: {actual_fps:.1f} / {TARGET_FPS} | Res: {WIDTH}x{HEIGHT}")
            last_report_time = current_time

    pygame.quit()

if __name__ == "__main__":
    main()