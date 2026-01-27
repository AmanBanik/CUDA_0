# config.py (Refined for Stability)

# --- Dimensions ---
# 1920x1080 is fine, but if it feels sluggish, try 1280x720
WIDTH = 1920
HEIGHT = 1080

# 720p for better performance on mid-range GPUs
# WIDTH = 1280
# HEIGHT = 720

# --- Dimensions (True 4K) ---
# 8.3 Million Pixels.
# Note: If this is too big for the monitor, Pygame might crop it.
# WIDTH = 3840
# HEIGHT = 2160

# If so, drop to QHD: 2560 x 1600
# WIDTH = 2560
# HEIGHT = 1600
# --- CUDA Config ---
# Threads per Block: 16x16 is still the sweet spot
TPB = 16

# --- Physics Constants (Robust Coral) ---
# We use the standard "Coral" spot, but with the new time step, 
# it will be much more stable.
Du = 1.0
Dv = 0.5
FEED = 0.0545
KILL = 0.0620

# --- Stability Settings ---
# dt: Time step. 1.0 is fast but unstable (causes bursts).
# 0.2 is "High Precision Mode".
dt = 0.2 

# Steps per frame:
# Since dt is 5x smaller, we need 5x more steps to keep the same visual speed.
# Try 30-40. If GPU lags, lower this to 20.
# Choose between 24 - 32 steps for smooth real-time performance at higher resolutions.
STEPS_PER_FRAME = 30

# Brush
# BRUSH_RADIUS = 10
BRUSH_RADIUS = 25
# Toggle to increase brush size for high-res modes
# BRUSH_RADIUS = 40

# UPDATES for V2

# --- VISUALIZATION SETTINGS ---
# 1.0 = Original. 0.9 = Thinner. <0.8 = Very thin/Skeleton-like.
THICKNESS_MODIFIER = 0.9

# The available colors to cycle through with 'T'
# Format: (Red, Green, Blue) normalized 0.0 - 1.0
COLOR_PALETTE = [
    (0.0, 1.0, 1.0), # 0: Cyan (Default)
    (1.0, 0.2, 0.1), # 1: Neon Red/Orange
    (0.2, 1.0, 0.2), # 2: Radioactive Green
    (0.5, 0.0, 1.0), # 3: Deep Purple
    (1.0, 1.0, 0.0), # 4: Lemon Yellow
    (1.0, 1.0, 1.0), # 5: Pure White
    (1.0, 0.0, 0.5), # 6: Hot Pink
]