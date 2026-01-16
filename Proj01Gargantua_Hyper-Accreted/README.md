# 🌌 Gargantua: Hyper-Accreted (CUDA Edition)

> *"The only thing faster than light is the speed of compute."* (Derived from PyJokes🤡)

![Sigma Laugh](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXk2d2Ftcno2cHIwN2EwY3VweDBndng0Y2o4dHNobThpajF4bm96aSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/UpobWd0mSpRfO/giphy.gif)

A high-fidelity, GPU-accelerated black hole visualization. This project is a massive evolution of the original CPU-based renderer, leveraging **NVIDIA CUDA** architecture to simulate relativistic physics in real-time. It renders a Schwarzschild black hole with an uncapped 4K resolution, scientifically informed accretion disk, and gravitational lensing effects at 144Hz+ frame rates.

![Black Hole](https://img.shields.io/badge/Simulation-Black%20Hole-orange) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.0+-green) ![Auto-FPS](https://img.shields.io/badge/Auto--FPS-Adaptive-brightgreen) ![NVIDIA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white)

## ✨ The "Hyper-Accreted" Upgrade

Moving from CPU loops to GPU Kernels changed everything. This version removes the event horizon of "performance caps" found in the legacy CPU version.

| Feature | Legacy (CPU) | **Hyper-Accreted (GPU)** |
| :--- | :--- | :--- |
| **Architecture** | Sequential `math` loops | **Parallel CUDA Kernels** |
| **Particle Count** | ~2,000 | **42,400+** (20x Density) |
| **Frame Rate** | ~9 FPS (Struggling) | **~145 FPS** (Silky Smooth) |
| **Resolution** | 1080p (Capped) | **UNLOCKED** (1200p / 1440p / 4K) |
| **Color Palette** | Basic Blue/Red | **Dynamic "Magma" Gradient** |
| **Hardware** | Basic CPU | **NVDIA RTX CUDA Ready GPUs** |

## ✨ Features

### 🔭 Realistic Physics & Visual Features
- **High-Density Accretion Disk**: 40,000+ individual points of light computed per frame.
- **Relativistic Beaming**: Real-time Doppler shifting (approaching side is brighter/bluer).
- **Gravitational Lensing**: Accurate "Einstein Ring" geometry bending light around the **event horizon**.
- **Gravitational Lensing**: Einstein ring effects showing light bending around spacetime
- **"Magma" Shader**: A new gold-orange-white gradient simulating extreme thermal radiation.
- **Z-axis Tilting**: Slow vertical rotation for dynamic perspective

### 🖥️ Intelligent CLI & Logging
- **Interactive Boot Menu**: Choose between "Auto-Detect" or "Custom Overclock" modes.
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Session Logger**: Automatically dumps performance metrics (FPS, Frame Time, Particle Count) to `session_log.json` for analysis.
- **Resolution Unlocked**: Native support for 16:10 aspect ratios (1920x1200) and full 4K.

### 🎨 ASCII Art Rendering
- **Multi-tier Character Sets**: Different symbols for various regions and intensities
- **Dynamic Lighting**: Real-time luminance calculations
- **Perspective Projection**: 3D-to-2D conversion with proper depth handling

## 🚀 Installation , Setup & Testing

**Recommendation:** Use **Conda**.
*Note: We strictly avoid pure `pip` installations for system-level dependencies like `pywin32` or `cudatoolkit` to prevent DLL conflicts and driver mismatches.*

### 1. Prerequisites
You need an NVIDIA GPU (RTX 30/40/50 Series) and the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed.

### 2. Conda Environment Setup
```bash
# Create a clean environment
conda create -n gpu_env
conda activate gpu_env

# Install CUDA Toolkit and Numba (The Engine)
conda install cudatoolkit numba numpy

# Suggested hard-way ~ fetching compilers
conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=13.0"

# Install System API (Critical for Refresh Rate Detection)
# We use Conda here to avoid the common "DLL load failed" errors seen with pip
conda install pywin32

# Install Pygame (The Display Layer)
conda install -c conda-forge pygame
pip install pygame # for verification
```

### 3. Fetch the project
```bash
git clone https://github.com/AmanBanik/CUDA_0.git
```
and load it in your `gpu_env` directory (where the env was created)

### 4. Running the Simulation
```
python -u "main_blackwell_02.py"
```
### 5. The Boot Menu

Upon launching, the terminal will request your configuration:
```
==================================================
 GARGANTUA: CUDA V2
==================================================
>> Choose your mode:
   1. Default (Auto-detect Resolution & Refresh Rate)
   2. Custom (Choose your own resolution and target fps)
```
*Select Option 2 to force custom presets (till 4K 240Hz rendering).*

**⚠️ Please consider checking your hardware limitations first**
### 6. Expected Output
*Would varry depending upon hardware capability and presets*
```
[INFO] Auto-detected: 1920x1200 @ 165Hz

[GPU] Simulation started. Monitoring active (every 4.0s).
[LOG] Writing metrics to session_log.json
[MONITOR] FPS: 153.8 / 165 | Res: 1920x1200
[MONITOR] FPS: 151.5 / 165 | Res: 1920x1200
[MONITOR] FPS: 147.1 / 165 | Res: 1920x1200
[MONITOR] FPS: 120.5 / 165 | Res: 1920x1200
[MONITOR] FPS: 149.3 / 165 | Res: 1920x1200
[MONITOR] FPS: 128.2 / 165 | Res: 1920x1200
[MONITOR] FPS: 140.8 / 165 | Res: 1920x1200
[MONITOR] FPS: 138.9 / 165 | Res: 1920x1200
[MONITOR] FPS: 142.9 / 165 | Res: 1920x1200
[MONITOR] FPS: 133.3 / 165 | Res: 1920x1200
[MONITOR] FPS: 137.0 / 165 | Res: 1920x1200
                   *
                   *
                   *
                   *
```
### 7. Controls
- **ESC**: Exit simulation
- **Close Window**: Standard window close
- **Forced**: `ctrl + c` in terminal

## 🛠️ Technical Details
### Dependencies
- **Python 3.10+**
- **Pygame 2.0+**
- **Built-in modules**: `math`, `colorsys`, `platform`, `subprocess`, `time`
- **Optional**: `pywin32` (Windows - for better refresh rate detection)

### In-code Configuration
The simulation can be customized by modifying these parameters:

*example:*
```python
# Display Settings
WIDTH, HEIGHT = 1920, 1080
theta_spacing, phi_spacing = 2, 4  # Rendering detail
font_size = 14

# Black Hole Physics
disk_inner_radius = 2.5   # Inner accretion disk boundary
disk_outer_radius = 8.0   # Outer accretion disk boundary
schwarzschild_radius = 2.0 # Event horizon size

# Character spacing
x_separator, y_separator = 8, 16
```
### Character Sets
```
Darkness:  [space] # Event horizon (pure black)
Dim:       . , -
Medium:    ~ : ; =
Bright:    ! * # $
Intense:   @ % &
```

## ![NVIDIA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white) Technical Deep Dive: The CUDA Engine
**The Shift to Parallelism**

In the legacy version, Python calculated the position of every single pixel one by one. This created a bottleneck at ~2,000 particles.

In this version, we utilize **Numba** to compile Python code directly into PTX (Parallel Thread Execution) instructions for the GPU.

**The Kernel (`compute_points_kernel`)**

Instead of nested loops, we use a flattened grid approach. The GPU spawns thousands of threads, and each thread calculates the physics for one particle simultaneously.
```python
@cuda.jit
def compute_points_kernel(A, B, points_out, ...):
    # Unique Thread ID
    idx = cuda.grid(1) 
    
    if idx < total_points:
        # 1. Unpack "Virtual" Coordinates (Phi/Theta) from flat Index
        # 2. Apply Schwarzschild Metrics
        # 3. Calculate 3D Rotation Matrices (A & B)
        # 4. Compute Relativistic Doppler Factor
        # 5. Write result directly to GPU VRAM
```
## 📐 Mathematical Foundation

### Core Coordinate Systems

#### 1. Cylindrical Coordinates (GPU Kernel)
The accretion disk is generated directly on the GPU by mapping a linear thread index to cylindrical coordinates ($r, \theta, z$). Unlike the CPU version, this happens in parallel for 40,000+ points.

```python
# Inside compute_points_kernel
phi_raw = phi_idx * (6.28318 / phi_steps)
theta_raw = theta_idx * (6.28318 / theta_steps)

# Generate Geometry
radius = disk_inner + (disk_outer - disk_inner) * (phi_idx / phi_steps)
x = radius * math.cos(theta_raw)
y = 0.2 * math.sin(theta_raw * 3) * math.sin(phi_raw * 2) # Warping factor
z = radius * math.sin(theta_raw)
```
#### 2. 3D Rotation Matrices (Kernel Level)
Rotations are applied inside the kernel for every particle using standard rotation matrices.

**Y-axis Rotation (Spin $A$):**
```python
cos_A, sin_A = math.cos(A), math.sin(A)
x, z = x * cos_A - z * sin_A, x * sin_A + z * cos_A
```

**X-axis Rotation (Tilt $B$):**
```python
cos_B, sin_B = math.cos(B), math.sin(B)
y, z = y * cos_B - z * sin_B, y * sin_B + z * cos_B
```

### Perspective Projection (JIT Rasterizer)

The 3D-to-2D conversion is handled by the CPU `rasterize_points` function (accelerated via `@jit`). The view distance has been increased to accommodate the larger event horizon.
```python
dist = z + 8.0  # Increased view distance (previously 6.0)
if dist > 0:
    D = 1.0 / dist  # Inverse depth
    sx = int(x_off + 30 * D * x)
    sy = int(y_off + 20 * D * y)
```

### Black Hole Physics Approximations
#### 1. Schwarzschild Radius (The Void)
The event horizon has been scaled up for visual impact. The kernel checks every particle's distance from the center; if it breaches the radius, it is swallowed (luminance set to 0).

```python
schwarzschild_radius = 3.5  # Increased from 2.0
if dist_center < schwarzschild_radius * 1.1:
    lum = 0.0  # Light cannot escape
```
#### 2. The Einstein Ring (Gravitational Lensing)
Instead of expensive ray-tracing for every point, we generate a specific set of high-luminance particles that represent the "Ring of Fire"—light bent around the black hole's gravity well.

```python
# Explicit geometry generation for the Lensing Ring
ring_r = schwarzschild_radius * 1.6
x = ring_r * math.cos(angle)
y = schwarzschild_radius * 0.25 * math.sin(angle * 2) # Elliptical distortion
lum = 1.0 # Maximum brightness
```

#### 3. Relativistic Doppler Shift
We simulate the "beaming" effect where plasma moving towards the observer (left side) appears brighter. This is calculated using the orbital velocity vector relative to the camera angle $A$.

```python
velocity = math.sin(theta_raw) * math.cos(A)
doppler = 1.0 + velocity * 0.4
# Result: Approaching (+velocity) gets a luminance boost > 1.0
```

#### 4. Temperature-Luminosity Gradient
The accretion disk follows a thermodynamic gradient: hotter/brighter near the center, cooler/dimmer at the edges.

```python
base_lum = 1.0 - (dist_center - disk_inner) / (disk_outer - disk_inner)
```

#### 5. Final Luminance Calculation
The final brightness of a pixel is a composite of its thermodynamic temperature and relativistic velocity.

```python
lum = base_lum * doppler
```

### Color Mathematics (Thermal Mapping)

**"Magma" Gradient**

Instead of calculating RGB values per pixel (which is bandwidth-heavy), we map the calculated `lum` to a pre-rendered character set tinted with a "Magma" thermal gradient using HSV logic.

**The Palette Logic:**
- **Brightest ($>0.9$):** Low Saturation, High Value (White/Yellow Hot)
- **Mid-Range ($0.5$):** High Saturation, Medium Value (Orange)
- **Darkest ($<0.2$):** High Saturation, Low Value (Deep Red)

```python
# Python Pre-render Logic
h = 0.02 + (norm_i * 0.08)  # Shift Red -> Gold
s = 1.0 - (norm_i * 0.4)    # Desaturate towards white
v = 0.5 + (norm_i * 0.5)    # Boost brightness intensity
```

### Z-Buffer Algorithm (Numpy Optimized)
We utilize a flat 1D array or 2D Numpy array for the Z-buffer, allowing for O(1) access times during rasterization.
```python
# Inside rasterize_points (JIT)
if D > z_buffer[sy, sx]:
    z_buffer[sy, sx] = D
    # Map luminance to character index
    c_idx = int(lum * chars_len)
    screen_indices[sy, sx] = c_idx
```
### Animation Timing
To decouple the simulation speed from the high frame rate (144Hz+), we apply a scaling factor.
```python
fps_factor = 60 / TARGET_FPS
h_speed = 0.01 * fps_factor   # Constant rotational velocity
v_speed = 0.005 * fps_factor
```

## 🎮 Performance Metrics
Benchmarks verified on **NVIDIA RTX 5060**, display preset : 1200p @165Hz

| Metric | Result |
|:--|:--|
| Resolution | 1920 x 1200 (16:10) |
| Refresh Rate | 165 Hz |
| Average FPS |~138 FPS |
| Peak FPS | 153 FPS |
| Particle Load | "42,400 active points" |
| Improvement | "1,455% vs CPU" |

> "The code detected current refresh rate 165hz... the color of the particles seemed like got a bit dull... as the motion got much smoother... the period for character repetition got minimised... I fixed it by boosting the Value in HSV."  Also I have hidden a lot of stray warnings... those were never critical though - Developer Notes

## 🎨 Visual Breakdown
**Regions**
- *🖤 Event Horizon*: The point of no return - pure black void.

- *🔥 Inner Accretion Disk*: Superheated plasma, Gold/White glow.

- *🌅 Outer Accretion Disk*: Cooler matter, Deep Orange/Red emission.

- *💫 Einstein Ring*: Gravitationally lensed light forming bright arcs.

## 🌟 Inspiration

This project draws inspiration from:
- **OG project**: My first ever simulation, PyGame based python project [Gargantua_in_1080p](https://github.com/AmanBanik/Py.revival_wolfworks-66/tree/main/Projects/Proj07_Gargantua_in_1080p)
- **Interstellar (2014)**: Christopher Nolan's scientifically-grounded visualization
- **Kip Thorne's Research**: Theoretical physics behind black hole imaging
- **Event Horizon Telescope**: Real black hole observations (M87*, Sagittarius A*)
- **Classic ASCII Art**: Terminal-based graphics tradition
- **Real-time Graphics**: Modern adaptive refresh rate technologies
- **[3D Rotating Donut by developerrahulofficial](https://github.com/developerrahulofficial/3D-rotating-Donut.git)**: Inspiration for 3D ASCII art rendering techniques and mathematical transformations

## 🤝 Contributing
**Feel free to contribute improvements:**
- Enhanced physics calculations
- Feature testing
- Code cleaning
- Additional visual effects

Feel free to contact for any queries [email](mailto:amanbanik2023@outlook.com)

## 👋Author
[**Aman Banik**](https://www.linkedin.com/in/aman-banik-9a6a87308)

Explore my other repos: [here](https://github.com/AmanBanik)


drop a mail for contacting on my [email](mailto:amanbanik2023@outlook.com)

## 📜 License

This project is open source. Feel free to use, modify, and distribute as needed. [MIT LICENSE](\LICENSE)

## 🙏 Acknowledgments

- **Kip Thorne** - Scientific consultation for Interstellar
- **Christopher Nolan** - Visionary direction
- **Event Horizon Telescope Team** - Real black hole imaging
- **Numba Community** - For the high-performance JIT compiler bridging Python and CUDA.
- **NVIDIA CUDA Developers** - For the parallel computing architecture enabling this simulation.
- **Pygame Community** - Excellent graphics framework
- **ASCII Art Community** - Creative text-based visualization techniques
- **Cross-platform developers** - Display detection methodologies


---

*"We are not meant to save the world. We are meant to leave it."* - Cooper, Interstellar

**Experience the majesty of a black hole with intelligent display adaptation ~ now in its UNLEASHED form** 🚀✨💥



