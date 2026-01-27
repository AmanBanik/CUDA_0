# kernels.py
from numba import cuda
import math
import config

@cuda.jit
def init_grid(u, v):
    """Initializes with Noise Patch (Same as before)"""
    x, y = cuda.grid(2)
    if x < config.WIDTH and y < config.HEIGHT:
        u[y, x] = 1.0
        v[y, x] = 0.0
        cx, cy = config.WIDTH // 2, config.HEIGHT // 2
        # Random noise seed
        if (x > cx - 20 and x < cx + 20 and y > cy - 20 and y < cy + 20):
            noise = ((x * y * 12.9898) % 1.0)
            if noise > 0.5: v[y, x] = 0.8
            else:           v[y, x] = 0.2

@cuda.jit
def update_step(u_in, v_in, u_out, v_out):
    """Standard Physics (Same as before)"""
    c, r = cuda.grid(2)
    w, h = config.WIDTH, config.HEIGHT
    if c < w and r < h:
        curr_u = u_in[r, c]
        curr_v = v_in[r, c]
        
        # Wrap-around neighbors
        left, right = (c - 1) % w, (c + 1) % w
        up, down    = (r - 1) % h, (r + 1) % h
        
        lap_u = (u_in[r, left] + u_in[r, right] + u_in[up, c] + u_in[down, c] - 4.0 * curr_u)
        lap_v = (v_in[r, left] + v_in[r, right] + v_in[up, c] + v_in[down, c] - 4.0 * curr_v)
        
        uvv = curr_u * curr_v * curr_v
        du = (config.Du * lap_u - uvv + config.FEED * (1.0 - curr_u))
        dv = (config.Dv * lap_v + uvv - (config.FEED + config.KILL) * curr_v)
        
        u_out[r, c] = curr_u + du * config.dt
        v_out[r, c] = curr_v + dv * config.dt

@cuda.jit
def render_camera_view(v_grid, image_out, zoom, pan_x, pan_y):
    """
    Renders the view based on a Virtual Camera.
    zoom: Float multiplier (1.0 = normal, 10.0 = 10x magnification)
    pan_x, pan_y: The coordinate of the simulation grid that is at the CENTER of the screen.
    """
    sx, sy = cuda.grid(2) # Screen Coordinates
    
    # Only render if within screen bounds
    if sx < config.WIDTH and sy < config.HEIGHT:
        
        # 1. Transform Screen Coord -> World (Grid) Coord
        # Offset from screen center
        dx = sx - config.WIDTH / 2.0
        dy = sy - config.HEIGHT / 2.0
        
        # Scale by zoom and add camera position
        grid_x = int(pan_x + dx / zoom)
        grid_y = int(pan_y + dy / zoom)
        
        # 2. Sample the Grid (Boundary Check)
        if 0 <= grid_x < config.WIDTH and 0 <= grid_y < config.HEIGHT:
            val = v_grid[grid_y, grid_x]
            
            # --- Coloring Logic (Same Vivid Palette) ---
            t = val * 4.0
            t = min(1.0, max(0.0, t))
            
            # Cosine Palette
            r_val = 0.1 + 0.9 * math.cos(6.28 * (t + 0.0))
            g_val = 0.5 + 0.5 * math.cos(6.28 * (t + 0.3))
            b_val = 0.6 + 0.4 * math.cos(6.28 * (t + 0.6))
            
            image_out[sy, sx, 0] = int(r_val * 255 * t)
            image_out[sy, sx, 1] = int(g_val * 255 * t)
            image_out[sy, sx, 2] = int(b_val * 255 * t)
        else:
            # Out of bounds (Background)
            image_out[sy, sx, 0] = 0
            image_out[sy, sx, 1] = 0
            image_out[sy, sx, 2] = 0

@cuda.jit
def paint(v_grid, target_x, target_y, radius):
    """Simple circular brush at specific World Coordinates"""
    c, r = cuda.grid(2)
    # Optimization: Only check pixels near the target
    # But for simplicity on GPU, we check full grid or use a bounding box approach.
    # To keep it fast/simple for now:
    if r < config.HEIGHT and c < config.WIDTH:
        dist_sq = (c - target_x)**2 + (r - target_y)**2
        if dist_sq < radius**2:
            v_grid[r, c] = 0.5