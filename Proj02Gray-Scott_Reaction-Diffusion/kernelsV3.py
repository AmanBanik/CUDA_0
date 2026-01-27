# kernels.py
from numba import cuda
import config

@cuda.jit
def init_grid(u, v, r_grid, g_grid, b_grid):
    """
    Initialize grid with a center seed of Cyan color.
    """
    x, y = cuda.grid(2)
    if x < config.WIDTH and y < config.HEIGHT:
        u[y, x] = 1.0
        v[y, x] = 0.0
        r_grid[y, x] = 0.0
        g_grid[y, x] = 0.0
        b_grid[y, x] = 0.0
        
        # Center Seed
        cx, cy = config.WIDTH // 2, config.HEIGHT // 2
        if (x > cx - 20 and x < cx + 20 and 
            y > cy - 20 and y < cy + 20):
            
            noise = ((x * y * 12.9898) % 1.0)
            if noise > 0.5:
                v[y, x] = 0.8
                # Seed color: Cyan (Green + Blue)
                r_grid[y, x] = 0.0
                g_grid[y, x] = 1.0
                b_grid[y, x] = 1.0
            else:
                v[y, x] = 0.2

@cuda.jit
def update_step(u_in, v_in, u_out, v_out, 
                r_in, g_in, b_in, r_out, g_out, b_out):
    """
    Physics Step + Color Diffusion Step
    """
    c, r = cuda.grid(2)
    w, h = config.WIDTH, config.HEIGHT
    
    if c < w and r < h:
        # --- 1. Gray-Scott Physics ---
        curr_u = u_in[r, c]
        curr_v = v_in[r, c]
        
        # Neighbors
        left, right = (c - 1) % w, (c + 1) % w
        up, down    = (r - 1) % h, (r + 1) % h
        
        lap_u = (u_in[r, left] + u_in[r, right] + u_in[up, c] + u_in[down, c] - 4.0 * curr_u)
        lap_v = (v_in[r, left] + v_in[r, right] + v_in[up, c] + v_in[down, c] - 4.0 * curr_v)
        
        uvv = curr_u * curr_v * curr_v
        du = (config.Du * lap_u - uvv + config.FEED * (1.0 - curr_u))
        dv = (config.Dv * lap_v + uvv - (config.FEED + config.KILL) * curr_v)
        
        curr_v_next = curr_v + dv * config.dt
        u_out[r, c] = curr_u + du * config.dt
        v_out[r, c] = curr_v_next
        
        # --- 2. Color Diffusion ---
        # Colors diffuse naturally
        cr, cg, cb = r_in[r, c], g_in[r, c], b_in[r, c]
        
        lap_r = (r_in[r, left] + r_in[r, right] + r_in[up, c] + r_in[down, c] - 4.0 * cr)
        lap_g = (g_in[r, left] + g_in[r, right] + g_in[up, c] + g_in[down, c] - 4.0 * cg)
        lap_b = (b_in[r, left] + b_in[r, right] + b_in[up, c] + b_in[down, c] - 4.0 * cb)
        
        diff_rate = 0.5
        r_out[r, c] = cr + (diff_rate * lap_r) * config.dt
        g_out[r, c] = cg + (diff_rate * lap_g) * config.dt
        b_out[r, c] = cb + (diff_rate * lap_b) * config.dt

@cuda.jit
def render_camera_view(v_grid, r_grid, g_grid, b_grid, image_out, zoom, pan_x, pan_y):
    """
    Renders the grid with Thickness Modifier and RGB Tinting.
    """
    sx, sy = cuda.grid(2)
    if sx < config.WIDTH and sy < config.HEIGHT:
        dx = sx - config.WIDTH / 2.0
        dy = sy - config.HEIGHT / 2.0
        grid_x = int(pan_x + dx / zoom)
        grid_y = int(pan_y + dy / zoom)
        
        if 0 <= grid_x < config.WIDTH and 0 <= grid_y < config.HEIGHT:
            val = v_grid[grid_y, grid_x]
            
            # --- Thickness Math ---
            # Boost signal
            t = val * 4.0
            t = min(1.0, max(0.0, t))
            
            # Apply Modifier (Power curve to thin out the blob edges)
            # If modifier is < 1.0, this makes the exponent larger -> thinner lines
            t = t ** (1.0 / config.THICKNESS_MODIFIER + 0.5)
            
            # Get Color
            c_r = r_grid[grid_y, grid_x]
            c_g = g_grid[grid_y, grid_x]
            c_b = b_grid[grid_y, grid_x]
            
            # Write RGB
            image_out[sy, sx, 0] = int(min(1.0, c_r) * 255 * t)
            image_out[sy, sx, 1] = int(min(1.0, c_g) * 255 * t)
            image_out[sy, sx, 2] = int(min(1.0, c_b) * 255 * t)
        else:
            image_out[sy, sx, 0] = 0
            image_out[sy, sx, 1] = 0
            image_out[sy, sx, 2] = 0

@cuda.jit
def paint(v_grid, r_grid, g_grid, b_grid, x, y, radius, r_val, g_val, b_val, intensity):
    """
    Injects Chemical V + Color with Variable Intensity (Alpha).
    """
    c, r = cuda.grid(2)
    if r < config.HEIGHT and c < config.WIDTH:
        dist_sq = (c - x)**2 + (r - y)**2
        if dist_sq < radius**2:
            # 1. Inject V (Reactant)
            # Use max to ensure we don't wipe out existing strong patterns with weak brush
            # But allow adding to empty space
            inject_amount = 0.5 * intensity
            v_grid[r, c] = max(v_grid[r, c], inject_amount)
            
            # 2. Inject/Mix Color
            # Lerp towards new color based on brush intensity
            curr_r = r_grid[r, c]
            curr_g = g_grid[r, c]
            curr_b = b_grid[r, c]
            
            r_grid[r, c] = curr_r + (r_val - curr_r) * intensity
            g_grid[r, c] = curr_g + (g_val - curr_g) * intensity
            b_grid[r, c] = curr_b + (b_val - curr_b) * intensity