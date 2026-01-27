# main.py
import pygame
import numpy as np
from numba import cuda
import config
import kernelsV3 as kernels
import utils
import sys
import math

def main():
    # 1. Setup Pygame
    pygame.init()
    # SCALED allows 4K config to fit on 1080p monitors if needed
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED)
    pygame.display.set_caption("Gray-Scott: Vivid Edition")
    
    # Internal surface for the simulation render
    display_surf = pygame.Surface((config.WIDTH, config.HEIGHT))
    
    print("Allocating Vivid Memory (5 Grids)...")
    
    # 2. Allocate GPU Memory (Double Buffered)
    u_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    v_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    u_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    v_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    
    # Colors (R, G, B)
    r_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    g_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    b_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    
    r_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    g_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    b_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    
    # Output Image
    gpu_image = cuda.device_array((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    host_image = np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    
    # 3. Grid Logic
    threads = (config.TPB, config.TPB)
    blocks_x = int(np.ceil(config.WIDTH / config.TPB))
    blocks_y = int(np.ceil(config.HEIGHT / config.TPB))
    blocks = (blocks_x, blocks_y)
    
    # Initialize
    kernels.init_grid[blocks, threads](u_curr, v_curr, r_curr, g_curr, b_curr)
    
    # 4. State Variables
    cam_zoom = 1.0
    cam_x = config.WIDTH / 2.0
    cam_y = config.HEIGHT / 2.0
    
    color_idx = 0
    curr_color = config.COLOR_PALETTE[color_idx]
    brush_alpha = 1.0  # Intensity (0.0 to 1.0)
    
    is_panning = False
    last_mouse_pos = (0, 0)
    clock = pygame.time.Clock()
    running = True
    
    print("--- SYSTEM READY ---")

    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        
        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Zoom
            elif event.type == pygame.MOUSEWHEEL:
                cam_zoom += event.y * 0.1 * cam_zoom
                cam_zoom = max(0.1, min(cam_zoom, 50.0))
            
            # Pan Start/Stop
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3: # Right Click
                    is_panning = True
                    last_mouse_pos = current_mouse_pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    is_panning = False
            
            # Keys
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset all grids
                    kernels.init_grid[blocks, threads](u_curr, v_curr, r_curr, g_curr, b_curr)
                elif event.key == pygame.K_s:
                    utils.save_snapshot(display_surf)
                elif event.key == pygame.K_ESCAPE:
                    running = False
                
                # COLOR SWAP (T)
                elif event.key == pygame.K_t:
                    color_idx = (color_idx + 1) % len(config.COLOR_PALETTE)
                    curr_color = config.COLOR_PALETTE[color_idx]
                    print(f"Color: {curr_color}")

                # INTENSITY ([ and ])
                elif event.key == pygame.K_LEFTBRACKET:
                    brush_alpha = max(0.1, brush_alpha - 0.1)
                    print(f"Intensity: {brush_alpha:.1f}")
                elif event.key == pygame.K_RIGHTBRACKET:
                    brush_alpha = min(1.0, brush_alpha + 0.1)
                    print(f"Intensity: {brush_alpha:.1f}")

        # --- Panning Logic ---
        if is_panning:
            dx = current_mouse_pos[0] - last_mouse_pos[0]
            dy = current_mouse_pos[1] - last_mouse_pos[1]
            cam_x -= dx / cam_zoom
            cam_y -= dy / cam_zoom
            last_mouse_pos = current_mouse_pos

        # --- Painting Logic ---
        if pygame.mouse.get_pressed()[0]:
            mx, my = current_mouse_pos
            # Convert Screen -> World
            sc_x, sc_y = config.WIDTH/2, config.HEIGHT/2
            world_x = cam_x + (mx - sc_x) / cam_zoom
            world_y = cam_y + (my - sc_y) / cam_zoom
            
            # Scale radius by zoom (so it doesn't get gigantic when zoomed out)
            eff_radius = config.BRUSH_RADIUS / max(0.5, math.log(cam_zoom + 1))
            
            r_val, g_val, b_val = curr_color
            
            kernels.paint[blocks, threads](
                v_curr, r_curr, g_curr, b_curr,
                world_x, world_y, eff_radius,
                r_val, g_val, b_val, brush_alpha
            )

        # --- Simulation Loop ---
        for _ in range(config.STEPS_PER_FRAME):
            kernels.update_step[blocks, threads](
                u_curr, v_curr, u_next, v_next,
                r_curr, g_curr, b_curr, r_next, g_next, b_next
            )
            # Swap buffers
            u_curr, u_next = u_next, u_curr
            v_curr, v_next = v_next, v_curr
            r_curr, r_next = r_next, r_curr
            g_curr, g_next = g_next, g_curr
            b_curr, b_next = b_next, b_curr

        # --- Render ---
        kernels.render_camera_view[blocks, threads](
            v_curr, r_curr, g_curr, b_curr, gpu_image, cam_zoom, cam_x, cam_y
        )
        
        gpu_image.copy_to_host(host_image)
        frame_data = np.transpose(host_image, (1, 0, 2))
        pygame.surfarray.blit_array(display_surf, frame_data)
        
        # --- UI Overlay ---
        # Create transparent surface for the UI
        ui_surf = pygame.Surface((150, 100), pygame.SRCALPHA)
        
        # Draw Color Indicator (With Alpha)
        c_r = int(curr_color[0] * 255)
        c_g = int(curr_color[1] * 255)
        c_b = int(curr_color[2] * 255)
        c_a = int(brush_alpha * 255)
        
        # Inner Circle (Color + Alpha)
        pygame.draw.circle(ui_surf, (c_r, c_g, c_b, c_a), (40, 40), 20)
        # Outer Ring (Solid White)
        pygame.draw.circle(ui_surf, (255, 255, 255), (40, 40), 22, 2)
        
        # Blit UI
        display_surf.blit(ui_surf, (0, 0))
        
        screen.blit(display_surf, (0, 0))
        pygame.display.flip()
        
        pygame.display.set_caption(f"Gray-Scott Vivid | FPS: {clock.get_fps():.1f} | Zoom: {cam_zoom:.1f}x")
        clock.tick()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()