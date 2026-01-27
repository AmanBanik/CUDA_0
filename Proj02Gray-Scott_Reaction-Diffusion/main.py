# main.py
import pygame
import numpy as np
import math
from numba import cuda
import config
import kernels
import utils
import sys

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Gray-Scott: Beast Mode")
    
    # Internal surface
    display_surf = pygame.Surface((config.WIDTH, config.HEIGHT))
    
    # --- GPU Allocations (Same as before) ---
    print(f"Allocating VRAM for {config.WIDTH}x{config.HEIGHT}...")
    u_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    v_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    u_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    v_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    gpu_image = cuda.device_array((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    host_image = np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    
    # --- Grid Dimensions ---
    threads = (config.TPB, config.TPB)
    blocks_x = int(np.ceil(config.WIDTH / config.TPB))
    blocks_y = int(np.ceil(config.HEIGHT / config.TPB))
    blocks = (blocks_x, blocks_y)
    
    # Initialize
    kernels.init_grid[blocks, threads](u_curr, v_curr)
    
    # --- CAMERA VARIABLES ---
    cam_zoom = 1.0
    cam_x = config.WIDTH / 2.0
    cam_y = config.HEIGHT / 2.0
    
    # Interaction State
    is_panning = False
    last_mouse_pos = (0, 0)
    
    clock = pygame.time.Clock()
    running = True
    
    print("\n--- BEAST MODE CONTROLS ---")
    print(" [Scroll Wheel] : Zoom In/Out")
    print(" [Right Click]  : Pan Camera")
    print(" [Left Click]   : Paint (Draws on grid)")
    print(" [R] : Reset | [S] : Save")

    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- Zooming (Scroll) ---
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom logic
                zoom_speed = 0.1
                prev_zoom = cam_zoom
                cam_zoom += event.y * zoom_speed * cam_zoom
                cam_zoom = max(0.1, min(cam_zoom, 50.0)) # Clamp zoom (0.1x to 50x)
                
            # --- Panning Start/End ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3: # Right Click
                    is_panning = True
                    last_mouse_pos = current_mouse_pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    is_panning = False
            
            # --- Keyboard ---
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    kernels.init_grid[blocks, threads](u_curr, v_curr)
                    # Reset camera too
                    cam_zoom = 1.0
                    cam_x, cam_y = config.WIDTH/2, config.HEIGHT/2
                elif event.key == pygame.K_s:
                    utils.save_snapshot(display_surf)
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # --- Continuous Panning ---
        if is_panning:
            dx = current_mouse_pos[0] - last_mouse_pos[0]
            dy = current_mouse_pos[1] - last_mouse_pos[1]
            # Move camera opposite to drag, scaled by zoom
            cam_x -= dx / cam_zoom
            cam_y -= dy / cam_zoom
            last_mouse_pos = current_mouse_pos

        # --- Continuous Painting ---
        if pygame.mouse.get_pressed()[0]: # Left Click
            mx, my = current_mouse_pos
            # Transform Screen Coords -> World Coords
            # World = CameraPos + (Screen - ScreenCenter) / Zoom
            screen_center_x = config.WIDTH / 2
            screen_center_y = config.HEIGHT / 2
            
            world_x = cam_x + (mx - screen_center_x) / cam_zoom
            world_y = cam_y + (my - screen_center_y) / cam_zoom
            
            # Adjust brush radius based on zoom (optional, feels more natural)
            # Effective radius in world units
            eff_radius = config.BRUSH_RADIUS / max(0.5, math.log(cam_zoom + 1)) 
            
            kernels.paint[blocks, threads](v_curr, world_x, world_y, eff_radius)

        # --- Physics Loop ---
        for _ in range(config.STEPS_PER_FRAME):
            kernels.update_step[blocks, threads](u_curr, v_curr, u_next, v_next)
            u_curr, u_next = u_next, u_curr
            v_curr, v_next = v_next, v_curr

        # --- Render View (New Camera Kernel) ---
        kernels.render_camera_view[blocks, threads](v_curr, gpu_image, cam_zoom, cam_x, cam_y)
        
        # --- Display ---
        gpu_image.copy_to_host(host_image)
        frame_data = np.transpose(host_image, (1, 0, 2))
        pygame.surfarray.blit_array(display_surf, frame_data)
        
        # Overlay UI info
        # (Optional: Draw a tiny text showing zoom level)
        
        screen.blit(display_surf, (0, 0))
        pygame.display.flip()
        
        pygame.display.set_caption(f"Gray-Scott Beast | FPS: {clock.get_fps():.1f} | Zoom: {cam_zoom:.2f}x")
        clock.tick()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()