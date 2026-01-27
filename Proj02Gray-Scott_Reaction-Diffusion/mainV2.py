# main.py
import pygame
import numpy as np
from numba import cuda
import config
import kernelsV2 as kernels
import utils
import sys
import math

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Gray-Scott: Vivid Color Edition")
    display_surf = pygame.Surface((config.WIDTH, config.HEIGHT))
    
    print("Allocating Vivid Memory...")
    
    # 1. Chemical Grids
    u_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    v_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    u_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    v_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    
    # 2. Color Grids (Double Buffered for diffusion)
    # We need R, G, B channels
    r_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    g_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    b_curr = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    
    r_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    g_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    b_next = cuda.device_array((config.HEIGHT, config.WIDTH), dtype=np.float32)
    
    # Output Image
    gpu_image = cuda.device_array((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    host_image = np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    
    # Dimensions
    threads = (config.TPB, config.TPB)
    blocks_x = int(np.ceil(config.WIDTH / config.TPB))
    blocks_y = int(np.ceil(config.HEIGHT / config.TPB))
    blocks = (blocks_x, blocks_y)
    
    # Init
    kernels.init_grid[blocks, threads](u_curr, v_curr, r_curr, g_curr, b_curr)
    
    # --- Camera & Color State ---
    cam_zoom = 1.0
    cam_x = config.WIDTH / 2.0
    cam_y = config.HEIGHT / 2.0
    
    # Color Selection
    color_index = 0
    current_color = config.COLOR_PALETTE[color_index]
    
    is_panning = False
    last_mouse_pos = (0, 0)
    clock = pygame.time.Clock()
    running = True
    
    print("CONTROLS:")
    print(" [T] : Switch Brush Color")
    print(" [Left Click] : Paint with current color")
    
    while running:
        current_mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                cam_zoom += event.y * 0.1 * cam_zoom
                cam_zoom = max(0.1, min(cam_zoom, 50.0))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3: is_panning = True; last_mouse_pos = current_mouse_pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3: is_panning = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    kernels.init_grid[blocks, threads](u_curr, v_curr, r_curr, g_curr, b_curr)
                elif event.key == pygame.K_s:
                    utils.save_snapshot(display_surf)
                elif event.key == pygame.K_ESCAPE:
                    running = False
                
                # --- CHANGE COLOR ---
                elif event.key == pygame.K_t:
                    color_index = (color_index + 1) % len(config.COLOR_PALETTE)
                    current_color = config.COLOR_PALETTE[color_index]
                    print(f"Brush Color Switched to: {current_color}")

        if is_panning:
            dx = current_mouse_pos[0] - last_mouse_pos[0]
            dy = current_mouse_pos[1] - last_mouse_pos[1]
            cam_x -= dx / cam_zoom
            cam_y -= dy / cam_zoom
            last_mouse_pos = current_mouse_pos
            
        if pygame.mouse.get_pressed()[0]:
            mx, my = current_mouse_pos
            screen_cx, screen_cy = config.WIDTH / 2, config.HEIGHT / 2
            world_x = cam_x + (mx - screen_cx) / cam_zoom
            world_y = cam_y + (my - screen_cy) / cam_zoom
            radius = config.BRUSH_RADIUS / max(0.5, math.log(cam_zoom + 1))
            
            # Pass RGB values to the paint kernel
            r_val, g_val, b_val = current_color
            kernels.paint[blocks, threads](
                v_curr, r_curr, g_curr, b_curr, 
                world_x, world_y, radius, r_val, g_val, b_val
            )

        # --- UPDATE LOOP ---
        for _ in range(config.STEPS_PER_FRAME):
            kernels.update_step[blocks, threads](
                u_curr, v_curr, u_next, v_next,
                r_curr, g_curr, b_curr, r_next, g_next, b_next
            )
            # Swap Chem
            u_curr, u_next = u_next, u_curr
            v_curr, v_next = v_next, v_curr
            # Swap Colors
            r_curr, r_next = r_next, r_curr
            g_curr, g_next = g_next, g_curr
            b_curr, b_next = b_next, b_curr

        # --- RENDER ---
        kernels.render_camera_view[blocks, threads](
            v_curr, r_curr, g_curr, b_curr, gpu_image, cam_zoom, cam_x, cam_y
        )
        
        gpu_image.copy_to_host(host_image)
        frame_data = np.transpose(host_image, (1, 0, 2))
        pygame.surfarray.blit_array(display_surf, frame_data)
        
        # Draw UI (Color Indicator)
        pygame.draw.circle(display_surf, 
                           (int(current_color[0]*255), int(current_color[1]*255), int(current_color[2]*255)), 
                           (30, 30), 20)
        pygame.draw.circle(display_surf, (255, 255, 255), (30, 30), 22, 2) # White border
        
        screen.blit(display_surf, (0, 0))
        pygame.display.flip()
        
        pygame.display.set_caption(f"Gray-Scott Vivid | FPS: {clock.get_fps():.1f}")
        clock.tick()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()