# utils.py
import pygame
import os
from datetime import datetime

def save_snapshot(surface):
    """Saves the current frame to a 'snapshots' folder."""
    if not os.path.exists("snapshots"):
        os.makedirs("snapshots")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshots/sim_{timestamp}.png"
    pygame.image.save(surface, filename)
    print(f"Captured: {filename}")