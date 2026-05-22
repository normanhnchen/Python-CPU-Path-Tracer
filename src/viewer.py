"""
Main pygame viewing module.

Displays a comparison slider between the original and denoised render outputs.
"""


import contextlib


import src.ui as ui
import src.settings as settings
import src.paths as paths

# Remove pygame message.
with contextlib.redirect_stdout(None):
    import pygame


def view():
    """Display the saved renders as a comparison with a slider using pygame."""
    
    WIDTH, HEIGHT = settings.screen_dimensions

    original_render = pygame.image.load(str(paths.ORIGINAL_RENDER_PATH))
    denoised_render = pygame.image.load(str(paths.DENOISED_RENDER_PATH))

    pygame.init()
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Path Tracer")

    slider_x = WIDTH // 2
    running = True
    dragging = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                dragging = True
                slider_x = max(0, min(WIDTH, event.pos[0]))
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                # Constrain slider to screen bounds.
                slider_x = max(0, min(WIDTH, event.pos[0]))

        font_size = WIDTH // 20
        text_offset = WIDTH // 50
        
        # Save the current clipping bounds so it can be restored later.
        original_clip = SCREEN.get_clip()

        # Left clip to allow drawing if it's left of the slider.
        SCREEN.set_clip(pygame.Rect(0, 0, slider_x, HEIGHT))
        SCREEN.blit(original_render, (0, 0))
        ui.draw_text(SCREEN, "Unispace", font_size, "Original", pos=(text_offset, text_offset), bg_color=(0, 0, 0, 150))

        # Right clip to allow drawing if it's right of the slider.
        SCREEN.set_clip(pygame.Rect(slider_x, 0, WIDTH - slider_x, HEIGHT))
        SCREEN.blit(denoised_render, (0, 0))
        ui.draw_text(SCREEN, "Unispace", font_size, "Denoised", pos=(WIDTH - text_offset, text_offset), right_aligned=True, bg_color=(0, 0, 0, 150))

        # Reset clip.
        SCREEN.set_clip(original_clip)

        # Draw slider.
        pygame.draw.line(SCREEN, "green", (slider_x, 0), (slider_x, HEIGHT), WIDTH // 100)

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    view()
