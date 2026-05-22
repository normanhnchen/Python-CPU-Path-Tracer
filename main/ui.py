"""Helper functions for rendering the ui in pygame."""


import contextlib

# Remove pygame message.
with contextlib.redirect_stdout(None):
    import pygame


def draw_text(screen, font_type, font_size, text, pos, right_aligned=False, font_color=(255, 255, 255),
              bg_color=(0, 0, 0, 0), bg_padding=5):

    font = pygame.font.SysFont(font_type, font_size)

    text_surface = font.render(text, True, font_color)
    text_rect = text_surface.get_rect()

    if right_aligned:
        text_rect.topright = pos
        pos = (text_rect.x, text_rect.y)
    
    bg_surface = pygame.Surface((text_rect.width + bg_padding*2, text_rect.height + bg_padding*2),
                                pygame.SRCALPHA)
    bg_surface.fill(bg_color)
    
    screen.blit(bg_surface, (pos[0] - bg_padding, pos[1] - bg_padding))
    screen.blit(text_surface, pos)