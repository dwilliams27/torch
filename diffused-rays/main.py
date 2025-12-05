"""Main entry point for diffused-rays."""

import sys
import time
import pygame
import numpy as np

import config
from game_state import Player, Map
from raycaster import cast_rays
from maps.test_map import MAP, PLAYER_START_X, PLAYER_START_Y, PLAYER_START_ANGLE


def main():
    """Main game loop."""
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
    pygame.display.set_caption("Diffused Rays")
    clock = pygame.time.Clock()

    # Initialize game state
    player = Player(PLAYER_START_X, PLAYER_START_Y, PLAYER_START_ANGLE)
    game_map = Map(MAP)

    # Font for FPS display
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # SD stylizer state
    sd_enabled = False
    stylize_frame = None
    sd_loading = False
    last_stylized_frame = None

    # Show loading message
    def show_message(text):
        screen.fill((0, 0, 0))
        msg = font.render(text, True, (255, 255, 255))
        rect = msg.get_rect(center=(config.DISPLAY_WIDTH // 2, config.DISPLAY_HEIGHT // 2))
        screen.blit(msg, rect)
        pygame.display.flip()

    running = True
    while running:
        frame_start = time.time()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Toggle SD processing
                    if not sd_enabled and stylize_frame is None:
                        # First time enabling - need to load the model
                        show_message("Loading Stable Diffusion model...")
                        pygame.event.pump()  # Keep window responsive
                        try:
                            from stylizer import stylize_frame as sf, load_pipeline
                            load_pipeline()
                            stylize_frame = sf
                            sd_enabled = True
                            last_stylized_frame = None
                        except Exception as e:
                            print(f"Failed to load SD: {e}")
                            show_message(f"SD load failed: {str(e)[:50]}")
                            pygame.time.wait(2000)
                    else:
                        sd_enabled = not sd_enabled
                        if not sd_enabled:
                            last_stylized_frame = None

        # Handle continuous key input
        keys = pygame.key.get_pressed()

        # Calculate delta time (use a fixed dt when SD is enabled for smoother input)
        if sd_enabled:
            dt = 0.1  # Assume ~10 FPS target for input smoothing
        else:
            dt = clock.tick(config.FPS_CAP) / 1000.0

        # Movement
        move_distance = config.MOVE_SPEED * dt
        turn_amount = config.TURN_SPEED * dt

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            player.move_forward(move_distance, game_map)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            player.move_backward(move_distance, game_map)
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            player.turn_left(turn_amount)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            player.turn_right(turn_amount)

        # Render frame from raycaster
        frame = cast_rays(player, game_map)

        # Apply SD stylization if enabled
        if sd_enabled and stylize_frame is not None:
            try:
                frame = stylize_frame(frame)
                last_stylized_frame = frame
            except Exception as e:
                print(f"SD error: {e}")
                sd_enabled = False

        # Convert numpy array to pygame surface
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        # Scale up to display size
        scaled_surface = pygame.transform.scale(
            surface,
            (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
        )

        # Draw to screen
        screen.blit(scaled_surface, (0, 0))

        # Calculate actual FPS
        frame_time = time.time() - frame_start
        actual_fps = 1.0 / frame_time if frame_time > 0 else 0

        # Draw FPS counter
        fps_text = font.render(f"FPS: {actual_fps:.1f}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        # Draw SD status
        sd_status = "SD: ON (SPACE to toggle)" if sd_enabled else "SD: OFF (SPACE to enable)"
        sd_color = (0, 255, 0) if sd_enabled else (255, 255, 0)
        sd_text = small_font.render(sd_status, True, sd_color)
        screen.blit(sd_text, (10, 45))

        # Draw controls help
        controls = small_font.render("WASD/Arrows: Move | ESC: Quit", True, (200, 200, 200))
        screen.blit(controls, (10, config.DISPLAY_HEIGHT - 30))

        pygame.display.flip()

        # If SD is enabled, don't use clock.tick (we already measured frame time)
        if not sd_enabled:
            pass  # clock.tick already called above
        else:
            clock.tick()  # Just update clock without limiting

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
