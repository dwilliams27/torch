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

    # Font for UI
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # SD stylizer state
    sd_enabled = False
    async_stylizer = None
    sd_fps = 0.0
    sd_frame_count = 0
    sd_last_time = time.time()

    # Visual styles to cycle through
    sd_styles = [
        ("Dungeon", "dark fantasy dungeon, stone walls, torchlight, mysterious"),
        ("Cyberpunk", "neon cyberpunk corridor, glowing lights, sci-fi, futuristic"),
        ("Underwater", "underwater ancient ruins, blue green tint, mystical, fish"),
        ("Hellscape", "hellscape corridor, fire and lava, demonic, red orange glow"),
        ("Ice Cave", "frozen ice cave, blue crystals, cold, magical winter"),
        ("Overgrown", "overgrown temple ruins, vines and moss, nature, sunbeams"),
        ("Oil Paint", "oil painting, impressionist brush strokes, artistic, colorful"),
        ("Anime", "anime style dungeon, cel shaded, vibrant colors, japanese animation"),
    ]
    sd_style_index = 0

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
                    if async_stylizer is None:
                        # First time - load and start async stylizer
                        show_message("Loading Stable Diffusion model...")
                        pygame.event.pump()
                        try:
                            from stylizer import AsyncStylizer
                            async_stylizer = AsyncStylizer()
                            async_stylizer.start()
                            sd_enabled = True
                            sd_last_time = time.time()
                            sd_frame_count = 0
                        except Exception as e:
                            print(f"Failed to load SD: {e}")
                            show_message(f"SD load failed: {str(e)[:50]}")
                            pygame.time.wait(2000)
                    else:
                        sd_enabled = not sd_enabled
                        if sd_enabled:
                            sd_last_time = time.time()
                            sd_frame_count = 0
                elif event.key == pygame.K_LEFTBRACKET:
                    # Previous style
                    sd_style_index = (sd_style_index - 1) % len(sd_styles)
                elif event.key == pygame.K_RIGHTBRACKET:
                    # Next style
                    sd_style_index = (sd_style_index + 1) % len(sd_styles)

        # Handle continuous key input
        keys = pygame.key.get_pressed()

        # Delta time for movement (capped for consistency)
        dt = min(clock.tick(config.FPS_CAP) / 1000.0, 0.1)

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
        raw_frame = cast_rays(player, game_map)

        # SD processing (async)
        if sd_enabled and async_stylizer is not None:
            # Submit current frame for processing with current style prompt
            _, prompt = sd_styles[sd_style_index]
            async_stylizer.submit_frame(raw_frame.copy(), prompt=prompt)

            # Get latest stylized result (or use raw if none ready)
            display_frame = async_stylizer.get_latest(raw_frame)

            # Track SD FPS
            current_count = async_stylizer.frames_processed
            if current_count > sd_frame_count:
                now = time.time()
                if now - sd_last_time > 0.5:  # Update every 0.5s
                    sd_fps = (current_count - sd_frame_count) / (now - sd_last_time)
                    sd_frame_count = current_count
                    sd_last_time = now
        else:
            display_frame = raw_frame

        # Convert numpy array to pygame surface
        surface = pygame.surfarray.make_surface(display_frame.swapaxes(0, 1))

        # Scale up to display size
        scaled_surface = pygame.transform.scale(
            surface,
            (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
        )

        # Draw to screen
        screen.blit(scaled_surface, (0, 0))

        # Calculate display FPS
        frame_time = time.time() - frame_start
        display_fps = 1.0 / frame_time if frame_time > 0 else 0

        # Draw FPS counter
        fps_text = font.render(f"Display: {display_fps:.0f} FPS", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        if sd_enabled:
            sd_fps_text = font.render(f"SD: {sd_fps:.1f} FPS", True, (0, 255, 0))
            screen.blit(sd_fps_text, (10, 40))

            # Draw current style
            style_name, _ = sd_styles[sd_style_index]
            style_text = font.render(f"Style: {style_name}", True, (255, 200, 100))
            screen.blit(style_text, (10, 70))

        # Draw SD status
        if sd_enabled:
            sd_status = "SD: ON [SPACE off] | [ ] change style"
            sd_color = (0, 255, 0)
        elif async_stylizer is not None:
            sd_status = "SD: OFF [SPACE on]"
            sd_color = (255, 255, 0)
        else:
            sd_status = "SD: [SPACE to load]"
            sd_color = (200, 200, 200)

        sd_text = small_font.render(sd_status, True, sd_color)
        screen.blit(sd_text, (10, 105))

        # Draw controls help
        controls = small_font.render("WASD/Arrows: Move | ESC: Quit", True, (200, 200, 200))
        screen.blit(controls, (10, config.DISPLAY_HEIGHT - 30))

        pygame.display.flip()

    # Cleanup
    if async_stylizer is not None:
        async_stylizer.stop()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
