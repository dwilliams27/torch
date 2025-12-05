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
    sd_blend = 1.0  # Blend factor: 0.0 = raw, 1.0 = full SD

    # Texture mode state
    texture_mode = False
    texture_manager = None
    texture_stylizer = None
    tex_fps = 0.0
    tex_frame_count = 0
    tex_last_time = time.time()
    tex_last_submit = 0.0  # Throttle submissions

    # Visual styles to cycle through
    sd_styles = [
        ("Torch", "dark medieval dungeon, burning torches on walls, flickering firelight, warm orange glow, ancient stone corridors, shadows dancing, atmospheric"),
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
                    # Toggle SD processing (mutually exclusive with texture mode)
                    if sd_enabled:
                        # Turn off SD view mode
                        sd_enabled = False
                    else:
                        # Turn on SD view mode - disable texture mode first
                        if texture_mode:
                            texture_mode = False
                            # Wait for current SD inference to complete (~300ms per frame)
                            show_message("Switching modes...")
                            pygame.event.pump()
                            time.sleep(1.0)

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
                            sd_enabled = True
                            sd_last_time = time.time()
                            sd_frame_count = 0
                elif event.key == pygame.K_LEFTBRACKET:
                    # Previous style
                    sd_style_index = (sd_style_index - 1) % len(sd_styles)
                elif event.key == pygame.K_RIGHTBRACKET:
                    # Next style
                    sd_style_index = (sd_style_index + 1) % len(sd_styles)
                elif event.key == pygame.K_MINUS:
                    # Decrease SD blend
                    sd_blend = max(0.0, sd_blend - 0.1)
                elif event.key == pygame.K_EQUALS:
                    # Increase SD blend
                    sd_blend = min(1.0, sd_blend + 0.1)
                elif event.key == pygame.K_t:
                    # Toggle texture mode (mutually exclusive with SD view mode)
                    if texture_mode:
                        # Turn off texture mode
                        texture_mode = False
                    else:
                        # Turn on texture mode - disable SD view mode first
                        if sd_enabled:
                            sd_enabled = False
                            # Wait for current SD inference to complete (~300ms per frame)
                            show_message("Switching modes...")
                            pygame.event.pump()
                            time.sleep(1.0)

                        if texture_manager is None:
                            # First time - initialize texture system
                            show_message("Initializing texture mode...")
                            pygame.event.pump()
                            try:
                                from texture_manager import TextureManager, AsyncTextureStylizer
                                texture_manager = TextureManager()
                                texture_stylizer = AsyncTextureStylizer()
                                texture_stylizer.start()
                                texture_mode = True
                                tex_last_time = time.time()
                                tex_frame_count = 0
                            except Exception as e:
                                print(f"Failed to initialize texture mode: {e}")
                                import traceback
                                traceback.print_exc()
                                show_message(f"Texture init failed: {str(e)[:50]}")
                                pygame.time.wait(2000)
                        else:
                            texture_mode = True
                            tex_last_time = time.time()
                            tex_frame_count = 0

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

        # Handle texture mode - submit atlas for stylization
        if texture_mode and texture_stylizer is not None:
            # Throttle submissions to reduce contention (every 200ms)
            now = time.time()
            if now - tex_last_submit > 0.2:
                tex_last_submit = now
                _, prompt = sd_styles[sd_style_index]
                # Cache base atlas - only create once
                if texture_manager.base_atlas is None:
                    texture_manager.create_base_atlas()
                texture_stylizer.submit(texture_manager.base_atlas, prompt)

            # Get latest styled atlas if available
            styled_atlas = texture_stylizer.get_result()
            if styled_atlas is not None:
                texture_manager.split_atlas(styled_atlas)

            # Track texture mode FPS
            current_count = texture_stylizer.frames_processed
            if current_count > tex_frame_count:
                now = time.time()
                if now - tex_last_time > 0.5:
                    tex_fps = (current_count - tex_frame_count) / (now - tex_last_time)
                    tex_frame_count = current_count
                    tex_last_time = now

        # Render frame from raycaster (pass texture_manager if in texture mode)
        if texture_mode and texture_manager is not None:
            raw_frame = cast_rays(player, game_map, texture_manager)
        else:
            raw_frame = cast_rays(player, game_map)

        # SD processing (async) - only if not in texture mode
        if sd_enabled and async_stylizer is not None and not texture_mode:
            # Submit current frame for processing with current style prompt
            _, prompt = sd_styles[sd_style_index]
            async_stylizer.submit_frame(raw_frame.copy(), prompt=prompt)

            # Get latest stylized result
            sd_frame = async_stylizer.get_latest(raw_frame)

            # Blend raw and SD frames based on sd_blend factor
            if sd_blend >= 1.0:
                display_frame = sd_frame
            elif sd_blend <= 0.0:
                display_frame = raw_frame
            else:
                display_frame = (
                    raw_frame.astype(np.float32) * (1 - sd_blend) +
                    sd_frame.astype(np.float32) * sd_blend
                ).astype(np.uint8)

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

        if sd_enabled and not texture_mode:
            sd_fps_text = font.render(f"SD: {sd_fps:.1f} FPS", True, (0, 255, 0))
            screen.blit(sd_fps_text, (10, 40))

            # Draw current style and blend
            style_name, _ = sd_styles[sd_style_index]
            style_text = font.render(f"Style: {style_name}", True, (255, 200, 100))
            screen.blit(style_text, (10, 70))

            blend_text = font.render(f"Blend: {int(sd_blend * 100)}%", True, (200, 200, 255))
            screen.blit(blend_text, (10, 100))
        elif texture_mode:
            tex_fps_text = font.render(f"Tex: {tex_fps:.1f} FPS", True, (100, 200, 255))
            screen.blit(tex_fps_text, (10, 40))

            # Draw current style
            style_name, _ = sd_styles[sd_style_index]
            style_text = font.render(f"Style: {style_name}", True, (255, 200, 100))
            screen.blit(style_text, (10, 70))

        # Draw mode status
        if texture_mode:
            mode_status = "TEX: ON [T off] | [ ] style | [SPACE view mode]"
            mode_color = (100, 200, 255)
            status_y = 105
        elif sd_enabled:
            mode_status = "[ ] style | [-/=] blend | [T texture] | [SPACE off]"
            mode_color = (0, 255, 0)
            status_y = 130
        elif async_stylizer is not None:
            mode_status = "SD: OFF [SPACE on] | [T texture mode]"
            mode_color = (255, 255, 0)
            status_y = 105
        else:
            mode_status = "[SPACE load SD] | [T texture mode]"
            mode_color = (200, 200, 200)
            status_y = 105

        mode_text = small_font.render(mode_status, True, mode_color)
        screen.blit(mode_text, (10, status_y))

        # Draw controls help
        controls = small_font.render("WASD/Arrows: Move | ESC: Quit", True, (200, 200, 200))
        screen.blit(controls, (10, config.DISPLAY_HEIGHT - 30))

        pygame.display.flip()

    # Cleanup
    if async_stylizer is not None:
        async_stylizer.stop()
    if texture_stylizer is not None:
        texture_stylizer.stop()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
