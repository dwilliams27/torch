"""Real-time audio engine using sounddevice."""

import threading
import queue
import numpy as np
from typing import Optional, Callable

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not installed. Audio disabled. Install with: pip install sounddevice")

from synth import SAMPLE_RATE


class AudioEngine:
    """Real-time audio engine with streaming output."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False
        self.stream = None

        # Audio buffer - ring buffer for continuous playback
        self.buffer_length = sample_rate * 2  # 2 seconds buffer
        self.audio_buffer = np.zeros(self.buffer_length, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()

        # Volume control
        self.master_volume = 0.5

        # Generator function for continuous audio
        self.generator: Optional[Callable[[], np.ndarray]] = None

    def start(self):
        """Start the audio stream."""
        if not SOUNDDEVICE_AVAILABLE:
            print("Audio disabled - sounddevice not available")
            return

        if self.running:
            return

        self.running = True

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.buffer_size,
                callback=self._audio_callback,
            )
            self.stream.start()
            print(f"Audio engine started (sample rate: {self.sample_rate})")
        except Exception as e:
            print(f"Failed to start audio: {e}")
            self.running = False

    def stop(self):
        """Stop the audio stream."""
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"Audio status: {status}")

        # Generate new audio if we have a generator
        if self.generator is not None:
            try:
                new_samples = self.generator()
                if new_samples is not None and len(new_samples) > 0:
                    self._write_to_buffer(new_samples)
            except Exception as e:
                print(f"Generator error: {e}")

        # Read from buffer
        samples = self._read_from_buffer(frames)
        outdata[:, 0] = samples * self.master_volume

    def _write_to_buffer(self, samples: np.ndarray):
        """Write samples to the ring buffer."""
        with self.lock:
            samples = samples.astype(np.float32)
            n = len(samples)

            # Handle wrap-around
            end_pos = self.write_pos + n
            if end_pos <= self.buffer_length:
                self.audio_buffer[self.write_pos:end_pos] = samples
            else:
                first_part = self.buffer_length - self.write_pos
                self.audio_buffer[self.write_pos:] = samples[:first_part]
                self.audio_buffer[:n - first_part] = samples[first_part:]

            self.write_pos = end_pos % self.buffer_length

    def _read_from_buffer(self, frames: int) -> np.ndarray:
        """Read samples from the ring buffer."""
        with self.lock:
            output = np.zeros(frames, dtype=np.float32)

            # Calculate available samples
            if self.write_pos >= self.read_pos:
                available = self.write_pos - self.read_pos
            else:
                available = self.buffer_length - self.read_pos + self.write_pos

            # Read what we can
            to_read = min(frames, available)
            if to_read > 0:
                end_pos = self.read_pos + to_read
                if end_pos <= self.buffer_length:
                    output[:to_read] = self.audio_buffer[self.read_pos:end_pos]
                else:
                    first_part = self.buffer_length - self.read_pos
                    output[:first_part] = self.audio_buffer[self.read_pos:]
                    output[first_part:to_read] = self.audio_buffer[:to_read - first_part]

                self.read_pos = end_pos % self.buffer_length

            return output

    def set_generator(self, generator: Callable[[], np.ndarray]):
        """Set the audio generator function."""
        self.generator = generator

    def set_volume(self, volume: float):
        """Set master volume (0.0 to 1.0)."""
        self.master_volume = max(0.0, min(1.0, volume))

    def play_samples(self, samples: np.ndarray):
        """Queue samples for playback."""
        self._write_to_buffer(samples)


class SimpleAudioEngine:
    """Simpler audio engine that generates audio in blocks."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.running = False
        self.stream = None
        self.master_volume = 0.5

        # Generator for continuous audio
        self.generator: Optional[Callable[[int], np.ndarray]] = None

        # Debug counter
        self.callback_count = 0

    def start(self):
        """Start the audio stream."""
        if not SOUNDDEVICE_AVAILABLE:
            print("Audio disabled - sounddevice not available")
            return False

        if self.running:
            return True

        self.running = True
        self.callback_count = 0

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=2048,
                callback=self._callback,
            )
            self.stream.start()
            print(f"Audio started (sample rate: {self.sample_rate})")
            return True
        except Exception as e:
            print(f"Failed to start audio: {e}")
            self.running = False
            return False

    def stop(self):
        """Stop the audio stream."""
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None

    def _callback(self, outdata, frames, time_info, status):
        """Audio callback - fills output buffer."""
        self.callback_count += 1

        if status:
            print(f"Audio status: {status}")

        # Default to silence
        outdata.fill(0)

        # Try to get samples from generator
        if self.generator is not None and self.running:
            try:
                samples = self.generator(frames)
                if samples is not None and len(samples) > 0:
                    n = min(len(samples), frames)
                    # Ensure float32 and proper shape
                    samples = np.asarray(samples, dtype=np.float32)
                    outdata[:n, 0] = samples[:n] * self.master_volume
            except Exception as e:
                if self.callback_count < 5:
                    print(f"Audio generator error: {e}")

    def set_generator(self, gen: Callable[[int], np.ndarray]):
        """Set audio generator function. Called with (num_samples) -> samples."""
        self.generator = gen

    def set_volume(self, vol: float):
        """Set volume 0.0 - 1.0."""
        self.master_volume = max(0.0, min(1.0, vol))
