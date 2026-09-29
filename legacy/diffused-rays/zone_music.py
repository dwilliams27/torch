"""Zone-based procedural music system.

Each zone has its own musical personality - key, scale, tempo, waveforms.
Music smoothly transitions as player moves between zones.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from synth import (
    SAMPLE_RATE, Voice, Oscillator, Envelope,
    get_scale_freqs, midi_to_freq, apply_lowpass, apply_delay, apply_chorus,
    WaveformType
)
from audio_engine import SimpleAudioEngine, SOUNDDEVICE_AVAILABLE


@dataclass
class ZoneMusicParams:
    """Musical parameters for a zone."""
    name: str
    root_note: str          # e.g., "D", "F#"
    scale: str              # e.g., "minor", "pentatonic"
    base_octave: int        # Starting octave
    tempo: float            # BPM
    waveforms: list[WaveformType]  # Waveforms to layer
    waveform_weights: list[float]  # Relative volumes
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.6
    release: float = 0.3
    detune: float = 0.0     # cents
    lowpass: float = 1.0    # 0-1, lower = more filtered
    delay_mix: float = 0.0  # 0-1
    chorus_mix: float = 0.0 # 0-1
    arp_pattern: list[int] = field(default_factory=lambda: [0, 2, 4])  # Scale degrees
    drone_notes: list[int] = field(default_factory=lambda: [0])  # Scale degrees for drone


# Zone music configurations
ZONE_MUSIC = {
    # 0: Torch - Warm, mysterious dungeon
    0: ZoneMusicParams(
        name="Torch",
        root_note="D",
        scale="minor",
        base_octave=3,
        tempo=75,
        waveforms=["triangle", "sine"],
        waveform_weights=[0.6, 0.4],
        attack=0.05,
        decay=0.2,
        sustain=0.5,
        release=0.4,
        lowpass=0.6,
        delay_mix=0.2,
        arp_pattern=[0, 2, 4, 5, 4, 2],
        drone_notes=[0, 4],
    ),

    # 1: Cyberpunk - Pulsing, synthetic
    1: ZoneMusicParams(
        name="Cyberpunk",
        root_note="A",
        scale="minor",
        base_octave=3,
        tempo=120,
        waveforms=["saw", "square"],
        waveform_weights=[0.5, 0.5],
        attack=0.005,
        decay=0.1,
        sustain=0.4,
        release=0.1,
        detune=8.0,
        lowpass=0.7,
        delay_mix=0.15,
        arp_pattern=[0, 0, 3, 5, 7, 5, 3, 0],
        drone_notes=[0],
    ),

    # 2: Underwater - Deep, flowing
    2: ZoneMusicParams(
        name="Underwater",
        root_note="C",
        scale="lydian",
        base_octave=2,
        tempo=50,
        waveforms=["sine", "triangle"],
        waveform_weights=[0.7, 0.3],
        attack=0.2,
        decay=0.3,
        sustain=0.6,
        release=0.8,
        lowpass=0.4,
        delay_mix=0.4,
        chorus_mix=0.5,
        arp_pattern=[0, 4, 7, 11, 7, 4],
        drone_notes=[0, 7],
    ),

    # 3: Hellscape - Aggressive, tense
    3: ZoneMusicParams(
        name="Hellscape",
        root_note="E",
        scale="diminished",
        base_octave=3,
        tempo=140,
        waveforms=["square", "saw"],
        waveform_weights=[0.6, 0.4],
        attack=0.002,
        decay=0.05,
        sustain=0.3,
        release=0.1,
        detune=15.0,
        lowpass=0.8,
        delay_mix=0.1,
        arp_pattern=[0, 1, 3, 6, 8, 6, 3, 1],
        drone_notes=[0, 6],
    ),

    # 4: Ice Cave - Ethereal, cold, crystalline
    4: ZoneMusicParams(
        name="Ice Cave",
        root_note="F#",
        scale="major",
        base_octave=4,
        tempo=55,
        waveforms=["sine", "triangle"],
        waveform_weights=[0.5, 0.5],
        attack=0.1,
        decay=0.2,
        sustain=0.7,
        release=0.6,
        lowpass=0.9,
        delay_mix=0.5,
        chorus_mix=0.3,
        arp_pattern=[0, 4, 7, 11, 14, 11, 7, 4],
        drone_notes=[0, 7, 14],
    ),

    # 5: Overgrown - Organic, peaceful
    5: ZoneMusicParams(
        name="Overgrown",
        root_note="G",
        scale="pentatonic",
        base_octave=3,
        tempo=65,
        waveforms=["triangle", "sine"],
        waveform_weights=[0.4, 0.6],
        attack=0.08,
        decay=0.15,
        sustain=0.6,
        release=0.5,
        lowpass=0.5,
        delay_mix=0.3,
        chorus_mix=0.2,
        arp_pattern=[0, 2, 4, 7, 9, 7, 4, 2],
        drone_notes=[0, 4, 9],
    ),

    # 6: Oil Paint - Dreamy, impressionist
    6: ZoneMusicParams(
        name="Oil Paint",
        root_note="Bb",
        scale="major",
        base_octave=3,
        tempo=60,
        waveforms=["sine", "sine", "triangle"],
        waveform_weights=[0.4, 0.3, 0.3],
        attack=0.15,
        decay=0.3,
        sustain=0.5,
        release=0.7,
        detune=5.0,
        lowpass=0.45,
        delay_mix=0.35,
        chorus_mix=0.4,
        arp_pattern=[0, 2, 4, 7, 9, 11, 9, 7],
        drone_notes=[0, 4, 7],
    ),

    # 7: Anime - Upbeat, energetic (bonus style)
    7: ZoneMusicParams(
        name="Anime",
        root_note="C",
        scale="major",
        base_octave=4,
        tempo=130,
        waveforms=["square", "triangle"],
        waveform_weights=[0.5, 0.5],
        attack=0.01,
        decay=0.1,
        sustain=0.5,
        release=0.15,
        lowpass=0.85,
        delay_mix=0.1,
        arp_pattern=[0, 4, 7, 12, 7, 4, 0, -5],
        drone_notes=[0],
    ),
}


class ZoneMusicPlayer:
    """Plays procedural music based on current zone."""

    def __init__(self):
        self.engine = SimpleAudioEngine()
        self.current_zone = 0
        self.target_zone = 0
        self.transition_progress = 1.0  # 1.0 = fully transitioned
        self.transition_speed = 0.5  # How fast to transition (0-1 per second)

        # Sequencer state
        self.time_started = 0.0
        self.samples_generated = 0
        self.current_beat = 0
        self.arp_index = 0

        # Pre-calculate scale frequencies for current zone
        self.scale_freqs: list[float] = []
        self._update_scale()

        # Audio state
        self.running = False

    def _update_scale(self):
        """Update scale frequencies for current zone."""
        params = ZONE_MUSIC.get(self.current_zone, ZONE_MUSIC[0])
        # Get 3 octaves of the scale
        self.scale_freqs = get_scale_freqs(
            params.root_note,
            params.scale,
            octaves=3,
            base_octave=params.base_octave
        )

    def start(self):
        """Start the music player."""
        if not SOUNDDEVICE_AVAILABLE:
            print("Music disabled - sounddevice not available")
            return False

        self.time_started = time.time()
        self.samples_generated = 0
        self.running = True  # Set running BEFORE starting engine
        self.engine.set_generator(self._generate_audio)
        self.engine.set_volume(0.4)

        if self.engine.start():
            return True
        self.running = False
        return False

    def stop(self):
        """Stop the music player."""
        self.running = False
        self.engine.stop()

    def set_zone(self, zone: int):
        """Set target zone - music will transition smoothly."""
        if zone != self.target_zone:
            self.target_zone = zone
            self.transition_progress = 0.0

    def set_volume(self, volume: float):
        """Set music volume (0.0 - 1.0)."""
        self.engine.set_volume(volume)

    def _generate_audio(self, num_samples: int) -> np.ndarray:
        """Generate audio samples - called by audio engine."""
        # ALWAYS return exactly num_samples
        output = np.zeros(num_samples, dtype=np.float32)

        if not self.running:
            return output

        try:
            # Handle zone transitions
            if self.transition_progress < 1.0:
                self.transition_progress += self.transition_speed * (num_samples / SAMPLE_RATE)
                if self.transition_progress >= 1.0:
                    self.transition_progress = 1.0
                    if self.current_zone != self.target_zone:
                        self.current_zone = self.target_zone
                        self._update_scale()

            params = ZONE_MUSIC.get(self.current_zone, ZONE_MUSIC[0])
            beat_duration = 60.0 / params.tempo
            samples_per_beat = int(SAMPLE_RATE * beat_duration)

            # Generate arpeggio - simple sine-based for now
            arp = self._simple_arpeggio(num_samples, params)

            # Generate drone
            drone = self._simple_drone(num_samples, params)

            # Mix
            output = arp * 0.5 + drone * 0.5

            # Soft clip
            output = np.tanh(output)

            self.samples_generated += num_samples

        except Exception as e:
            # On any error, return silence
            pass

        return output.astype(np.float32)

    def _simple_arpeggio(self, num_samples: int, params: ZoneMusicParams) -> np.ndarray:
        """Generate simple arpeggio - guaranteed to return exact size."""
        output = np.zeros(num_samples, dtype=np.float32)

        if len(self.scale_freqs) == 0:
            return output

        # Time array for this chunk
        start_time = self.samples_generated / SAMPLE_RATE
        t = np.arange(num_samples) / SAMPLE_RATE + start_time

        # Note timing
        note_duration = 60.0 / params.tempo / 4  # sixteenth notes

        # Which note in the pattern are we on?
        pattern = params.arp_pattern

        for i in range(num_samples):
            time_pos = start_time + i / SAMPLE_RATE
            note_idx = int(time_pos / note_duration) % len(pattern)
            scale_degree = pattern[note_idx]

            # Get frequency
            if scale_degree < 0:
                freq_idx = max(0, len(self.scale_freqs) // 3 + scale_degree)
            else:
                freq_idx = min(abs(scale_degree), len(self.scale_freqs) - 1)

            freq = self.scale_freqs[freq_idx]

            # Generate sample based on waveform
            phase = 2 * np.pi * freq * time_pos

            # Mix waveforms
            sample = 0.0
            for wf, weight in zip(params.waveforms, params.waveform_weights):
                if wf == "sine":
                    sample += weight * np.sin(phase)
                elif wf == "square":
                    sample += weight * (1.0 if np.sin(phase) > 0 else -1.0)
                elif wf == "saw":
                    sample += weight * (2 * (time_pos * freq - np.floor(0.5 + time_pos * freq)))
                elif wf == "triangle":
                    sample += weight * (2 * abs(2 * (time_pos * freq - np.floor(0.5 + time_pos * freq))) - 1)

            # Simple envelope based on position within note
            note_phase = (time_pos % note_duration) / note_duration
            envelope = 1.0 - note_phase  # Simple decay

            output[i] = sample * envelope * 0.3

        return output

    def _simple_drone(self, num_samples: int, params: ZoneMusicParams) -> np.ndarray:
        """Generate simple drone - guaranteed to return exact size."""
        output = np.zeros(num_samples, dtype=np.float32)

        if len(self.scale_freqs) == 0:
            return output

        start_time = self.samples_generated / SAMPLE_RATE

        # Drone on root note
        freq = self.scale_freqs[0]

        for i in range(num_samples):
            time_pos = start_time + i / SAMPLE_RATE
            # Slow sine drone with vibrato
            vibrato = 1 + 0.003 * np.sin(2 * np.pi * 4 * time_pos)
            output[i] = 0.2 * np.sin(2 * np.pi * freq * vibrato * time_pos)

        return output

    def _generate_arpeggio(self, num_samples: int, params: ZoneMusicParams,
                          samples_per_beat: int) -> np.ndarray:
        """Generate arpeggio pattern."""
        output = np.zeros(num_samples, dtype=np.float32)

        # Note duration (fraction of beat)
        note_duration = 0.25  # sixteenth notes
        samples_per_note = int(samples_per_beat * note_duration)

        if samples_per_note < 100:
            samples_per_note = 100

        # Create voice for arpeggio
        oscillators = []
        for wf, weight in zip(params.waveforms, params.waveform_weights):
            oscillators.append(Oscillator(
                waveform=wf,
                amplitude=weight,
                detune=params.detune if len(oscillators) > 0 else 0
            ))

        voice = Voice(
            oscillators=oscillators,
            envelope=Envelope(
                attack=params.attack,
                decay=params.decay,
                sustain=params.sustain,
                release=params.release
            )
        )

        # Generate notes
        sample_pos = 0
        while sample_pos < num_samples:
            # Get current note from pattern
            pattern = params.arp_pattern
            note_idx = self.arp_index % len(pattern)
            scale_degree = pattern[note_idx]

            # Handle negative scale degrees (go down)
            if scale_degree < 0:
                freq_idx = max(0, len(self.scale_freqs) // 3 + scale_degree)
            else:
                freq_idx = min(scale_degree, len(self.scale_freqs) - 1)

            if freq_idx < len(self.scale_freqs):
                freq = self.scale_freqs[freq_idx]

                # Generate note
                note_samples = min(samples_per_note, num_samples - sample_pos)
                duration = note_samples / SAMPLE_RATE

                note = voice.play_note(freq, duration)

                # Add to output
                end_pos = min(sample_pos + len(note), num_samples)
                output[sample_pos:end_pos] += note[:end_pos - sample_pos]

            sample_pos += samples_per_note
            self.arp_index += 1

        return output

    def _generate_drone(self, num_samples: int, params: ZoneMusicParams) -> np.ndarray:
        """Generate drone/pad sound."""
        output = np.zeros(num_samples, dtype=np.float32)

        t = np.arange(num_samples) / SAMPLE_RATE
        start_time = self.samples_generated / SAMPLE_RATE

        # Generate drone from specified notes
        for i, scale_degree in enumerate(params.drone_notes):
            if scale_degree < len(self.scale_freqs):
                freq = self.scale_freqs[scale_degree]

                # Use sine wave for smooth drone
                # Add slight vibrato
                vibrato = 1 + 0.003 * np.sin(2 * np.pi * 4 * (start_time + t) + i)
                drone = np.sin(2 * np.pi * freq * vibrato * (start_time + t))

                # Amplitude envelope for smooth entry
                if start_time < 1.0:
                    fade_in = np.minimum(1.0, (start_time + t))
                    drone *= fade_in

                output += drone / len(params.drone_notes)

        # Apply lowpass for warmth
        output = apply_lowpass(output, 0.15)

        return output * 0.5


# Singleton instance for easy access
_music_player: Optional[ZoneMusicPlayer] = None


def get_music_player() -> ZoneMusicPlayer:
    """Get or create the music player singleton."""
    global _music_player
    if _music_player is None:
        _music_player = ZoneMusicPlayer()
    return _music_player


def start_music() -> bool:
    """Start the music system."""
    return get_music_player().start()


def stop_music():
    """Stop the music system."""
    if _music_player is not None:
        _music_player.stop()


def set_music_zone(zone: int):
    """Set the current music zone."""
    get_music_player().set_zone(zone)


def set_music_volume(volume: float):
    """Set music volume (0.0 - 1.0)."""
    get_music_player().set_volume(volume)
