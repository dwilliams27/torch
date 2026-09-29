"""Retro software synthesizer with classic waveforms."""

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass, field


# Standard sample rate
SAMPLE_RATE = 44100

# Waveform types
WaveformType = Literal["sine", "square", "saw", "triangle", "noise"]


def generate_sine(freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def generate_square(freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a square wave (classic chiptune sound)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sign(np.sin(2 * np.pi * freq * t))


def generate_saw(freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a sawtooth wave (aggressive, buzzy)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * (t * freq - np.floor(0.5 + t * freq))


def generate_triangle(freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a triangle wave (softer than square)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * np.abs(2 * (t * freq - np.floor(0.5 + t * freq))) - 1


def generate_noise(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate white noise."""
    return np.random.uniform(-1, 1, int(sample_rate * duration))


def generate_waveform(
    waveform: WaveformType,
    freq: float,
    duration: float,
    sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Generate a waveform of the specified type."""
    if waveform == "sine":
        return generate_sine(freq, duration, sample_rate)
    elif waveform == "square":
        return generate_square(freq, duration, sample_rate)
    elif waveform == "saw":
        return generate_saw(freq, duration, sample_rate)
    elif waveform == "triangle":
        return generate_triangle(freq, duration, sample_rate)
    elif waveform == "noise":
        return generate_noise(duration, sample_rate)
    else:
        return generate_sine(freq, duration, sample_rate)


@dataclass
class Envelope:
    """ADSR envelope for shaping amplitude over time."""
    attack: float = 0.01   # seconds
    decay: float = 0.1     # seconds
    sustain: float = 0.7   # level (0-1)
    release: float = 0.2   # seconds

    def apply(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Apply envelope to samples."""
        total_samples = len(samples)
        envelope = np.ones(total_samples)

        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)

        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay phase
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, total_samples)
        if decay_end > decay_start:
            envelope[decay_start:decay_end] = np.linspace(1, self.sustain, decay_end - decay_start)

        # Sustain phase (already at sustain level)
        sustain_start = decay_end
        sustain_end = max(0, total_samples - release_samples)
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = self.sustain

        # Release phase
        if release_samples > 0 and sustain_end < total_samples:
            envelope[sustain_end:] = np.linspace(self.sustain, 0, total_samples - sustain_end)

        return samples * envelope


@dataclass
class Oscillator:
    """A single oscillator with waveform, frequency, and amplitude."""
    waveform: WaveformType = "sine"
    detune: float = 0.0  # cents (100 cents = 1 semitone)
    amplitude: float = 1.0

    def generate(self, freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Generate samples at the given frequency."""
        # Apply detune
        detuned_freq = freq * (2 ** (self.detune / 1200))
        samples = generate_waveform(self.waveform, detuned_freq, duration, sample_rate)
        return samples * self.amplitude


@dataclass
class Voice:
    """A synthesizer voice with multiple oscillators and envelope."""
    oscillators: list[Oscillator] = field(default_factory=lambda: [Oscillator()])
    envelope: Envelope = field(default_factory=Envelope)

    def play_note(self, freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Generate a note with all oscillators mixed."""
        # Generate from all oscillators
        samples = np.zeros(int(sample_rate * duration))
        for osc in self.oscillators:
            samples += osc.generate(freq, duration, sample_rate)

        # Normalize if multiple oscillators
        if len(self.oscillators) > 1:
            samples /= len(self.oscillators)

        # Apply envelope
        samples = self.envelope.apply(samples, sample_rate)

        return samples


# Musical note frequencies (A4 = 440 Hz)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note_to_freq(note: str, octave: int = 4) -> float:
    """Convert note name to frequency. e.g., 'A4' -> 440.0"""
    if len(note) > 1 and note[-1].isdigit():
        octave = int(note[-1])
        note = note[:-1]

    try:
        semitone = NOTE_NAMES.index(note.upper())
    except ValueError:
        semitone = 9  # Default to A

    # A4 = 440 Hz, calculate relative semitones
    a4_semitone = NOTE_NAMES.index('A') + 4 * 12
    note_semitone = semitone + octave * 12

    return 440.0 * (2 ** ((note_semitone - a4_semitone) / 12))


def freq_to_midi(freq: float) -> int:
    """Convert frequency to MIDI note number."""
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency."""
    return 440.0 * (2 ** ((midi_note - 69) / 12))


# Scale patterns (intervals from root)
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "diminished": [0, 2, 3, 5, 6, 8, 9, 11],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "chromatic": list(range(12)),
}


def get_scale_freqs(root: str, scale: str, octaves: int = 2, base_octave: int = 3) -> list[float]:
    """Get frequencies for a scale starting at root note."""
    root_freq = note_to_freq(root, base_octave)
    root_midi = freq_to_midi(root_freq)

    intervals = SCALES.get(scale, SCALES["minor"])
    freqs = []

    for octave in range(octaves):
        for interval in intervals:
            midi = root_midi + interval + (octave * 12)
            freqs.append(midi_to_freq(midi))

    return freqs


# Effects

def apply_lowpass(samples: np.ndarray, cutoff: float = 0.1) -> np.ndarray:
    """Simple one-pole lowpass filter."""
    output = np.zeros_like(samples)
    output[0] = samples[0]
    for i in range(1, len(samples)):
        output[i] = output[i-1] + cutoff * (samples[i] - output[i-1])
    return output


def apply_delay(samples: np.ndarray, delay_ms: float = 200, feedback: float = 0.3,
                mix: float = 0.3, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply delay effect."""
    delay_samples = int(delay_ms * sample_rate / 1000)
    output = samples.copy()

    for i in range(delay_samples, len(samples)):
        output[i] += output[i - delay_samples] * feedback

    return samples * (1 - mix) + output * mix


def apply_chorus(samples: np.ndarray, depth: float = 0.002, rate: float = 1.5,
                 mix: float = 0.3, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply chorus effect for thickness."""
    t = np.arange(len(samples)) / sample_rate
    mod = depth * sample_rate * np.sin(2 * np.pi * rate * t)

    output = np.zeros_like(samples)
    for i in range(len(samples)):
        delay = int(mod[i])
        idx = i - delay
        if 0 <= idx < len(samples):
            output[i] = samples[idx]

    return samples * (1 - mix) + output * mix
