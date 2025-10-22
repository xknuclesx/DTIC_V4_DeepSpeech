"""
Módulo de motores de transcripción Speech-to-Text
Soporta múltiples engines: DeepSpeech, Whisper, Silero, Vosk
"""

from .base_engine import BaseTranscriptionEngine
from .deepspeech_engine import DeepSpeechEngine
from .whisper_engine import WhisperEngine
from .silero_engine import SileroEngine
from .vosk_engine import VoskEngine

__all__ = [
    'BaseTranscriptionEngine',
    'DeepSpeechEngine', 
    'WhisperEngine',
    'SileroEngine',
    'VoskEngine'
]
