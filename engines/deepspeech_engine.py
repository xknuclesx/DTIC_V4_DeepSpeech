"""
Motor de transcripción usando Google Speech Recognition (DeepSpeech compatible)
"""

from typing import Dict, Any, Optional
import speech_recognition as sr
from .base_engine import BaseTranscriptionEngine

class DeepSpeechEngine(BaseTranscriptionEngine):
    """
    Motor de transcripción usando Google Speech Recognition API.
    Mantiene compatibilidad con la implementación anterior "DeepSpeech".
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar motor DeepSpeech (Google SR)"""
        super().__init__(config or {})
        self.engine_name = "DeepSpeech (Google SR)"
        self.engine_version = "1.0"
    
    def initialize(self) -> bool:
        """
        Inicializar el motor DeepSpeech
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Verificar que el recognizer esté disponible
            with sr.Microphone() as source:
                # Test rápido de inicialización
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando DeepSpeech Engine: {e}")
            self.is_initialized = False
            return False
    
    def transcribe_audio(self, audio_data: sr.AudioData) -> Optional[str]:
        """
        Transcribir audio usando Google Speech Recognition
        
        Args:
            audio_data: Datos de audio para transcribir
            
        Returns:
            str: Texto transcrito o None si hay error
        """
        try:
            # Usar Google Speech Recognition
            text = self.recognizer.recognize_google(
                audio_data, 
                language=self.language
            )
            return text.strip() if text else None
            
        except sr.UnknownValueError:
            return None  # No se pudo entender el audio
        except sr.RequestError as e:
            print(f"❌ Error en DeepSpeech Engine: {e}")
            return None
        except Exception as e:
            print(f"❌ Error inesperado en DeepSpeech: {e}")
            return None
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Obtener información del motor DeepSpeech
        
        Returns:
            dict: Información del motor
        """
        return {
            'name': self.engine_name,
            'version': self.engine_version,
            'description': 'Motor de transcripción usando Google Speech Recognition API',
            'supported_languages': self.get_supported_languages(),
            'requires_internet': True,
            'accuracy': 'Alta',
            'speed': 'Rápida',
            'dependencies': ['speech_recognition', 'pyaudio']
        }
    
    def get_supported_languages(self) -> list:
        """
        Idiomas soportados por Google Speech Recognition
        
        Returns:
            list: Lista de códigos de idioma
        """
        return [
            'es-ES', 'es-MX', 'es-AR', 'es-CO', 'es-PE', 'es-VE',
            'en-US', 'en-GB', 'en-AU', 'en-CA', 'en-IN',
            'fr-FR', 'fr-CA', 'de-DE', 'it-IT', 'pt-BR', 'pt-PT',
            'ru-RU', 'ja-JP', 'ko-KR', 'zh-CN', 'zh-TW', 'ar-SA'
        ]
