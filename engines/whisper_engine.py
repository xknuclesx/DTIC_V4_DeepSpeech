"""
Motor de transcripción usando OpenAI Whisper y Faster-Whisper
"""

from typing import Dict, Any, Optional
import speech_recognition as sr
from .base_engine import BaseTranscriptionEngine

class WhisperEngine(BaseTranscriptionEngine):
    """
    Motor de transcripción usando OpenAI Whisper y Faster-Whisper
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar motor Whisper"""
        super().__init__(config or {})
        self.engine_name = "OpenAI Whisper"
        self.engine_version = "1.0"
        self.whisper_model = None
        self.use_faster_whisper = self.config.get('use_faster_whisper', True)
        self.model_size = self.config.get('model_size', 'base')
    
    def initialize(self) -> bool:
        """
        Inicializar el motor Whisper
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            if self.use_faster_whisper:
                # Verificar primero si torch está disponible (faster-whisper lo requiere)
                try:
                    import torch
                except ImportError:
                    print("⚠️ torch no disponible, faster-whisper requiere torch")
                    self.use_faster_whisper = False
                
                # Intentar usar faster-whisper si torch está disponible
                if self.use_faster_whisper:
                    try:
                        from faster_whisper import WhisperModel
                        self.whisper_model = WhisperModel(
                            self.model_size, 
                            device="cpu",
                            compute_type="int8"
                        )
                        print(f"✅ Faster-Whisper modelo '{self.model_size}' cargado")
                    except Exception as e:
                        print(f"⚠️ faster-whisper no disponible: {type(e).__name__}")
                        self.use_faster_whisper = False
            
            if not self.use_faster_whisper:
                # Usar whisper estándar
                try:
                    import whisper
                    self.whisper_model = whisper.load_model(self.model_size)
                    print(f"✅ Whisper modelo '{self.model_size}' cargado")
                except Exception as e:
                    print(f"❌ No se pudo importar whisper: {type(e).__name__}")
                    return False
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando Whisper Engine: {e}")
            self.is_initialized = False
            return False
    
    def transcribe_audio(self, audio_data: sr.AudioData) -> Optional[str]:
        """
        Transcribir audio usando Whisper
        
        Args:
            audio_data: Datos de audio para transcribir
            
        Returns:
            str: Texto transcrito o None si hay error
        """
        if not self.is_initialized or not self.whisper_model:
            return None
        
        try:
            # Convertir audio a formato numpy
            import numpy as np
            audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
            
            if self.use_faster_whisper:
                # Usar faster-whisper
                segments, info = self.whisper_model.transcribe(
                    audio_np,
                    language="es" if self.language.startswith('es') else "en"
                )
                text = " ".join([segment.text for segment in segments])
            else:
                # Usar whisper estándar
                result = self.whisper_model.transcribe(
                    audio_np,
                    language="es" if self.language.startswith('es') else "en"
                )
                text = result["text"]
            
            return text.strip() if text else None
            
        except Exception as e:
            print(f"❌ Error en Whisper Engine: {e}")
            return None
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Obtener información del motor Whisper
        
        Returns:
            dict: Información del motor
        """
        return {
            'name': self.engine_name,
            'version': self.engine_version,
            'description': 'Motor de transcripción usando OpenAI Whisper',
            'model_type': 'faster-whisper' if self.use_faster_whisper else 'whisper',
            'model_size': self.model_size,
            'supported_languages': self.get_supported_languages(),
            'requires_internet': False,
            'accuracy': 'Muy Alta',
            'speed': 'Media' if self.use_faster_whisper else 'Lenta',
            'dependencies': ['whisper', 'faster-whisper', 'torch']
        }
    
    def get_supported_languages(self) -> list:
        """
        Idiomas soportados por Whisper
        
        Returns:
            list: Lista de códigos de idioma
        """
        return [
            'es', 'en', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'hi', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi'
        ]
    
    def get_available_models(self) -> list:
        """
        Obtener modelos disponibles de Whisper
        
        Returns:
            list: Lista de tamaños de modelo disponibles
        """
        return ['tiny', 'base', 'small', 'medium', 'large']
