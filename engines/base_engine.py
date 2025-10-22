"""
Clase base abstracta para todos los motores de transcripción Speech-to-Text
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import speech_recognition as sr

class BaseTranscriptionEngine(ABC):
    """
    Clase base para todos los motores de transcripción.
    Define la interfaz común que deben implementar todos los engines.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el motor de transcripción
        
        Args:
            config: Configuración específica del motor
        """
        self.config = config or {}
        self.is_initialized = False
        self.recognizer = sr.Recognizer()
        self._setup_audio_config()
    
    def _setup_audio_config(self):
        """Configurar parámetros de audio comunes"""
        audio_config = self.config.get('audio', {})
        
        # Configuración básica de audio
        self.recognizer.energy_threshold = audio_config.get('energy_threshold', 2000)
        self.recognizer.dynamic_energy_threshold = audio_config.get('dynamic_energy_threshold', False)
        self.recognizer.dynamic_energy_adjustment_damping = audio_config.get('dynamic_energy_adjustment_damping', 0.15)
        self.recognizer.pause_threshold = audio_config.get('pause_threshold', 0.5)
        
        # Timeouts
        self.listen_timeout = audio_config.get('listen_timeout', 2)
        self.phrase_time_limit = audio_config.get('phrase_time_limit', 8)
        
        # Idioma
        self.language = audio_config.get('language', 'es-ES')
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Inicializar el motor específico
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        pass
    
    @abstractmethod
    def transcribe_audio(self, audio_data: sr.AudioData) -> Optional[str]:
        """
        Transcribir audio usando el motor específico
        
        Args:
            audio_data: Datos de audio para transcribir
            
        Returns:
            str: Texto transcrito o None si hay error
        """
        pass
    
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Obtener información del motor
        
        Returns:
            dict: Información del motor (nombre, versión, etc.)
        """
        pass
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Actualizar configuración del motor
        
        Args:
            new_config: Nueva configuración
        """
        self.config.update(new_config)
        self._setup_audio_config()
    
    def get_supported_languages(self) -> List[str]:
        """
        Obtener lista de idiomas soportados
        
        Returns:
            list: Lista de códigos de idioma soportados
        """
        return ['es-ES', 'en-US', 'en-GB', 'fr-FR', 'de-DE', 'it-IT', 'pt-PT']
    
    def is_ready(self) -> bool:
        """
        Verificar si el motor está listo para transcribir
        
        Returns:
            bool: True si está listo
        """
        return self.is_initialized
    
    def cleanup(self):
        """Limpiar recursos del motor"""
        pass
