"""
Motor de transcripción usando Vosk (Kaldi)
"""

import os
from typing import Dict, Any, Optional
import speech_recognition as sr
from .base_engine import BaseTranscriptionEngine

class VoskEngine(BaseTranscriptionEngine):
    """
    Motor de transcripción usando Vosk (basado en Kaldi)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar motor Vosk"""
        super().__init__(config or {})
        self.engine_name = "Vosk (Kaldi)"
        self.engine_version = "1.0"
        self.vosk_model = None
        self.vosk_rec = None
        self.model_path = self.config.get('model_path', None)
        self.model_language = self.config.get('model_language', 'es')
    
    def initialize(self) -> bool:
        """
        Inicializar el motor Vosk
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            import vosk
            import json
            
            # Configurar log level de Vosk
            vosk.SetLogLevel(-1)  # Silenciar logs de Vosk
            
            # Cargar modelo Vosk
            if self.model_path and os.path.exists(self.model_path):
                # Usar modelo personalizado
                self.vosk_model = vosk.Model(self.model_path)
                print(f"✅ Modelo Vosk personalizado cargado: {self.model_path}")
            else:
                # Intentar usar modelo por defecto
                try:
                    # Buscar modelo por defecto en directorio común
                    import os
                    default_models = {
                        'es': 'vosk-model-small-es-0.42',  # Usando modelo pequeño
                        'en': 'vosk-model-en-us-0.22',
                        'de': 'vosk-model-de-0.21',
                        'fr': 'vosk-model-fr-0.22'
                    }
                    
                    model_name = default_models.get(self.model_language, 'vosk-model-small-es-0.42')
                    model_path = os.path.join(os.getcwd(), 'models', model_name)
                    
                    if os.path.exists(model_path):
                        self.vosk_model = vosk.Model(model_path)
                        print(f"✅ Modelo Vosk por defecto cargado: {model_name}")
                    else:
                        print(f"⚠️ Modelo Vosk no encontrado en: {model_path}")
                        print("💡 Descarga modelos desde: https://alphacephei.com/vosk/models")
                        return False
                        
                except Exception as e:
                    print(f"❌ Error cargando modelo Vosk: {e}")
                    return False
            
            # Crear recognizer con sample rate de 16000Hz
            self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, 16000)
            
            self.is_initialized = True
            return True
            
        except ImportError as e:
            print(f"❌ Error importando Vosk: {e}")
            print("💡 Instalar con: pip install vosk")
            self.is_initialized = False
            return False
        except Exception as e:
            print(f"❌ Error inicializando Vosk Engine: {e}")
            self.is_initialized = False
            return False
    
    def transcribe_audio(self, audio_data: sr.AudioData) -> Optional[str]:
        """
        Transcribir audio usando Vosk
        
        Args:
            audio_data: Datos de audio para transcribir
            
        Returns:
            str: Texto transcrito o None si hay error
        """
        if not self.is_initialized or not self.vosk_rec:
            return None
        
        try:
            import json
            import numpy as np
            
            # Convertir audio a PCM 16-bit mono 16kHz
            audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
            
            # Resamplear a 16kHz si es necesario
            if audio_data.sample_rate != 16000:
                import scipy.signal
                audio_np = scipy.signal.resample(
                    audio_np, 
                    int(len(audio_np) * 16000 / audio_data.sample_rate)
                ).astype(np.int16)
            
            # Procesar audio con Vosk
            audio_bytes = audio_np.tobytes()
            
            # Procesar en chunks para mejor rendimiento
            if self.vosk_rec.AcceptWaveform(audio_bytes):
                result = json.loads(self.vosk_rec.Result())
                text = result.get('text', '')
            else:
                result = json.loads(self.vosk_rec.PartialResult())
                text = result.get('partial', '')
            
            return text.strip() if text else None
            
        except Exception as e:
            print(f"❌ Error en Vosk Engine: {e}")
            return None
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Obtener información del motor Vosk
        
        Returns:
            dict: Información del motor
        """
        return {
            'name': self.engine_name,
            'version': self.engine_version,
            'description': 'Motor de transcripción usando Vosk (basado en Kaldi)',
            'model_language': self.model_language,
            'model_path': self.model_path,
            'supported_languages': self.get_supported_languages(),
            'requires_internet': False,
            'accuracy': 'Alta',
            'speed': 'Muy Rápida',
            'dependencies': ['vosk', 'scipy']
        }
    
    def get_supported_languages(self) -> list:
        """
        Idiomas soportados por Vosk
        
        Returns:
            list: Lista de códigos de idioma
        """
        return [
            'es', 'en', 'de', 'fr', 'it', 'pt', 'ru', 'uk', 'kz', 'ja', 'ko', 'zh',
            'ar', 'fa', 'hi', 'tr', 'vn', 'ca', 'eu', 'br', 'cs', 'nl', 'pl'
        ]
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Obtener modelos disponibles de Vosk
        
        Returns:
            dict: Diccionario con idioma -> nombre del modelo
        """
        return {
            'es': 'vosk-model-es-0.42',
            'en': 'vosk-model-en-us-0.22',
            'de': 'vosk-model-de-0.21',
            'fr': 'vosk-model-fr-0.22',
            'it': 'vosk-model-it-0.22',
            'pt': 'vosk-model-pt-0.3',
            'ru': 'vosk-model-ru-0.42'
        }
