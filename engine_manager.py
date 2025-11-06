"""
Gestor de motores de transcripción Speech-to-Text
"""

from typing import Dict, Any, Optional, List
import importlib
from engines.base_engine import BaseTranscriptionEngine

class TranscriptionEngineManager:
    """
    Gestor centralizado para todos los motores de transcripción
    """
    
    def __init__(self):
        """Inicializar el gestor de motores"""
        self.engines = {}
        self.current_engine = None
        self.current_engine_name = None
        self._load_engines()
    
    def _load_engines(self):
        """Cargar todos los motores disponibles"""
        engine_configs = {
            'deepspeech': {
                'module': 'engines.deepspeech_engine',
                'class': 'DeepSpeechEngine',
                'display_name': 'DeepSpeech (Google SR)',
                'description': 'Motor rápido usando Google Speech Recognition API',
                'requires_internet': True
            },
            'whisper': {
                'module': 'engines.whisper_engine',
                'class': 'WhisperEngine', 
                'display_name': 'OpenAI Whisper',
                'description': 'Motor de alta precisión usando OpenAI Whisper',
                'requires_internet': False
            },
            'silero': {
                'module': 'engines.silero_engine',
                'class': 'SileroEngine',
                'display_name': 'Silero STT',
                'description': 'Motor rápido y eficiente usando Silero',
                'requires_internet': False
            }
        }
        
        for engine_id, config in engine_configs.items():
            try:
                # Importar módulo dinámicamente
                module = importlib.import_module(config['module'])
                engine_class = getattr(module, config['class'])
                
                # Crear instancia del motor y verificar que se puede inicializar
                engine_instance = engine_class()
                
                # Verificar disponibilidad real intentando inicializar
                try:
                    can_initialize = engine_instance.initialize()
                    if can_initialize:
                        available = True
                        status_msg = "cargado y disponible"
                    else:
                        available = False
                        status_msg = "cargado pero no disponible (dependencias faltantes)"
                except KeyboardInterrupt:
                    # No capturar KeyboardInterrupt - dejarlo propagarse
                    raise
                except Exception as init_error:
                    available = False
                    status_msg = f"cargado pero no disponible: {str(init_error)}"
                
                self.engines[engine_id] = {
                    'instance': engine_instance,
                    'config': config,
                    'initialized': can_initialize if available else False,
                    'available': available
                }
                
                if available:
                    print(f"✅ Motor '{config['display_name']}' {status_msg}")
                else:
                    print(f"⚠️ Motor '{config['display_name']}' {status_msg}")
                
            except Exception as e:
                # Motor no se puede cargar en absoluto
                self.engines[engine_id] = {
                    'instance': None,
                    'config': config,
                    'initialized': False,
                    'available': False,
                    'error': str(e)
                }
                print(f"⚠️ Motor '{config['display_name']}' no disponible: {e}")
                if "No module named" in str(e):
                    print(f"💡 Para instalar: .\\instalar_motores.bat")
    
    def get_available_engines(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener lista de motores disponibles
        
        Returns:
            dict: Diccionario con información de motores disponibles
        """
        available = {}
        for engine_id, engine_data in self.engines.items():
            if not engine_data.get('available', True):
                # Motor no disponible
                available[engine_id] = {
                    'display_name': engine_data['config']['display_name'],
                    'description': engine_data['config']['description'] + " (⚠️ No instalado)",
                    'requires_internet': engine_data['config']['requires_internet'],
                    'initialized': False,
                    'available': False,
                    'error': engine_data.get('error', 'No disponible')
                }
                continue
                
            try:
                engine_info = engine_data['instance'].get_engine_info()
                available[engine_id] = {
                    'display_name': engine_data['config']['display_name'],
                    'description': engine_data['config']['description'],
                    'requires_internet': engine_data['config']['requires_internet'],
                    'initialized': engine_data['initialized'],
                    'available': True,
                    'info': engine_info
                }
            except Exception as e:
                print(f"❌ Error obteniendo info de {engine_id}: {e}")
        
        return available
    
    def set_engine(self, engine_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Establecer el motor activo
        
        Args:
            engine_id: ID del motor a activar
            config: Configuración opcional para el motor
            
        Returns:
            bool: True si el motor se activó correctamente
        """
        if engine_id not in self.engines:
            print(f"❌ Motor '{engine_id}' no disponible")
            return False
        
        engine_data = self.engines[engine_id]
        
        # Verificar si el motor está disponible
        if not engine_data.get('available', True):
            print(f"❌ Motor '{engine_id}' no está instalado")
            print(f"💡 Error: {engine_data.get('error', 'No disponible')}")
            print(f"💡 Para instalar: .\\instalar_motores.bat")
            return False
        
        try:
            engine_data = self.engines[engine_id]
            engine_instance = engine_data['instance']
            
            # Actualizar configuración si se proporciona
            if config:
                engine_instance.update_config(config)
            
            # Inicializar motor si no está inicializado
            if not engine_data['initialized']:
                if not engine_instance.initialize():
                    print(f"❌ Error inicializando motor '{engine_id}'")
                    return False
                engine_data['initialized'] = True
            
            # Limpiar motor anterior si existe
            if self.current_engine:
                self.current_engine.cleanup()
            
            # Establecer nuevo motor activo
            self.current_engine = engine_instance
            self.current_engine_name = engine_id
            
            print(f"✅ Motor activo: {engine_data['config']['display_name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error activando motor '{engine_id}': {e}")
            return False
    
    def transcribe(self, audio_data) -> Optional[str]:
        """
        Transcribir audio usando el motor activo
        
        Args:
            audio_data: Datos de audio para transcribir
            
        Returns:
            str: Texto transcrito o None si hay error
        """
        if not self.current_engine or not self.current_engine.is_ready():
            print("❌ No hay motor activo o no está listo")
            return None
        
        try:
            return self.current_engine.transcribe_audio(audio_data)
        except Exception as e:
            print(f"❌ Error transcribiendo: {e}")
            return None
    
    def get_current_engine_info(self) -> Optional[Dict[str, Any]]:
        """
        Obtener información del motor activo
        
        Returns:
            dict: Información del motor activo o None
        """
        if not self.current_engine:
            return None
        
        try:
            info = self.current_engine.get_engine_info()
            info['engine_id'] = self.current_engine_name
            return info
        except Exception as e:
            print(f"❌ Error obteniendo info del motor activo: {e}")
            return None
    
    def update_engine_config(self, config: Dict[str, Any]) -> bool:
        """
        Actualizar configuración del motor activo
        
        Args:
            config: Nueva configuración
            
        Returns:
            bool: True si se actualizó correctamente
        """
        if not self.current_engine:
            return False
        
        try:
            self.current_engine.update_config(config)
            return True
        except Exception as e:
            print(f"❌ Error actualizando configuración: {e}")
            return False
    
    def get_supported_languages(self) -> List[str]:
        """
        Obtener idiomas soportados por el motor activo
        
        Returns:
            list: Lista de códigos de idioma soportados
        """
        if not self.current_engine:
            return []
        
        try:
            return self.current_engine.get_supported_languages()
        except Exception as e:
            print(f"❌ Error obteniendo idiomas soportados: {e}")
            return []
    
    def cleanup(self):
        """Limpiar recursos de todos los motores"""
        for engine_data in self.engines.values():
            try:
                if engine_data['initialized']:
                    engine_data['instance'].cleanup()
            except Exception as e:
                print(f"❌ Error limpiando motor: {e}")
        
        self.current_engine = None
        self.current_engine_name = None
