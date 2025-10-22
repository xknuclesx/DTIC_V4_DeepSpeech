#!/usr/bin/env python3
"""
Transcriptor Modular Speech-to-Text con Múltiples Motores
Soporta: DeepSpeech, Whisper, Silero STT, Vosk
"""

try:
    # Imports básicos
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    import time
    import threading
    from datetime import datetime
    import json
    import queue
    import os
    import sys
    
    # Imports para audio y reconocimiento de voz
    import speech_recognition as sr
    import pyaudio
    import wave
    import io
    
    # VAD (detección de actividad vocal)
    try:
        import webrtcvad
        _VAD_AVAILABLE = True
    except Exception as _vad_err:
        _VAD_AVAILABLE = False
    
    # Imports para ML y detección de fraude
    import joblib
    import re
    import numpy as np
    from collections import deque
    
    # Import del gestor de motores
    from engine_manager import TranscriptionEngineManager
    
    print("✅ Imports básicos cargados correctamente")
    
except ImportError as e:
    print(f"❌ Error importando dependencias básicas: {e}")
    print("💡 Asegúrate de tener instalado: pip install SpeechRecognition pyaudio Flask Flask-SocketIO")
    exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'transcriptor_modular_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

class FraudDetector:
    def __init__(self):
        """Detector de fraude optimizado"""
        try:
            # Cargar modelos ML
            self.model = joblib.load('best_model_lr.joblib')
            self.vectorizer = joblib.load('vectorizer_tfidf.joblib')
            
            # Keywords de fraude mejoradas
            self.fraud_keywords = [
                'dinero fácil', 'dinero rapido', 'ganancia garantizada', 'sin riesgo',
                'inversión segura', 'oportunidad única', 'acción urgente', 'acción inmediata',
                'aprovecha ahora', 'oferta limitada', 'multiplica tu dinero', 'ingresos pasivos',
                'trabajo desde casa', 'gana dinero online', 'millonario en meses',
                'sistema infalible', 'fórmula secreta', 'estrategia ganadora'
            ]
            
            # Configuración de threshold
            self.fraud_threshold = 60  # 60% threshold para detección
            
            print("✅ Detector de fraude cargado correctamente")
            
        except Exception as e:
            print(f"❌ Error cargando detector de fraude: {e}")
            self.model = None
            self.vectorizer = None
    
    def analyze_text(self, text):
        """Analizar texto para detectar fraude"""
        if not self.model or not self.vectorizer or not text:
            return {
                'is_fraud': False,
                'probability': 0,
                'keywords_found': [],
                'status': 'error' if not self.model else 'no_text'
            }
        
        try:
            # Análisis de keywords
            text_lower = text.lower()
            keywords_found = [kw for kw in self.fraud_keywords if kw in text_lower]
            
            # Análisis ML
            text_vectorized = self.vectorizer.transform([text])
            fraud_probability = float(self.model.predict_proba(text_vectorized)[0][1]) * 100
            
            # Determinar si es fraude
            is_fraud = fraud_probability >= self.fraud_threshold
            
            return {
                'is_fraud': is_fraud,
                'probability': round(fraud_probability, 2),
                'keywords_found': keywords_found,
                'status': 'analyzed',
                'threshold': self.fraud_threshold
            }
            
        except Exception as e:
            print(f"❌ Error analizando texto: {e}")
            return {
                'is_fraud': False,
                'probability': 0,
                'keywords_found': [],
                'status': 'error'
            }

class AudioTranscriptor:
    def __init__(self):
        """Inicializar transcriptor de audio modular"""
        
        # Inicializar gestor de motores
        self.engine_manager = TranscriptionEngineManager()
        
        # Configuración de audio por defecto
        self.audio_config = {
            'energy_threshold': 2000,
            'dynamic_energy_threshold': False,
            'dynamic_energy_adjustment_damping': 0.15,
            'pause_threshold': 0.5,
            'listen_timeout': 2,
            'phrase_time_limit': 8,
            'language': 'es-ES',
            # VAD (webrtcvad)
            'vad_enabled': False,
            'vad_aggressiveness': 2,   # 0..3
            'vad_padding_ms': 300,     # pre/post relleno en ms
            'vad_frame_ms': 30,        # 10/20/30 ms soportados
            'sample_rate': 16000       # 16k mono para VAD
        }
        
        # Estado del sistema
        self.is_listening = False
        self.microphone = None
        self.audio_queue = queue.Queue()
        self.listen_thread = None
        
        # Detector de fraude
        self.fraud_detector = FraudDetector()
        
        # Historial de transcripciones
        self.transcription_history = deque(maxlen=50)
        
        # Estadísticas
        self.stats = {
            'total_transcriptions': 0,
            'fraud_detected': 0,
            'session_start': datetime.now()
        }
        
        # Configurar motor por defecto (DeepSpeech)
        self._setup_default_engine()
        
        print("🔧 Configuración de audio aplicada:", self.audio_config)
        
        # Inicializar micrófono
        self._initialize_microphone()
    
    def _setup_default_engine(self):
        """Configurar motor por defecto"""
        config = {'audio': self.audio_config}
        if self.engine_manager.set_engine('deepspeech', config):
            print("✅ Motor DeepSpeech configurado como predeterminado")
        else:
            print("⚠️ Error configurando motor predeterminado")
    
    def _initialize_microphone(self):
        """Inicializar micrófono"""
        try:
            print("🎤 Buscando micrófonos disponibles...")
            self.microphone = sr.Microphone()
            
            # Configurar recognizer con el motor actual
            if self.engine_manager.current_engine:
                recognizer = self.engine_manager.current_engine.recognizer
                
                # Aplicar configuración de audio
                recognizer.energy_threshold = self.audio_config['energy_threshold']
                recognizer.dynamic_energy_threshold = self.audio_config['dynamic_energy_threshold']
                recognizer.dynamic_energy_adjustment_damping = self.audio_config['dynamic_energy_adjustment_damping']
                recognizer.pause_threshold = self.audio_config['pause_threshold']
                
                print("✅ Micrófono inicializado correctamente")
                
                # Calibrar micrófono
                with self.microphone as source:
                    print("🔧 Calibrando micrófono para ruido ambiental...")
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    print(f"✅ Umbral de energía calibrado: {recognizer.energy_threshold}")
            
        except Exception as e:
            print(f"❌ Error inicializando micrófono: {e}")
            self.microphone = None
    
    def change_engine(self, engine_id, engine_config=None):
        """Cambiar motor de transcripción"""
        try:
            # Detener transcripción si está activa
            was_listening = self.is_listening
            if was_listening:
                self.stop_listening()
            
            # Preparar configuración completa
            config = {'audio': self.audio_config}
            if engine_config:
                config.update(engine_config)
            
            # Cambiar motor
            if self.engine_manager.set_engine(engine_id, config):
                # Reinicializar micrófono con nuevo motor
                self._initialize_microphone()
                
                # Reanudar transcripción si estaba activa
                if was_listening:
                    self.start_listening()
                
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error cambiando motor: {e}")
            return False
    
    def get_available_engines(self):
        """Obtener motores disponibles"""
        return self.engine_manager.get_available_engines()
    
    def get_current_engine_info(self):
        """Obtener información del motor actual"""
        return self.engine_manager.get_current_engine_info()
    
    def update_audio_config(self, new_config):
        """Actualizar configuración de audio"""
        try:
            # Actualizar configuración local
            self.audio_config.update(new_config)
            
            # Actualizar motor actual
            audio_config = {'audio': self.audio_config}
            self.engine_manager.update_engine_config(audio_config)
            
            # Reconfigurar recognizer si existe
            if self.engine_manager.current_engine:
                recognizer = self.engine_manager.current_engine.recognizer
                recognizer.energy_threshold = self.audio_config.get('energy_threshold', 2000)
                recognizer.dynamic_energy_threshold = self.audio_config.get('dynamic_energy_threshold', False)
                recognizer.dynamic_energy_adjustment_damping = self.audio_config.get('dynamic_energy_adjustment_damping', 0.15)
                recognizer.pause_threshold = self.audio_config.get('pause_threshold', 0.5)
            
            print(f"🔧 Configuración de audio actualizada: {new_config}")
            return True
            
        except Exception as e:
            print(f"❌ Error actualizando configuración: {e}")
            return False
    
    def start_listening(self):
        """Iniciar transcripción en tiempo real"""
        if self.is_listening:
            return
        
        if not self.microphone or not self.engine_manager.current_engine:
            print("❌ Micrófono o motor no disponible")
            return
        
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        print("🎤 Iniciando transcripción en tiempo real...")
    
    def stop_listening(self):
        """Detener transcripción"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        print("⏹️ Transcripción detenida")
    
    def _listen_loop(self):
        """Loop principal de escucha"""
        if not self.engine_manager.current_engine or not self.microphone:
            print("❌ No hay motor activo o micrófono disponible")
            socketio.emit('error', {'message': 'No hay motor activo o micrófono disponible'})
            return
        
        recognizer = self.engine_manager.current_engine.recognizer
        print(f"🎤 Iniciando loop de escucha con motor: {self.engine_manager.current_engine_name}")
        
        # Emitir estado de escucha iniciado
        socketio.emit('listening_status', {'status': 'started', 'message': 'Escuchando...'})
        
        # Si VAD está habilitado y disponible, usar ruta VAD
        use_vad = bool(self.audio_config.get('vad_enabled', False)) and globals().get('_VAD_AVAILABLE', False)
        if bool(self.audio_config.get('vad_enabled', False)) and not globals().get('_VAD_AVAILABLE', False):
            print("�YY� VAD habilitado pero 'webrtcvad' no est�� instalado. Usando modo est��ndar.")
        if use_vad:
            try:
                self._listen_with_vad()
                print("�o. VAD finalizado")
                return
            except Exception as e:
                print(f"�?O Error en modo VAD: {e}. Cambiando a modo est��ndar.")
                # Continúa a modo estándar (SpeechRecognition)
        
        with self.microphone as source:
            print("🎧 Micrófono abierto, iniciando escucha...")
            
            while self.is_listening:
                try:
                    print("👂 Esperando audio...")
                    
                    # Escuchar audio con timeouts configurados
                    audio_data = recognizer.listen(
                        source,
                        timeout=self.audio_config.get('listen_timeout', 2),
                        phrase_time_limit=self.audio_config.get('phrase_time_limit', 8)
                    )
                    
                    print("🎵 Audio capturado, procesando...")
                    
                    # Procesar en thread separado para no bloquear
                    threading.Thread(
                        target=self._process_audio,
                        args=(audio_data,),
                        daemon=True
                    ).start()
                    
                except sr.WaitTimeoutError:
                    # Timeout normal, continuar escuchando
                    continue
                except sr.RequestError as e:
                    if self.is_listening:
                        print(f"❌ Error de conexión en loop de escucha: {e}")
                        socketio.emit('error', {'message': f'Error de conexión: {e}'})
                    break
                except Exception as e:
                    if self.is_listening:
                        print(f"❌ Error inesperado en loop de escucha: {e}")
                        socketio.emit('error', {'message': f'Error inesperado: {e}'})
                        # Intentar continuar en lugar de romper
                        continue
                        
        print("⏹️ Loop de escucha terminado")
        socketio.emit('listening_status', {'status': 'stopped', 'message': 'Escucha detenida'})
    
    def _process_audio(self, audio_data):
        """Procesar audio transcrito"""
        try:
            print("🔄 Procesando audio capturado...")
            
            # Transcribir usando el motor actual
            text = self.engine_manager.transcribe(audio_data)
            
            if text:
                print(f"📝 Texto transcrito: '{text}'")
                
                # Actualizar estadísticas
                self.stats['total_transcriptions'] += 1
                
                # Análisis de fraude
                fraud_analysis = self.fraud_detector.analyze_text(text)
                if fraud_analysis['is_fraud']:
                    self.stats['fraud_detected'] += 1
                    print(f"⚠️ FRAUDE DETECTADO ({fraud_analysis['probability']}%)")
                
                # Crear resultado
                result = {
                    'text': text,
                    'timestamp': datetime.now().isoformat(),
                    'fraud_analysis': fraud_analysis,
                    'engine_info': self.get_current_engine_info(),
                    'audio_config': self.audio_config,
                    'success': True
                }
                
                # Guardar en historial
                self.transcription_history.append(result)
                
                # Emitir resultado via SocketIO
                socketio.emit('transcription_result', result)
                
                print(f"✅ Resultado enviado via SocketIO")
                
            else:
                print("🔇 No se pudo transcribir el audio (silencio o ruido)")
                # Emitir información de debug
                socketio.emit('transcription_debug', {
                    'message': 'Audio capturado pero no se pudo transcribir (posiblemente silencio)',
                    'timestamp': datetime.now().isoformat(),
                    'engine_info': self.get_current_engine_info()
                })
                
        except Exception as e:
            error_msg = f"Error procesando audio: {e}"
            print(f"❌ {error_msg}")
            
            # Emitir error específico
            socketio.emit('transcription_error', {
                'message': error_msg,
                'timestamp': datetime.now().isoformat(),
                'engine_info': self.get_current_engine_info()
            })

    def _listen_with_vad(self):
        """Capturador continuo con PyAudio + WebRTC VAD, creando segmentos con padding."""
        if not globals().get('_VAD_AVAILABLE', False):
            raise RuntimeError("webrtcvad no disponible")

        vad = webrtcvad.Vad(int(self.audio_config.get('vad_aggressiveness', 2)))
        sample_rate = int(self.audio_config.get('sample_rate', 16000))
        frame_ms = int(self.audio_config.get('vad_frame_ms', 30))
        padding_ms = int(self.audio_config.get('vad_padding_ms', 300))

        if frame_ms not in (10, 20, 30):
            frame_ms = 30

        from collections import deque as _dq
        bytes_per_sample = 2  # 16-bit
        channels = 1
        frame_size = int(sample_rate * frame_ms / 1000)
        bytes_per_frame = frame_size * bytes_per_sample
        num_padding_frames = max(1, int(padding_ms / frame_ms))

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=frame_size
        )
        print(f"�Y'' VAD activo: agg={vad.mode}, frame={frame_ms}ms, padding={padding_ms}ms")

        ring_buffer = _dq(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []

        try:
            while self.is_listening:
                frame = stream.read(frame_size, exception_on_overflow=False)
                if len(frame) != bytes_per_frame:
                    continue
                is_speech = vad.is_speech(frame, sample_rate)

                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = len([1 for f, s in ring_buffer if s])
                    # Disparar cuando haya suficiente voz en el padding
                    if ring_buffer.maxlen and num_voiced > 0.6 * ring_buffer.maxlen:
                        triggered = True
                        print("🎬 VAD: inicio de voz")
                        for f, s in ring_buffer:
                            voiced_frames.append(f)
                        ring_buffer.clear()
                else:
                    # Ya en segmento de voz
                    voiced_frames.append(frame)
                    ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([1 for f, s in ring_buffer if not s])
                    if ring_buffer.maxlen and num_unvoiced > 0.6 * ring_buffer.maxlen:
                        # Fin del segmento; incluir padding de cola implícito en ring_buffer
                        print("🏁 VAD: fin de voz")
                        segment_bytes = b''.join(voiced_frames)
                        voiced_frames = []
                        ring_buffer.clear()
                        triggered = False

                        # Convertir a AudioData (PCM16 mono)
                        audio_data = sr.AudioData(segment_bytes, sample_rate, bytes_per_sample)
                        threading.Thread(
                            target=self._process_audio,
                            args=(audio_data,),
                            daemon=True
                        ).start()

        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

# Instancia global del transcriptor
transcriptor = AudioTranscriptor()

# HTML integrado con selección de motores
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎤 Transcriptor Modular Speech-to-Text</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .main-container { margin-top: 2rem; }
        .card { box-shadow: 0 10px 30px rgba(0,0,0,0.2); border: none; }
        .engine-selector { background: #e3f2fd; border: 2px solid #2196f3; }
        .audio-config { background: #fff9c4; border: 2px solid #ffc107; }
        .transcription-area { background: #f0f8f7; border: 2px solid #4caf50; }
        .fraud-alert { border: 2px solid #f44336; background: #ffebee; }
        .listening-indicator { animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
        .engine-card { transition: all 0.3s; cursor: pointer; }
        .engine-card:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        .engine-card.active { border: 3px solid #28a745; background: #d4edda; }
        .stat-card { background: linear-gradient(45deg, #667eea, #764ba2); color: white; }
    </style>
</head>
<body>
    <div class="container main-container">
        <!-- Header -->
        <div class="row mb-4">
            <div class="col-12 text-center">
                <h1 class="text-white mb-3">
                    <i class="fas fa-microphone"></i>
                    Transcriptor Modular Speech-to-Text
                </h1>
                <p class="text-white-50">Múltiples motores: DeepSpeech • Whisper • Silero • Vosk</p>
            </div>
        </div>

        <!-- Debug Panel -->
        <div class="row mb-3">
            <div class="col-12">
                <div class="card bg-dark text-white" style="max-height: 150px; overflow-y: auto;">
                    <div class="card-header">
                        <h6 class="mb-0">🐛 Debug Logs</h6>
                    </div>
                    <div class="card-body p-2">
                        <div id="debug-logs" style="font-family: monospace; font-size: 12px;">
                            <div>Inicializando...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Selección de Motor -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card engine-selector">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5><i class="fas fa-cogs"></i> Selección de Motor de Transcripción</h5>
                        <button type="button" class="btn btn-sm btn-outline-info" 
                                data-bs-toggle="tooltip" data-bs-placement="left" 
                                title="Cada motor tiene diferentes fortalezas: DeepSpeech (equilibrado), Whisper (alta precisión), Silero (rápido), Vosk (offline/privacidad). Haz clic en una tarjeta para cambiar de motor">
                            <i class="fas fa-info-circle"></i>
                        </button>
                    </div>
                    <div class="card-body">
                        <div class="row" id="engine-grid">
                            <!-- Motores se cargan dinámicamente -->
                        </div>
                        <div class="mt-3">
                            <span class="badge bg-primary me-2">Motor Actual:</span>
                            <span id="current-engine" class="fw-bold">Cargando...</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Configuración de Audio -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card audio-config">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5><i class="fas fa-sliders-h"></i> Configuración de Audio</h5>
                        <button class="btn btn-sm btn-outline-primary" onclick="toggleAudioConfig()">
                            <i class="fas fa-chevron-down" id="audio-chevron"></i>
                        </button>
                    </div>
                    <div class="card-body" id="audio-config-body">
                        <div class="row">
                            <div class="col-md-4">
                                <label class="form-label">
                                    Umbral de Energía
                                    <button type="button" class="btn btn-sm btn-outline-info ms-2" 
                                            data-bs-toggle="tooltip" data-bs-placement="top" 
                                            title="Controla la sensibilidad del micrófono. Valores altos = menos sensible (menos ruido de fondo), valores bajos = más sensible (capta sonidos más suaves)">
                                        <i class="fas fa-info-circle"></i>
                                    </button>
                                </label>
                                <input type="range" class="form-range" id="energy_threshold" min="300" max="4000" value="2000">
                                <small class="text-muted">Valor: <span id="energy_value">2000</span></small>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">
                                    Pausa Entre Frases (s)
                                    <button type="button" class="btn btn-sm btn-outline-info ms-2" 
                                            data-bs-toggle="tooltip" data-bs-placement="top" 
                                            title="Tiempo de silencio que debe pasar para considerar que terminaste de hablar. Valores bajos = respuesta más rápida, valores altos = espera más tiempo">
                                        <i class="fas fa-info-circle"></i>
                                    </button>
                                </label>
                                <input type="range" class="form-range" id="pause_threshold" min="0.1" max="2.0" step="0.1" value="0.5">
                                <small class="text-muted">Valor: <span id="pause_value">0.5</span>s</small>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">
                                    Límite de Frase (s)
                                    <button type="button" class="btn btn-sm btn-outline-info ms-2" 
                                            data-bs-toggle="tooltip" data-bs-placement="top" 
                                            title="Tiempo máximo que puedes hablar sin pausa antes de procesar la transcripción. Útil para frases muy largas">
                                        <i class="fas fa-info-circle"></i>
                                    </button>
                                </label>
                                <input type="range" class="form-range" id="phrase_time_limit" min="3" max="15" value="8">
                                <small class="text-muted">Valor: <span id="phrase_value">8</span>s</small>
                            </div>
                            <div class="col-md-4">
                                <div class="form-check form-switch mt-4">
                                    <input class="form-check-input" type="checkbox" id="vad_enabled">
                                    <label class="form-check-label">
                                        VAD (Detecci��n de Voz)
                                        <button type="button" class="btn btn-sm btn-outline-info ms-2" 
                                                data-bs-toggle="tooltip" data-bs-placement="top" 
                                                title="Usa WebRTC VAD para detectar segmentos de voz con menor latencia y mejor control del ruido">
                                            <i class="fas fa-info-circle"></i>
                                        </button>
                                    </label>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Agresividad VAD (0-3): <span id="vad_agg_value">2</span></label>
                                <input type="range" class="form-range" id="vad_aggressiveness" min="0" max="3" step="1" value="2">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Relleno VAD (ms): <span id="vad_padding_value">300</span></label>
                                <input type="range" class="form-range" id="vad_padding_ms" min="100" max="600" step="50" value="300">
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-4">
                                <label class="form-label">
                                    Timeout de Escucha (s)
                                    <button type="button" class="btn btn-sm btn-outline-info ms-2" 
                                            data-bs-toggle="tooltip" data-bs-placement="top" 
                                            title="Tiempo máximo que el sistema esperará a que empieces a hablar. Si no detecta voz en este tiempo, para la grabación">
                                        <i class="fas fa-info-circle"></i>
                                    </button>
                                </label>
                                <input type="range" class="form-range" id="listen_timeout" min="1" max="5" value="2">
                                <small class="text-muted">Valor: <span id="timeout_value">2</span>s</small>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">
                                    Idioma
                                    <button type="button" class="btn btn-sm btn-outline-info ms-2" 
                                            data-bs-toggle="tooltip" data-bs-placement="top" 
                                            title="Selecciona el idioma para la transcripción. Esto mejora la precisión al reconocer palabras específicas del idioma elegido">
                                        <i class="fas fa-info-circle"></i>
                                    </button>
                                </label>
                                <select class="form-select" id="language">
                                    <option value="es-ES">Español (España)</option>
                                    <option value="es-MX">Español (México)</option>
                                    <option value="en-US">English (US)</option>
                                    <option value="en-GB">English (UK)</option>
                                    <option value="fr-FR">Français</option>
                                    <option value="de-DE">Deutsch</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <div class="form-check form-switch mt-4">
                                    <input class="form-check-input" type="checkbox" id="dynamic_energy_threshold">
                                    <label class="form-check-label">
                                        Ajuste Dinámico de Energía
                                        <button type="button" class="btn btn-sm btn-outline-info ms-2" 
                                                data-bs-toggle="tooltip" data-bs-placement="top" 
                                                title="Cuando está activado, el sistema ajusta automáticamente la sensibilidad del micrófono según el ruido del ambiente. Útil en lugares con ruido variable">
                                            <i class="fas fa-info-circle"></i>
                                        </button>
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Controles Principales -->
        <div class="row mb-4">
            <div class="col-md-8">
                <div class="card transcription-area">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5><i class="fas fa-microphone-alt"></i> Transcripción en Tiempo Real</h5>
                        <div>
                            <button id="startBtn" class="btn btn-success me-2" onclick="startListening()">
                                <i class="fas fa-play"></i> Iniciar
                            </button>
                            <button id="stopBtn" class="btn btn-danger" onclick="stopListening()" disabled>
                                <i class="fas fa-stop"></i> Detener
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="listening-status" class="alert alert-info d-none">
                            <i class="fas fa-microphone listening-indicator"></i>
                            <strong>Escuchando...</strong> Habla ahora
                        </div>
                        <div id="transcription-results" style="max-height: 400px; overflow-y: auto;">
                            <p class="text-muted text-center">Presiona 'Iniciar' para comenzar la transcripción</p>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card stat-card mb-3">
                    <div class="card-body text-center">
                        <h6><i class="fas fa-chart-line"></i> Estadísticas</h6>
                        <p class="mb-1">Transcripciones: <span id="total-transcriptions">0</span></p>
                        <p class="mb-0">Fraudes Detectados: <span id="fraud-detected">0</span></p>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <h6><i class="fas fa-shield-alt"></i> Detección de Fraude</h6>
                    </div>
                    <div class="card-body">
                        <div id="fraud-info">
                            <p class="text-muted">Sistema listo para análisis</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        console.log('🟢 Iniciando JavaScript...');
        
        // Función para mostrar logs visibles en la página
        function debugLog(message) {
            console.log(message);
            const debugArea = document.getElementById('debug-logs');
            if (debugArea) {
                const logEntry = document.createElement('div');
                logEntry.textContent = new Date().toLocaleTimeString() + ': ' + message;
                debugArea.appendChild(logEntry);
                debugArea.scrollTop = debugArea.scrollHeight;
            }
        }
        
        debugLog('🟢 Iniciando JavaScript...');
        
        // Variables globales
        let isListening = false;
        let audioConfigVisible = true;

        // Cargar motores disponibles
        function loadEngines() {
            debugLog('🔍 Cargando motores desde API...');
            
            fetch('/api/engines')
                .then(response => {
                    debugLog('📡 Respuesta recibida: ' + response.status);
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    debugLog('📊 Datos recibidos: ' + JSON.stringify(data));
                    const grid = document.getElementById('engine-grid');
                    
                    if (!grid) {
                        debugLog('❌ No se encontró el elemento engine-grid');
                        return;
                    }
                    
                    grid.innerHTML = '';
                    
                    if (!data.engines || Object.keys(data.engines).length === 0) {
                        debugLog('❌ No hay motores en la respuesta');
                        grid.innerHTML = '<div class="col-12"><p class="text-danger">Error: No se pudieron cargar los motores</p></div>';
                        return;
                    }
                    
                    debugLog(`✅ ${Object.keys(data.engines).length} motores encontrados`);
                    
                    Object.entries(data.engines).forEach(([engineId, engine]) => {
                        debugLog(`🔧 Procesando motor: ${engineId} - ${engine.display_name}`);
                        const engineCard = document.createElement('div');
                        engineCard.className = 'col-md-6 mb-3';
                        
                        // Determinar si el motor está disponible e inicializado
                        const isAvailable = engine.available !== false;
                        const isInitialized = engine.initialized && isAvailable;
                        const isActive = data.current_engine && data.current_engine.engine_id === engineId;
                        
                        const statusText = isInitialized ? 'Listo' : 
                                         isAvailable ? 'Disponible' : 'No instalado';
                        
                        engineCard.innerHTML = `
                            <div class="card engine-card h-100 ${isActive ? 'active' : ''} ${!isAvailable ? 'opacity-50' : ''}" 
                                 onclick="${isAvailable ? `selectEngine('${engineId}')` : `showInstallMessage('${engine.display_name}')`}">
                                <div class="card-body">
                                    <h6 class="card-title">
                                        <i class="fas fa-cog"></i> ${engine.display_name}
                                        ${isActive ? '<span class="badge bg-success ms-2">Activo</span>' : ''}
                                        ${!isAvailable ? '<span class="badge bg-danger ms-2">No instalado</span>' : ''}
                                    </h6>
                                    <p class="card-text small">Estado: ${statusText}</p>
                                    ${engine.description ? `<p class="card-text small text-muted">${engine.description}</p>` : ''}
                                    ${engine.requires_internet ? 
                                        '<p class="card-text small"><i class="fas fa-wifi text-primary"></i> Requiere Internet</p>' : 
                                        '<p class="card-text small"><i class="fas fa-hard-drive text-success"></i> Offline</p>'}
                                </div>
                            </div>
                        `;
                        grid.appendChild(engineCard);
                    });
                    
                    debugLog('✅ Interface de motores actualizada correctamente');
                    
                    // Mostrar motor actual si existe
                    if (data.current_engine) {
                        const currentEngineText = data.current_engine.engine_id || 'Ninguno';
                        const currentEngineElement = document.getElementById('current-engine');
                        if (currentEngineElement) {
                            currentEngineElement.textContent = currentEngineText;
                        }
                    }
                })
                .catch(error => {
                    debugLog('❌ Error cargando motores: ' + error.message);
                    console.error('Error:', error);
                    const grid = document.getElementById('engine-grid');
                    if (grid) {
                        grid.innerHTML = '<div class="col-12"><p class="text-danger">Error cargando motores: ' + error.message + '</p></div>';
                    }
                });
        }

        function selectEngine(engineId) {
            debugLog('🔧 Seleccionando motor: ' + engineId);
            
            // Usar SocketIO para cambiar motor
            socket.emit('change_engine', {engine_id: engineId});
        }

        function showInstallMessage(engineName) {
            showNotification(
                `${engineName} no está instalado. Ejecuta: .\\instalar_motores.bat para instalar todos los motores.`,
                'warning'
            );
        }

        function markActiveEngine(engineId) {
            document.querySelectorAll('.engine-card').forEach(card => {
                card.classList.remove('active');
            });
            
            // Encontrar y marcar la tarjeta activa
            const cards = document.querySelectorAll('.engine-card');
            cards.forEach(card => {
                const onclickAttr = card.getAttribute('onclick');
                if (onclickAttr && onclickAttr.includes(engineId)) {
                    card.classList.add('active');
                }
            });
        }

        function toggleAudioConfig() {
            const body = document.getElementById('audio-config-body');
            const chevron = document.getElementById('audio-chevron');
            
            if (audioConfigVisible) {
                body.style.display = 'none';
                chevron.className = 'fas fa-chevron-right';
            } else {
                body.style.display = 'block';
                chevron.className = 'fas fa-chevron-down';
            }
            audioConfigVisible = !audioConfigVisible;
        }

        function startListening() {
            if (isListening) return;
            
            debugLog('🎤 Iniciando transcripción...');
            
            // Usar SocketIO en lugar de fetch para mejor confiabilidad
            socket.emit('start_listening');
        }

        function stopListening() {
            if (!isListening) return;
            
            debugLog('⏹️ Deteniendo transcripción...');
            
            // Usar SocketIO en lugar de fetch
            socket.emit('stop_listening');
        }

        function updateAudioConfig() {
            const config = {
                energy_threshold: parseInt(document.getElementById('energy_threshold').value),
                pause_threshold: parseFloat(document.getElementById('pause_threshold').value),
                phrase_time_limit: parseInt(document.getElementById('phrase_time_limit').value),
                listen_timeout: parseInt(document.getElementById('listen_timeout').value),
                language: document.getElementById('language').value,
                dynamic_energy_threshold: document.getElementById('dynamic_energy_threshold').checked,
                vad_enabled: document.getElementById('vad_enabled').checked,
                vad_aggressiveness: parseInt(document.getElementById('vad_aggressiveness').value),
                vad_padding_ms: parseInt(document.getElementById('vad_padding_ms').value)
            };

            fetch('/update_audio_config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('Configuración actualizada', 'success');
                } else {
                    showNotification('Error actualizando configuración', 'error');
                }
            })
            .catch(error => console.error('Error:', error));
        }

        function showNotification(message, type) {
            // Crear notificación toast
            const toast = document.createElement('div');
            const alertClass = type === 'success' ? 'success' : 
                              type === 'error' ? 'danger' : 
                              type === 'warning' ? 'warning' : 'info';
            toast.className = `alert alert-${alertClass} position-fixed`;
            toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
            
            const icon = type === 'success' ? '✅' : 
                        type === 'error' ? '❌' : 
                        type === 'warning' ? '⚠️' : 'ℹ️';
            
            toast.innerHTML = `
                <strong>${icon}</strong> ${message}
                <button type="button" class="btn-close float-end" onclick="this.parentElement.remove()"></button>
            `;
            
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 7000); // Más tiempo para warnings
        }

        // Event listeners para configuración de audio
        document.getElementById('energy_threshold').addEventListener('input', function() {
            document.getElementById('energy_value').textContent = this.value;
            updateAudioConfig();
        });

        document.getElementById('pause_threshold').addEventListener('input', function() {
            document.getElementById('pause_value').textContent = this.value;
            updateAudioConfig();
        });

        document.getElementById('phrase_time_limit').addEventListener('input', function() {
            document.getElementById('phrase_value').textContent = this.value;
            updateAudioConfig();
        });

        document.getElementById('listen_timeout').addEventListener('input', function() {
            document.getElementById('timeout_value').textContent = this.value;
            updateAudioConfig();
        });

        document.getElementById('language').addEventListener('change', updateAudioConfig);
        document.getElementById('dynamic_energy_threshold').addEventListener('change', updateAudioConfig);
        // VAD controls
        const _vadEnabledEl = document.getElementById('vad_enabled');
        const _vadAggEl = document.getElementById('vad_aggressiveness');
        const _vadPadEl = document.getElementById('vad_padding_ms');
        if (_vadEnabledEl) _vadEnabledEl.addEventListener('change', updateAudioConfig);
        if (_vadAggEl) _vadAggEl.addEventListener('input', function() {
            const el = document.getElementById('vad_agg_value');
            if (el) el.textContent = this.value;
            updateAudioConfig();
        });
        if (_vadPadEl) _vadPadEl.addEventListener('input', function() {
            const el = document.getElementById('vad_padding_value');
            if (el) el.textContent = this.value;
            updateAudioConfig();
        });

        // Socket.IO eventos - declarados fuera para asegurar que estén listos
        let socket;
        
        // Inicializar Socket.IO inmediatamente
        try {
            socket = io();
            console.log('✅ Socket.IO inicializado correctamente');
            
            // Handler de conexión
            socket.on('connect', function() {
                debugLog('✅ Conectado al servidor via SocketIO');
                showNotification('Conectado al servidor', 'success');
            });
            
            socket.on('disconnect', function() {
                debugLog('❌ Desconectado del servidor');
                showNotification('Desconectado del servidor', 'warning');
                
                // Resetear estado de UI
                isListening = false;
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('listening-status').classList.add('d-none');
            });
            
            // Handler de estado de escucha
            socket.on('listening_status', function(data) {
                debugLog('🎤 Estado de escucha: ' + data.status + ' - ' + data.message);
                
                if (data.status === 'started') {
                    isListening = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('listening-status').classList.remove('d-none');
                    showNotification('Transcripción iniciada', 'success');
                } else if (data.status === 'stopped') {
                    isListening = false;
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    document.getElementById('listening-status').classList.add('d-none');
                    showNotification('Transcripción detenida', 'info');
                }
            });
            
            // Handler de cambio de motor
            socket.on('engine_changed', function(data) {
                debugLog('🔧 Motor cambiado: ' + JSON.stringify(data));
                if (data.success) {
                    showNotification('Motor cambiado exitosamente a ' + data.engine_id, 'success');
                    loadEngines(); // Recargar para actualizar UI
                    
                    // Actualizar indicador de motor actual
                    const currentEngineElement = document.getElementById('current-engine');
                    if (currentEngineElement && data.engine_info) {
                        currentEngineElement.textContent = data.engine_info.name || data.engine_id;
                    }
                }
            });
            
            // Handler de errores
            socket.on('error', function(data) {
                debugLog('❌ Error del servidor: ' + data.message);
                showNotification('Error: ' + data.message, 'error');
            });
            
            // Handler de debug de transcripción
            socket.on('transcription_debug', function(data) {
                debugLog('🔍 Debug transcripción: ' + data.message);
            });
            
            // Handler de errores de transcripción
            socket.on('transcription_error', function(data) {
                debugLog('❌ Error transcripción: ' + data.message);
                showNotification('Error transcribiendo: ' + data.message, 'error');
            });
            
            // Handler de transcripción
            socket.on('transcription_result', function(data) {
                console.log('📝 Resultado de transcripción recibido: ' + data.text);
                debugLog('📝 Transcripción: "' + data.text + '"');
                
                const resultsDiv = document.getElementById('transcription-results');
                
                if (!resultsDiv) {
                    console.log('❌ No se encontró el elemento transcription-results');
                    return;
                }
                
                // Crear elemento de resultado
                const resultDiv = document.createElement('div');
                resultDiv.className = 'mb-3 p-3 border rounded';
                
                const timestamp = new Date(data.timestamp).toLocaleTimeString();
                const fraudClass = data.fraud_analysis.is_fraud ? 'border-danger bg-danger bg-opacity-10' : 'border-success bg-success bg-opacity-10';
                resultDiv.className += ' ' + fraudClass;
                
                const engineInfo = data.engine_info ? `<small class="text-muted">[${data.engine_info.name}]</small>` : '';
                
                resultDiv.innerHTML = `
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <strong>"${data.text}"</strong>
                            ${engineInfo}
                            <div class="mt-1">
                                <small class="text-muted">${timestamp}</small>
                            </div>
                        </div>
                        <div class="text-end">
                            ${data.fraud_analysis.is_fraud ? 
                                `<span class="badge bg-danger">⚠️ FRAUDE</span><br><small>${data.fraud_analysis.probability}%</small>` :
                                `<span class="badge bg-success">✅ NORMAL</span><br><small>${data.fraud_analysis.probability}%</small>`
                            }
                        </div>
                    </div>
                `;
                
                // Insertar al principio
                if (resultsDiv.children.length === 1 && resultsDiv.children[0].classList.contains('text-muted')) {
                    resultsDiv.innerHTML = '';
                }
                resultsDiv.insertBefore(resultDiv, resultsDiv.firstChild);
                
                // Limitar número de resultados mostrados
                if (resultsDiv.children.length > 20) {
                    resultsDiv.removeChild(resultsDiv.lastChild);
                }
                
                // Actualizar estadísticas
                const totalElement = document.getElementById('total-transcriptions');
                if (totalElement) {
                    totalElement.textContent = parseInt(totalElement.textContent || '0') + 1;
                }
                
                if (data.fraud_analysis.is_fraud) {
                    const fraudElement = document.getElementById('fraud-detected');
                    if (fraudElement) {
                        fraudElement.textContent = parseInt(fraudElement.textContent || '0') + 1;
                    }
                }
                
                // Actualizar info de fraude
                const fraudInfo = document.getElementById('fraud-info');
                if (fraudInfo) {
                    if (data.fraud_analysis.is_fraud) {
                        fraudInfo.innerHTML = `
                            <div class="alert alert-danger p-2 mb-2">
                                <strong>⚠️ FRAUDE DETECTADO</strong><br>
                                <small>Probabilidad: ${data.fraud_analysis.probability}%</small><br>
                                <small>Keywords: ${data.fraud_analysis.keywords_found.join(', ')}</small>
                            </div>
                        `;
                    } else {
                        fraudInfo.innerHTML = `
                            <div class="alert alert-success p-2 mb-2">
                                <strong>✅ Texto Normal</strong><br>
                                <small>Probabilidad fraude: ${data.fraud_analysis.probability}%</small>
                            </div>
                        `;
                    }
                }
            });
            
        } catch (error) {
            console.error('❌ Error inicializando Socket.IO: ' + error.message);
        }

        // Cargar datos iniciales
        document.addEventListener('DOMContentLoaded', function() {
            debugLog('🚀 DOMContentLoaded ejecutándose...');
            
            // Inicializar tooltips de Bootstrap
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
            
            // Cargar engines con delay para asegurar que la página esté lista
            setTimeout(function() {
                debugLog('🚀 Iniciando carga de engines...');
                loadEngines();
            }, 1000);
            
            // Cargar configuración de audio actual
            fetch('/api/audio_config')
                .then(response => response.json())
                .then(data => {
                    if (data.config) {
                        const config = data.config;
                        document.getElementById('energy_threshold').value = config.energy_threshold || 2000;
                        document.getElementById('energy_value').textContent = config.energy_threshold || 2000;
                        document.getElementById('pause_threshold').value = config.pause_threshold || 0.5;
                        document.getElementById('pause_value').textContent = config.pause_threshold || 0.5;
                        document.getElementById('phrase_time_limit').value = config.phrase_time_limit || 8;
                        document.getElementById('phrase_value').textContent = config.phrase_time_limit || 8;
                        document.getElementById('listen_timeout').value = config.listen_timeout || 2;
                        document.getElementById('timeout_value').textContent = config.listen_timeout || 2;
                        document.getElementById('language').value = config.language || 'es-ES';
                        document.getElementById('dynamic_energy_threshold').checked = config.dynamic_energy_threshold || false;
                        // VAD fields
                        const vadEnabledEl = document.getElementById('vad_enabled');
                        const vadAggEl = document.getElementById('vad_aggressiveness');
                        const vadPadEl = document.getElementById('vad_padding_ms');
                        const vadAggValEl = document.getElementById('vad_agg_value');
                        const vadPadValEl = document.getElementById('vad_padding_value');
                        if (vadEnabledEl) vadEnabledEl.checked = !!config.vad_enabled;
                        if (vadAggEl) { vadAggEl.value = (config.vad_aggressiveness ?? 2); if (vadAggValEl) vadAggValEl.textContent = vadAggEl.value; }
                        if (vadPadEl) { vadPadEl.value = (config.vad_padding_ms ?? 300); if (vadPadValEl) vadPadValEl.textContent = vadPadEl.value; }
                    }
                })
                .catch(error => console.error('Error cargando configuración:', error));
        });
    </script>

    <!-- Script adicional para cargar motores - FUNCIONAL -->
    <script>
        console.log('🟢 Script adicional de engines iniciando...');
        
        // Función funcional para cargar engines
        function loadEnginesWorking() {
            console.log('🔍 Cargando motores desde API...');
            
            fetch('/api/engines')
                .then(response => {
                    console.log('📡 Respuesta recibida:', response.status);
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('📊 Datos recibidos:', data);
                    const grid = document.getElementById('engine-grid');
                    
                    if (!grid) {
                        console.log('❌ No se encontró el elemento engine-grid');
                        return;
                    }
                    
                    grid.innerHTML = '';
                    
                    if (!data.engines || Object.keys(data.engines).length === 0) {
                        console.log('❌ No hay motores en la respuesta');
                        grid.innerHTML = '<div class="col-12"><p class="text-danger">Error: No se pudieron cargar los motores</p></div>';
                        return;
                    }
                    
                    console.log(`✅ ${Object.keys(data.engines).length} motores encontrados`);
                    
                    Object.entries(data.engines).forEach(([engineId, engine]) => {
                        console.log(`🔧 Procesando motor: ${engineId} - ${engine.display_name}`);
                        const engineCard = document.createElement('div');
                        engineCard.className = 'col-md-6 col-lg-3 mb-3';
                        
                        // Determinar si el motor está disponible e inicializado
                        const isAvailable = engine.available !== false;
                        const isActive = data.current_engine && data.current_engine.engine_id === engineId;
                        
                        engineCard.innerHTML = `
                            <div class="card engine-card h-100 ${isActive ? 'active' : ''} ${!isAvailable ? 'opacity-50' : ''}" 
                                 onclick="${isAvailable ? `selectEngineWorking('${engineId}')` : `alert('Motor ${engine.display_name} no está instalado')`}">
                                <div class="card-body text-center">
                                    <h6 class="card-title">
                                        <i class="fas fa-microphone"></i> ${engine.display_name}
                                        ${isActive ? '<span class="badge bg-success ms-2">Activo</span>' : ''}
                                        ${!isAvailable ? '<span class="badge bg-danger ms-2">No instalado</span>' : ''}
                                    </h6>
                                    <p class="card-text small">Estado: ${isAvailable ? 'Disponible' : 'No instalado'}</p>
                                    ${engine.description ? `<p class="card-text small text-muted">${engine.description}</p>` : ''}
                                </div>
                            </div>
                        `;
                        grid.appendChild(engineCard);
                    });
                    
                    console.log('✅ Interface de motores actualizada correctamente');
                })
                .catch(error => {
                    console.error('❌ Error cargando motores:', error);
                    const grid = document.getElementById('engine-grid');
                    if (grid) {
                        grid.innerHTML = '<div class="col-12"><p class="text-danger">Error cargando motores: ' + error.message + '</p></div>';
                    }
                });
        }
        
        // Función funcional para seleccionar engine
        function selectEngineWorking(engineId) {
            console.log('🎯 Seleccionando engine:', engineId);
            
            fetch('/api/change_engine', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ engine_id: engineId })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(result => {
                console.log('✅ Engine seleccionado:', result);
                // Recargar la lista para actualizar el estado activo
                loadEnginesWorking();
            })
            .catch(error => {
                console.error('❌ Error seleccionando engine:', error);
                alert('Error al seleccionar el motor: ' + error.message);
            });
        }
        
        // Inicializar cuando el DOM esté listo
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 DOM listo, iniciando carga de engines en 2 segundos...');
            
            setTimeout(function() {
                loadEnginesWorking();
            }, 2000);
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Página principal"""
    return HTML_TEMPLATE

@app.route('/debug')
def debug_page():
    """Página de debug simple"""
    with open('debug_simple.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/simple')
def simple_page():
    """Página simple del transcriptor"""
    with open('transcriptor_simple.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/working')
def working_page():
    """Página funcional del transcriptor"""
    with open('transcriptor_working.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/engines')
def get_engines():
    """API para obtener motores disponibles"""
    try:
        engines = transcriptor.get_available_engines()
        current_engine = transcriptor.get_current_engine_info()
        
        return jsonify({
            'success': True,
            'engines': engines,
            'current_engine': current_engine
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/change_engine', methods=['POST'])
def change_engine():
    """API para cambiar motor de transcripción"""
    try:
        data = request.get_json()
        engine_id = data.get('engine_id')
        engine_config = data.get('config', {})
        
        if transcriptor.change_engine(engine_id, engine_config):
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'No se pudo cambiar el motor'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/audio_config')
def get_audio_config():
    """API para obtener configuración de audio actual"""
    try:
        return jsonify({
            'success': True,
            'config': transcriptor.audio_config
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start_listening', methods=['POST'])
def start_listening():
    """Iniciar transcripción"""
    try:
        transcriptor.start_listening()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_listening', methods=['POST'])
def stop_listening():
    """Detener transcripción"""
    try:
        transcriptor.stop_listening()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/update_audio_config', methods=['POST'])
def update_audio_config():
    """Actualizar configuración de audio"""
    try:
        config = request.get_json()
        
        # Validar y convertir tipos
        if 'energy_threshold' in config:
            config['energy_threshold'] = int(config['energy_threshold'])
        if 'pause_threshold' in config:
            config['pause_threshold'] = float(config['pause_threshold'])
        if 'phrase_time_limit' in config:
            config['phrase_time_limit'] = int(config['phrase_time_limit'])
        if 'listen_timeout' in config:
            config['listen_timeout'] = int(config['listen_timeout'])
        if 'dynamic_energy_threshold' in config:
            config['dynamic_energy_threshold'] = bool(config['dynamic_energy_threshold'])
        # VAD fields
        if 'vad_enabled' in config:
            config['vad_enabled'] = bool(config['vad_enabled'])
        if 'vad_aggressiveness' in config:
            config['vad_aggressiveness'] = int(config['vad_aggressiveness'])
        if 'vad_padding_ms' in config:
            config['vad_padding_ms'] = int(config['vad_padding_ms'])
        
        success = transcriptor.update_audio_config(config)
        return jsonify({'success': success})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stats')
def get_stats():
    """Obtener estadísticas del sistema"""
    try:
        return jsonify({
            'success': True,
            'stats': transcriptor.stats,
            'history_count': len(transcriptor.transcription_history)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test')
def test_engines():
    """Página de test para debuggear la carga de engines"""
    try:
        with open('test_engines.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except:
        return "<h1>Error: No se encontró test_engines.html</h1>"

# ========== EVENTOS SOCKETIO NECESARIOS ==========

@socketio.on('connect')
def handle_connect():
    """Cliente conectado via SocketIO"""
    print(f"✅ Cliente conectado via SocketIO")
    emit('connection_status', {'status': 'connected', 'message': 'Conexión SocketIO establecida'})

@socketio.on('disconnect') 
def handle_disconnect():
    """Cliente desconectado via SocketIO"""
    print(f"❌ Cliente desconectado via SocketIO")

@socketio.on('start_listening')
def handle_start_listening():
    """Manejar inicio de transcripción via SocketIO"""
    try:
        print("🎤 Iniciando transcripción via SocketIO...")
        if not transcriptor.microphone or not transcriptor.engine_manager.current_engine:
            emit('error', {'message': 'Micrófono o motor no disponible'})
            return
        
        transcriptor.start_listening()
        emit('listening_status', {'status': 'started', 'message': 'Transcripción iniciada correctamente'})
        print("✅ Transcripción iniciada via SocketIO")
        
    except Exception as e:
        error_msg = f"Error iniciando transcripción: {str(e)}"
        print(f"❌ {error_msg}")
        emit('error', {'message': error_msg})

@socketio.on('stop_listening')
def handle_stop_listening():
    """Manejar detención de transcripción via SocketIO"""
    try:
        print("⏹️ Deteniendo transcripción via SocketIO...")
        transcriptor.stop_listening()
        emit('listening_status', {'status': 'stopped', 'message': 'Transcripción detenida correctamente'})
        print("✅ Transcripción detenida via SocketIO")
        
    except Exception as e:
        error_msg = f"Error deteniendo transcripción: {str(e)}"
        print(f"❌ {error_msg}")
        emit('error', {'message': error_msg})

@socketio.on('change_engine')
def handle_change_engine(data):
    """Manejar cambio de motor via SocketIO"""
    try:
        engine_id = data.get('engine_id')
        if not engine_id:
            emit('error', {'message': 'ID de motor no especificado'})
            return
        
        print(f"🔧 Cambiando motor a: {engine_id}")
        
        # Detener transcripción si está activa
        was_listening = transcriptor.is_listening
        if was_listening:
            transcriptor.stop_listening()
        
        # Cambiar motor
        if transcriptor.change_engine(engine_id):
            emit('engine_changed', {
                'success': True,
                'engine_id': engine_id,
                'engine_info': transcriptor.get_current_engine_info()
            })
            
            # Reanudar si estaba transcribiendo
            if was_listening:
                transcriptor.start_listening()
                emit('listening_status', {'status': 'started', 'message': f'Transcripción reanudada con {engine_id}'})
            
            print(f"✅ Motor cambiado exitosamente a: {engine_id}")
        else:
            emit('error', {'message': f'No se pudo cambiar al motor {engine_id}'})
            
    except Exception as e:
        error_msg = f"Error cambiando motor: {str(e)}"
        print(f"❌ {error_msg}")
        emit('error', {'message': error_msg})

@socketio.on('update_audio_config')
def handle_update_audio_config(data):
    """Manejar actualización de configuración de audio via SocketIO"""
    try:
        print(f"🔧 Actualizando configuración de audio: {data}")
        
        # Validar y convertir tipos
        config = {}
        if 'energy_threshold' in data:
            config['energy_threshold'] = int(data['energy_threshold'])
        if 'pause_threshold' in data:
            config['pause_threshold'] = float(data['pause_threshold'])
        if 'phrase_time_limit' in data:
            config['phrase_time_limit'] = int(data['phrase_time_limit'])
        if 'listen_timeout' in data:
            config['listen_timeout'] = int(data['listen_timeout'])
        if 'language' in data:
            config['language'] = str(data['language'])
        if 'dynamic_energy_threshold' in data:
            config['dynamic_energy_threshold'] = bool(data['dynamic_energy_threshold'])
        # VAD fields
        if 'vad_enabled' in data:
            config['vad_enabled'] = bool(data['vad_enabled'])
        if 'vad_aggressiveness' in data:
            config['vad_aggressiveness'] = int(data['vad_aggressiveness'])
        if 'vad_padding_ms' in data:
            config['vad_padding_ms'] = int(data['vad_padding_ms'])
        
        if transcriptor.update_audio_config(config):
            emit('config_updated', {'success': True, 'config': config})
            print("✅ Configuración de audio actualizada")
        else:
            emit('error', {'message': 'No se pudo actualizar la configuración'})
            
    except Exception as e:
        error_msg = f"Error actualizando configuración: {str(e)}"
        print(f"❌ {error_msg}")
        emit('error', {'message': error_msg})

@socketio.on('get_engines')
def handle_get_engines():
    """Obtener lista de motores via SocketIO"""
    try:
        engines = transcriptor.get_available_engines()
        current_engine = transcriptor.get_current_engine_info()
        
        emit('engines_list', {
            'engines': engines,
            'current_engine': current_engine
        })
        
    except Exception as e:
        print(f"❌ Error obteniendo motores: {e}")
        emit('error', {'message': f'Error obteniendo motores: {str(e)}'})

@socketio.on('get_stats')
def handle_get_stats():
    """Obtener estadísticas via SocketIO"""
    try:
        emit('stats_update', {
            'stats': transcriptor.stats,
            'history_count': len(transcriptor.transcription_history),
            'current_engine': transcriptor.get_current_engine_info()
        })
        
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")
        emit('error', {'message': f'Error obteniendo estadísticas: {str(e)}'})

# ========== FIN EVENTOS SOCKETIO ==========

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 TRANSCRIPTOR MODULAR SPEECH-TO-TEXT")
    print("🔧 Motores disponibles: DeepSpeech • Whisper • Silero • Vosk")
    print("🎤 Panel de configuración de audio FUNCIONAL")
    print("🌐 URL: http://localhost:5003")
    print("="*80)
    
    try:
        socketio.run(app, host='0.0.0.0', port=5003, debug=False)
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo servidor...")
        transcriptor.stop_listening()
        transcriptor.engine_manager.cleanup()
        print("✅ Servidor detenido correctamente")
