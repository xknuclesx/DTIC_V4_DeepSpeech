#!/usr/bin/env python3
"""
Transcriptor Modular Speech-to-Text con Múltiples Motores
Soporta: DeepSpeech, Whisper, Silero STT
"""

# Suprimir warnings de deprecación para salida limpia
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pkg_resources')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')

# Suprimir warnings de sklearn version mismatch (modelos compatibles)
import os
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

try:
    # Imports básicos
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    import time
    import threading
    from datetime import datetime
    import json
    import queue
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
    
    # Import del gestor de sentimientos
    from sentiment_manager import SentimentEngineManager
    
    print("[OK] Imports básicos cargados correctamente")
    
except ImportError as e:
    print(f"[ERROR] Error importando dependencias básicas: {e}")
    print("[INFO] Asegúrate de tener instalado: pip install SpeechRecognition pyaudio Flask Flask-SocketIO")
    exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'transcriptor_modular_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

class AdaptiveThreshold:
    """
    Sistema de thresholds dinámicos con calibración automática
    Ajusta los umbrales según el contexto y performance histórica
    MEJORA 6: Thresholds Dinámicos con Calibración
    """
    
    def __init__(self):
        # Múltiples perfiles de thresholds según nivel de seguridad requerido
        self.thresholds = {
            'high_security': {  # Banca, datos sensibles, temas críticos
                'fraud': 0.40,      # Threshold bajo = más sensible
                'warning': 0.30,
                'monitor': 0.20
            },
            'medium_security': {  # General, uso estándar
                'fraud': 0.60,
                'warning': 0.45,
                'monitor': 0.30
            },
            'low_security': {  # Conversación casual, bajo riesgo
                'fraud': 0.75,      # Threshold alto = menos sensible
                'warning': 0.60,
                'monitor': 0.45
            }
        }
        
        # Thresholds por defecto (antes del sistema adaptativo)
        self.default_thresholds = {
            'critical': 0.75,
            'high': 0.60,
            'medium': 0.45,
            'low': 0.30
        }
        
        # Log de performance para auto-calibración
        self.performance_log = deque(maxlen=1000)
        
        # Estadísticas de calibración
        self.calibration_stats = {
            'total_predictions': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'last_calibration': None,
            'calibration_count': 0
        }
        
        # Configuración de auto-calibración
        self.auto_calibration_enabled = True
        self.calibration_interval = 100  # Recalibrar cada 100 casos
        self.max_threshold_adjustment = 0.10  # Máximo ajuste por calibración
        
        print("[OK] AdaptiveThreshold inicializado con 3 perfiles de seguridad")
    
    def detect_security_context(self, text, keyword_analysis=None):
        """
        Detectar automáticamente el nivel de seguridad requerido según el contenido
        
        Args:
            text (str): Texto a analizar
            keyword_analysis (dict): Análisis de keywords contextuales
        
        Returns:
            str: 'high_security', 'medium_security' o 'low_security'
        """
        text_lower = text.lower()
        
        # Palabras clave que indican alto riesgo
        high_risk_keywords = [
            'banco', 'tarjeta', 'cuenta', 'cvv', 'pin', 'contraseña',
            'clave', 'token', 'otp', 'transferencia', 'pago',
            'seguridad social', 'pasaporte', 'cédula', 'dni'
        ]
        
        # Contar keywords de alto riesgo
        high_risk_count = sum(1 for kw in high_risk_keywords if kw in text_lower)
        
        # Si hay análisis de keywords, verificar categorías críticas
        if keyword_analysis and keyword_analysis.get('categories'):
            critical_categories = ['bancarias', 'datos_sensibles']
            has_critical = any(
                cat in keyword_analysis['categories'] 
                for cat in critical_categories
            )
            if has_critical:
                return 'high_security'
        
        # Clasificar según cantidad de keywords de alto riesgo
        if high_risk_count >= 2:
            return 'high_security'
        elif high_risk_count >= 1:
            return 'medium_security'
        else:
            return 'low_security'
    
    def classify(self, score, context='medium_security', text=None, keyword_analysis=None):
        """
        Clasificar el score según thresholds adaptativos
        
        Args:
            score (float): Score de vishing (0-1)
            context (str): Contexto de seguridad o 'auto' para detección automática
            text (str): Texto para detección automática de contexto
            keyword_analysis (dict): Análisis de keywords para contexto
        
        Returns:
            tuple: (clasificación, nivel_riesgo, threshold_usado, contexto_usado)
        """
        # Auto-detectar contexto si es necesario
        if context == 'auto' and text:
            context = self.detect_security_context(text, keyword_analysis)
        
        # Validar contexto
        if context not in self.thresholds:
            context = 'medium_security'
        
        thresholds = self.thresholds[context]
        
        # Clasificar según thresholds del contexto
        if score >= thresholds['fraud']:
            classification = 'FRAUDE'
            risk_level = 'CRÍTICO'
            threshold_used = thresholds['fraud']
        elif score >= thresholds['warning']:
            classification = 'SOSPECHOSO'
            risk_level = 'ALTO'
            threshold_used = thresholds['warning']
        elif score >= thresholds['monitor']:
            classification = 'MONITOREAR'
            risk_level = 'MEDIO'
            threshold_used = thresholds['monitor']
        else:
            classification = 'NORMAL'
            risk_level = 'BAJO'
            threshold_used = 0.0
        
        return classification, risk_level, threshold_used, context
    
    def log_performance(self, prediction, actual_label, score, context='medium_security'):
        """
        Registrar predicción para auto-calibración
        
        Args:
            prediction (str): Predicción realizada ('FRAUDE', 'SOSPECHOSO', etc.)
            actual_label (bool): Etiqueta real (True = fraude, False = legítimo)
            score (float): Score que generó la predicción
            context (str): Contexto de seguridad usado
        """
        import time
        
        self.performance_log.append({
            'prediction': prediction,
            'actual': actual_label,
            'score': score,
            'context': context,
            'timestamp': time.time()
        })
        
        # Actualizar estadísticas
        self.calibration_stats['total_predictions'] += 1
        
        if prediction in ['FRAUDE', 'SOSPECHOSO'] and actual_label:
            self.calibration_stats['true_positives'] += 1
        elif prediction in ['FRAUDE', 'SOSPECHOSO'] and not actual_label:
            self.calibration_stats['false_positives'] += 1
        elif prediction in ['MONITOREAR', 'NORMAL'] and actual_label:
            self.calibration_stats['false_negatives'] += 1
        elif prediction in ['MONITOREAR', 'NORMAL'] and not actual_label:
            self.calibration_stats['true_negatives'] += 1
        
        # Auto-calibrar si es necesario
        if (self.auto_calibration_enabled and 
            len(self.performance_log) % self.calibration_interval == 0 and
            len(self.performance_log) >= self.calibration_interval):
            self._recalibrate()
    
    def _recalibrate(self):
        """
        Ajustar thresholds según precision/recall del log de performance
        """
        import time
        
        if len(self.performance_log) < 10:  # Mínimo 10 casos para calibrar
            return
        
        # Calcular métricas
        tp = self.calibration_stats['true_positives']
        fp = self.calibration_stats['false_positives']
        fn = self.calibration_stats['false_negatives']
        tn = self.calibration_stats['true_negatives']
        
        total = tp + fp + fn + tn
        if total == 0:
            return
        
        # Calcular precision y recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n[CALIBRATION] Recalibrando thresholds...")
        print(f"[CALIBRATION] Casos analizados: {total}")
        print(f"[CALIBRATION] Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}%")
        print(f"[CALIBRATION] TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        
        # Estrategia de ajuste
        adjustment_made = False
        
        # Si muchos falsos positivos (precision baja) → AUMENTAR thresholds
        if fp > tp * 0.3 and precision < 0.7:  # Más de 30% FP y precision < 70%
            adjustment = min(0.05, self.max_threshold_adjustment)
            for context in self.thresholds:
                self.thresholds[context]['fraud'] = min(0.95, self.thresholds[context]['fraud'] + adjustment)
                self.thresholds[context]['warning'] = min(0.90, self.thresholds[context]['warning'] + adjustment)
            print(f"[CALIBRATION] ⬆️ Thresholds aumentados +{adjustment:.2f} (reducir falsos positivos)")
            adjustment_made = True
        
        # Si muchos falsos negativos (recall bajo) → REDUCIR thresholds
        elif fn > tp * 0.2 and recall < 0.8:  # Más de 20% FN y recall < 80%
            adjustment = min(0.05, self.max_threshold_adjustment)
            for context in self.thresholds:
                self.thresholds[context]['fraud'] = max(0.20, self.thresholds[context]['fraud'] - adjustment)
                self.thresholds[context]['warning'] = max(0.15, self.thresholds[context]['warning'] - adjustment)
            print(f"[CALIBRATION] ⬇️ Thresholds reducidos -{adjustment:.2f} (capturar más fraudes)")
            adjustment_made = True
        
        if adjustment_made:
            self.calibration_stats['calibration_count'] += 1
            self.calibration_stats['last_calibration'] = time.time()
            print(f"[CALIBRATION] Nuevos thresholds:")
            for ctx, vals in self.thresholds.items():
                print(f"[CALIBRATION]   {ctx}: fraud={vals['fraud']:.2f}, warning={vals['warning']:.2f}")
        else:
            print(f"[CALIBRATION] ✅ Thresholds óptimos, no se requiere ajuste")
    
    def get_stats(self):
        """Obtener estadísticas de calibración"""
        return {
            'thresholds': self.thresholds,
            'calibration_stats': self.calibration_stats,
            'performance_log_size': len(self.performance_log),
            'auto_calibration_enabled': self.auto_calibration_enabled
        }
    
    def reset_stats(self):
        """Resetear estadísticas de calibración"""
        self.performance_log.clear()
        self.calibration_stats = {
            'total_predictions': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'last_calibration': None,
            'calibration_count': 0
        }
        print("[INFO] Estadísticas de calibración reseteadas")

class AcousticAnalyzer:
    """
    Analizador de características acústicas del audio
    Extrae features prosódicas y paralingüísticas para detección de vishing
    MEJORA 7: Features Acústicas Básicas
    """
    
    def __init__(self):
        # Thresholds para detección de patrones sospechosos
        self.thresholds = {
            'min_silence_ratio': 0.10,      # Mínimo % de pausas esperado
            'max_silence_ratio': 0.40,      # Máximo % de pausas normal
            'min_energy_std_ratio': 0.50,   # Mínima variación de energía
            'max_zero_crossing': 0.15,      # Máximo ZCR para habla normal
            'min_speaking_rate': 2.0,       # Mínimas palabras/segundo
            'max_speaking_rate': 4.5,       # Máximas palabras/segundo
        }
        
        # Pesos de features para scoring
        self.feature_weights = {
            'scripted_speech': 0.35,        # Habla leída/robótica
            'excessive_speed': 0.25,        # Velocidad anormal
            'unnatural_pauses': 0.20,       # Pausas sospechosas
            'energy_anomaly': 0.20          # Energía anómala
        }
        
        print("[OK] AcousticAnalyzer inicializado con 6 features acústicas")
    
    def analyze_audio(self, audio_data, text=None, sample_rate=16000):
        """
        Analizar características acústicas del audio
        
        Args:
            audio_data: AudioData object de speech_recognition
            text (str): Transcripción del audio (para calcular velocidad)
            sample_rate (int): Frecuencia de muestreo
        
        Returns:
            dict: Resultado del análisis con score, features y flags
        """
        import numpy as np
        
        try:
            # Convertir audio a array numpy
            audio_array = np.frombuffer(
                audio_data.get_wav_data(), 
                dtype=np.int16
            ).astype(np.float32)
            
            # Normalizar a [-1, 1]
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            features = {}
            flags = []
            
            # ===== FEATURE 1: Duración y velocidad de habla =====
            duration = len(audio_array) / sample_rate
            features['duration'] = round(duration, 2)
            
            if text and duration > 0:
                word_count = len(text.split())
                speaking_rate = word_count / duration
                features['speaking_rate'] = round(speaking_rate, 2)
                
                # Detectar velocidad anormal
                if speaking_rate > self.thresholds['max_speaking_rate']:
                    flags.append('VELOCIDAD_EXCESIVA')
                elif speaking_rate < self.thresholds['min_speaking_rate']:
                    flags.append('VELOCIDAD_MUY_LENTA')
            else:
                features['speaking_rate'] = 0.0
            
            # ===== FEATURE 2: Energía promedio y variación =====
            energy = np.mean(np.abs(audio_array))
            energy_std = np.std(audio_array)
            energy_std_ratio = energy_std / energy if energy > 0 else 0
            
            features['energy'] = round(float(energy), 4)
            features['energy_std'] = round(float(energy_std), 4)
            features['energy_std_ratio'] = round(energy_std_ratio, 3)
            
            # Detectar energía muy uniforme (habla robótica/script)
            if energy_std_ratio < self.thresholds['min_energy_std_ratio']:
                flags.append('HABLA_ROBOTICA')
            
            # ===== FEATURE 3: Tasa de cruce por cero (ZCR) =====
            # Indica fricción, urgencia, estrés vocal
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            zcr = zero_crossings / len(audio_array)
            features['zero_crossing_rate'] = round(float(zcr), 4)
            
            if zcr > self.thresholds['max_zero_crossing']:
                flags.append('FRICCION_VOCAL_ALTA')
            
            # ===== FEATURE 4: Ratio de silencios/pausas =====
            # Detectar si hay pausas naturales o habla continua (script)
            threshold = np.max(np.abs(audio_array)) * 0.1
            silence_frames = np.sum(np.abs(audio_array) < threshold)
            silence_ratio = silence_frames / len(audio_array)
            features['silence_ratio'] = round(float(silence_ratio), 3)
            
            # Detectar pausas anormales
            if silence_ratio < self.thresholds['min_silence_ratio']:
                flags.append('PAUSAS_MINIMAS')  # Habla continua sin respirar
            elif silence_ratio > self.thresholds['max_silence_ratio']:
                flags.append('PAUSAS_EXCESIVAS')  # Muchas pausas largas
            
            # ===== FEATURE 5: Picos de energía (variabilidad) =====
            # Detectar si hay énfasis natural o monotonía
            energy_peaks = np.sum(np.abs(audio_array) > 0.7)
            peak_ratio = energy_peaks / len(audio_array)
            features['peak_ratio'] = round(float(peak_ratio), 4)
            
            if peak_ratio < 0.01:  # Muy pocos picos
                flags.append('VOZ_MONOTONA')
            
            # ===== FEATURE 6: Segmentos de habla continua =====
            # Contar segmentos donde hay habla sin pausas
            speech_threshold = np.max(np.abs(audio_array)) * 0.2
            is_speech = np.abs(audio_array) > speech_threshold
            
            # Contar transiciones silencio→habla
            speech_segments = np.sum(np.diff(is_speech.astype(int)) == 1)
            features['speech_segments'] = int(speech_segments)
            
            if speech_segments < 2 and duration > 2.0:
                flags.append('SEGMENTO_UNICO')  # Habla sin pausas naturales
            
            # ===== SCORING: Calcular score acústico =====
            score_components = {}
            
            # 1. Script detection (35%): energía uniforme + pausas mínimas
            scripted_score = 0.0
            if 'HABLA_ROBOTICA' in flags:
                scripted_score += 0.5
            if 'PAUSAS_MINIMAS' in flags:
                scripted_score += 0.3
            if 'VOZ_MONOTONA' in flags:
                scripted_score += 0.2
            score_components['scripted_speech'] = min(1.0, scripted_score)
            
            # 2. Excessive speed (25%): velocidad anormal
            speed_score = 0.0
            if features['speaking_rate'] > 0:
                if features['speaking_rate'] > self.thresholds['max_speaking_rate']:
                    # Cuanto más rápido, más sospechoso
                    speed_excess = (features['speaking_rate'] - self.thresholds['max_speaking_rate']) / 2.0
                    speed_score = min(1.0, speed_excess)
            score_components['excessive_speed'] = speed_score
            
            # 3. Unnatural pauses (20%): pausas sospechosas
            pause_score = 0.0
            if silence_ratio < self.thresholds['min_silence_ratio']:
                # Muy pocas pausas = habla apresurada
                pause_score = 1.0 - (silence_ratio / self.thresholds['min_silence_ratio'])
            elif silence_ratio > self.thresholds['max_silence_ratio']:
                # Muchas pausas = dubitación
                pause_score = 0.5
            score_components['unnatural_pauses'] = min(1.0, pause_score)
            
            # 4. Energy anomaly (20%): energía anómala
            energy_score = 0.0
            if energy_std_ratio < self.thresholds['min_energy_std_ratio']:
                # Baja variación = robótico
                energy_score = 1.0 - (energy_std_ratio / self.thresholds['min_energy_std_ratio'])
            score_components['energy_anomaly'] = min(1.0, energy_score)
            
            # Calcular score final ponderado
            total_score = sum(
                score_components[component] * self.feature_weights[component]
                for component in score_components
            )
            
            # Clasificar nivel de riesgo
            if total_score >= 0.70:
                risk_level = 'ALTO'
            elif total_score >= 0.50:
                risk_level = 'MEDIO'
            elif total_score >= 0.30:
                risk_level = 'BAJO'
            else:
                risk_level = 'NORMAL'
            
            return {
                'score': round(total_score, 3),
                'percentage': round(total_score * 100, 1),
                'risk_level': risk_level,
                'features': features,
                'flags': flags,
                'flag_count': len(flags),
                'score_components': score_components,
                'analysis_success': True
            }
            
        except Exception as e:
            print(f"[ERROR] Error en análisis acústico: {str(e)}")
            return {
                'score': 0.0,
                'percentage': 0.0,
                'risk_level': 'NORMAL',
                'features': {},
                'flags': [],
                'flag_count': 0,
                'score_components': {},
                'analysis_success': False,
                'error': str(e)
            }
    
    def get_feature_explanation(self, feature_name):
        """Obtener explicación de una feature acústica"""
        explanations = {
            'speaking_rate': 'Velocidad de habla en palabras por segundo',
            'energy': 'Energía promedio del audio (volumen)',
            'energy_std': 'Desviación estándar de la energía (variabilidad)',
            'zero_crossing_rate': 'Tasa de cruce por cero (fricción vocal)',
            'silence_ratio': 'Ratio de silencios/pausas en el audio',
            'peak_ratio': 'Ratio de picos de energía (énfasis)',
            'speech_segments': 'Número de segmentos de habla continua'
        }
        return explanations.get(feature_name, 'Feature acústica no documentada')
    
    def get_flag_explanation(self, flag_name):
        """Obtener explicación de un flag acústico"""
        explanations = {
            'VELOCIDAD_EXCESIVA': 'Habla muy rápida (>4.5 palabras/seg) - Típico de scripts o urgencia',
            'VELOCIDAD_MUY_LENTA': 'Habla muy lenta (<2.0 palabras/seg) - Posible lectura o dubitación',
            'HABLA_ROBOTICA': 'Energía muy uniforme - Típico de voz sintética o lectura de script',
            'FRICCION_VOCAL_ALTA': 'Alta tasa de cruce por cero - Indica estrés o urgencia vocal',
            'PAUSAS_MINIMAS': 'Muy pocas pausas (<10%) - Habla apresurada sin respirar',
            'PAUSAS_EXCESIVAS': 'Muchas pausas (>40%) - Posible dubitación o nerviosismo',
            'VOZ_MONOTONA': 'Pocos picos de energía - Falta de énfasis natural',
            'SEGMENTO_UNICO': 'Habla continua sin pausas naturales - Típico de lectura'
        }
        return explanations.get(flag_name, 'Flag acústico no documentado')

class ExplainableVishingDetector:
    """
    Generador de explicaciones humanas para resultados de detección de vishing
    Convierte análisis técnicos en explicaciones claras y recomendaciones accionables
    MEJORA 8: Dashboard de Explicabilidad
    """
    
    def __init__(self):
        # Mapeo de severidades por tipo de evidencia
        self.severity_mapping = {
            'KEYWORDS': 'ALTA',
            'ML_MODEL': 'ALTA',
            'SENTIMENT': 'MEDIA',
            'LINGUISTIC': 'MEDIA',
            'TEMPORAL': 'MEDIA',
            'ACOUSTIC': 'BAJA',
            'INCONGRUENCE': 'ALTA'
        }
        
        # Iconos por tipo de evidencia
        self.evidence_icons = {
            'KEYWORDS': '🔑',
            'ML_MODEL': '🤖',
            'SENTIMENT': '😰',
            'LINGUISTIC': '📝',
            'TEMPORAL': '⏱️',
            'ACOUSTIC': '🎤',
            'INCONGRUENCE': '⚠️'
        }
        
        print("[OK] ExplainableVishingDetector inicializado para generar explicaciones")
    
    def generate_explanation(self, vishing_result, fraud_analysis=None, sentiment_result=None, 
                            linguistic_result=None, temporal_result=None, acoustic_result=None,
                            incongruence_result=None, adaptive_result=None):
        """
        Generar explicación completa y humana del resultado de detección
        
        Args:
            vishing_result (dict): Resultado del VishingScorer
            fraud_analysis (dict): Análisis de keywords
            sentiment_result (dict): Análisis de sentimiento
            linguistic_result (dict): Análisis lingüístico
            temporal_result (dict): Análisis temporal
            acoustic_result (dict): Análisis acústico
            incongruence_result (dict): Análisis de incongruencias
            adaptive_result (dict): Clasificación adaptativa
        
        Returns:
            dict: Explicación estructurada con veredicto, evidencia y recomendaciones
        """
        
        # Determinar veredicto
        if adaptive_result:
            classification = adaptive_result.get('classification', 'DESCONOCIDO')
            risk_level = adaptive_result.get('risk_level', vishing_result['risk_level'])
            context = adaptive_result.get('security_context', 'medium_security')
        else:
            classification = 'FRAUDE' if vishing_result['is_vishing'] else 'LEGÍTIMO'
            risk_level = vishing_result['risk_level']
            context = 'medium_security'
        
        explanation = {
            'verdict': classification,
            'confidence': f"{vishing_result['percentage']}%",
            'risk_level': risk_level,
            'security_context': context,
            'evidence': [],
            'breakdown': {},
            'recommendations': [],
            'summary': ''
        }
        
        # ===== RECOLECTAR EVIDENCIA =====
        
        # 1. Keywords (25% del score)
        if fraud_analysis and fraud_analysis.get('keyword_analysis'):
            kw = fraud_analysis['keyword_analysis']
            if kw.get('categories') and kw['category_count'] > 0:
                categories_str = ', '.join([
                    f"{cat.capitalize()} ({data['count']})" 
                    for cat, data in list(kw['categories'].items())[:3]
                ])
                
                explanation['evidence'].append({
                    'type': 'KEYWORDS',
                    'icon': self.evidence_icons['KEYWORDS'],
                    'severity': self.severity_mapping['KEYWORDS'],
                    'score_contribution': vishing_result['breakdown']['keywords']['contribution'] * 100,
                    'detail': f"Detectadas {kw['category_count']} categorías sospechosas: {categories_str}",
                    'count': kw['total_keywords']
                })
                
                explanation['breakdown']['keywords'] = {
                    'categories': kw['category_count'],
                    'total_keywords': kw['total_keywords'],
                    'score': vishing_result['breakdown']['keywords']['value'] * 100
                }
        
        # 2. Modelo ML (20% del score)
        if fraud_analysis and fraud_analysis.get('probability'):
            ml_prob = fraud_analysis['probability']
            if ml_prob > 30:  # Solo mostrar si es significativo
                explanation['evidence'].append({
                    'type': 'ML_MODEL',
                    'icon': self.evidence_icons['ML_MODEL'],
                    'severity': self.severity_mapping['ML_MODEL'],
                    'score_contribution': vishing_result['breakdown']['ml_model']['contribution'] * 100,
                    'detail': f"Modelo ML detecta {ml_prob:.1f}% de probabilidad de fraude",
                    'count': 1
                })
                
                explanation['breakdown']['ml_model'] = {
                    'probability': ml_prob,
                    'score': vishing_result['breakdown']['ml_model']['value'] * 100
                }
        
        # 3. Sentimiento (15% del score)
        if sentiment_result:
            sentiment_text = sentiment_result.get('sentiment', 'neutral').upper()
            sentiment_scores = sentiment_result.get('scores', {})
            
            # Detectar sentimiento sospechoso
            is_negative = sentiment_result.get('sentiment') == 'negative'
            has_fear = sentiment_scores.get('fear', 0) > 0.3
            
            if is_negative or has_fear:
                detail_parts = [f"Sentimiento: {sentiment_text}"]
                if has_fear:
                    detail_parts.append(f"Miedo detectado ({sentiment_scores['fear']*100:.0f}%)")
                
                explanation['evidence'].append({
                    'type': 'SENTIMENT',
                    'icon': self.evidence_icons['SENTIMENT'],
                    'severity': self.severity_mapping['SENTIMENT'],
                    'score_contribution': vishing_result['breakdown']['sentiment']['contribution'] * 100,
                    'detail': ' - '.join(detail_parts),
                    'count': 1
                })
                
                explanation['breakdown']['sentiment'] = {
                    'sentiment': sentiment_text,
                    'fear': sentiment_scores.get('fear', 0) * 100,
                    'score': vishing_result['breakdown']['sentiment']['value'] * 100
                }
        
        # 4. Análisis Lingüístico (20% del score)
        if linguistic_result and linguistic_result.get('flags'):
            flags_count = linguistic_result['pattern_count']
            flags_str = ', '.join(linguistic_result['flags'][:3])
            if flags_count > 3:
                flags_str += f" (+{flags_count - 3} más)"
            
            explanation['evidence'].append({
                'type': 'LINGUISTIC',
                'icon': self.evidence_icons['LINGUISTIC'],
                'severity': self.severity_mapping['LINGUISTIC'],
                'score_contribution': vishing_result['breakdown']['linguistic']['contribution'] * 100,
                'detail': f"{flags_count} patrones lingüísticos: {flags_str}",
                'count': flags_count
            })
            
            explanation['breakdown']['linguistic'] = {
                'patterns': flags_count,
                'flags': linguistic_result['flags'][:5],
                'score': vishing_result['breakdown']['linguistic']['value'] * 100
            }
        
        # 5. Análisis Temporal (10% del score)
        if temporal_result and temporal_result.get('flags'):
            patterns_count = temporal_result['pattern_count']
            patterns_str = ', '.join(temporal_result['flags'][:3])
            if patterns_count > 3:
                patterns_str += f" (+{patterns_count - 3} más)"
            
            explanation['evidence'].append({
                'type': 'TEMPORAL',
                'icon': self.evidence_icons['TEMPORAL'],
                'severity': self.severity_mapping['TEMPORAL'],
                'score_contribution': vishing_result['breakdown']['temporal']['contribution'] * 100,
                'detail': f"{patterns_count} anomalías conversacionales: {patterns_str}",
                'count': patterns_count
            })
            
            explanation['breakdown']['temporal'] = {
                'patterns': patterns_count,
                'flags': temporal_result['flags'][:5],
                'turn_count': temporal_result.get('turn_count', 0),
                'score': vishing_result['breakdown']['temporal']['value'] * 100
            }
        
        # 6. Análisis Acústico (10% del score)
        if acoustic_result and acoustic_result.get('flags') and acoustic_result.get('analysis_success'):
            flags_count = acoustic_result['flag_count']
            flags_str = ', '.join(acoustic_result['flags'][:3])
            if flags_count > 3:
                flags_str += f" (+{flags_count - 3} más)"
            
            explanation['evidence'].append({
                'type': 'ACOUSTIC',
                'icon': self.evidence_icons['ACOUSTIC'],
                'severity': self.severity_mapping['ACOUSTIC'],
                'score_contribution': vishing_result['breakdown']['acoustic']['contribution'] * 100,
                'detail': f"{flags_count} características acústicas sospechosas: {flags_str}",
                'count': flags_count
            })
            
            explanation['breakdown']['acoustic'] = {
                'flags': flags_count,
                'features': acoustic_result['flags'][:5],
                'score': vishing_result['breakdown']['acoustic']['value'] * 100
            }
        
        # 7. Incongruencias (adicional al score)
        if incongruence_result and incongruence_result.get('incongruence_count', 0) > 0:
            incong_count = incongruence_result['incongruence_count']
            incong_str = ', '.join(incongruence_result['flags'][:3])
            if incong_count > 3:
                incong_str += f" (+{incong_count - 3} más)"
            
            explanation['evidence'].append({
                'type': 'INCONGRUENCE',
                'icon': self.evidence_icons['INCONGRUENCE'],
                'severity': self.severity_mapping['INCONGRUENCE'],
                'score_contribution': incongruence_result['total_score'] * 100,
                'detail': f"{incong_count} contradicciones detectadas: {incong_str}",
                'count': incong_count
            })
            
            explanation['breakdown']['incongruence'] = {
                'count': incong_count,
                'flags': incongruence_result['flags'][:5],
                'score': incongruence_result['total_score'] * 100
            }
        
        # ===== GENERAR RECOMENDACIONES =====
        
        score = vishing_result['score']
        
        if score >= 0.75 or classification == 'FRAUDE':
            # CRÍTICO
            explanation['recommendations'] = [
                {
                    'priority': 'CRÍTICA',
                    'icon': '🚨',
                    'action': 'TERMINAR LA LLAMADA INMEDIATAMENTE',
                    'reason': 'Múltiples indicadores de vishing detectados'
                },
                {
                    'priority': 'CRÍTICA',
                    'icon': '❌',
                    'action': 'NO proporcionar NINGÚN dato personal o financiero',
                    'reason': 'Alto riesgo de robo de identidad'
                },
                {
                    'priority': 'ALTA',
                    'icon': '📞',
                    'action': 'Contactar DIRECTAMENTE a su banco usando el número oficial',
                    'reason': 'Verificar legitimidad de la comunicación'
                },
                {
                    'priority': 'ALTA',
                    'icon': '📝',
                    'action': 'Reportar el incidente a las autoridades',
                    'reason': 'Contribuir a prevención de fraude'
                },
                {
                    'priority': 'MEDIA',
                    'icon': '🔒',
                    'action': 'Monitorear sus cuentas bancarias',
                    'reason': 'Detectar actividad no autorizada'
                }
            ]
            
            explanation['summary'] = (
                f"🚨 ALERTA CRÍTICA: Detectado intento de VISHING con {score*100:.0f}% de confianza. "
                f"Se identificaron {len(explanation['evidence'])} tipos de evidencia sospechosa. "
                f"Acción recomendada: TERMINAR LLAMADA INMEDIATAMENTE."
            )
        
        elif score >= 0.60 or classification == 'SOSPECHOSO':
            # ALTO
            explanation['recommendations'] = [
                {
                    'priority': 'ALTA',
                    'icon': '⚠️',
                    'action': 'Proceder con EXTREMA cautela',
                    'reason': 'Patrones sospechosos detectados'
                },
                {
                    'priority': 'ALTA',
                    'icon': '🔍',
                    'action': 'Verificar identidad del llamante por canales oficiales',
                    'reason': 'Confirmar legitimidad antes de continuar'
                },
                {
                    'priority': 'ALTA',
                    'icon': '❌',
                    'action': 'NO proporcionar datos sensibles (CVV, PIN, contraseñas)',
                    'reason': 'Riesgo elevado de fraude'
                },
                {
                    'priority': 'MEDIA',
                    'icon': '📞',
                    'action': 'Colgar y llamar al número oficial de la entidad',
                    'reason': 'Verificar autenticidad de la solicitud'
                },
                {
                    'priority': 'MEDIA',
                    'icon': '👥',
                    'action': 'Consultar con un familiar o persona de confianza',
                    'reason': 'Segunda opinión puede prevenir fraude'
                }
            ]
            
            explanation['summary'] = (
                f"⚠️ ALERTA ALTA: Posible intento de vishing ({score*100:.0f}% confianza). "
                f"Detectadas {len(explanation['evidence'])} señales de alerta. "
                f"Recomendación: Verificar identidad antes de continuar."
            )
        
        elif score >= 0.45 or classification == 'MONITOREAR':
            # MEDIO
            explanation['recommendations'] = [
                {
                    'priority': 'MEDIA',
                    'icon': '🔍',
                    'action': 'Mantenerse alerta y escéptico',
                    'reason': 'Algunas características sospechosas detectadas'
                },
                {
                    'priority': 'MEDIA',
                    'icon': '❓',
                    'action': 'Hacer preguntas para verificar identidad',
                    'reason': 'Legitimar llamantes podrán responder correctamente'
                },
                {
                    'priority': 'MEDIA',
                    'icon': '⏸️',
                    'action': 'Posponer decisiones importantes',
                    'reason': 'Evitar presión para tomar decisiones apresuradas'
                },
                {
                    'priority': 'BAJA',
                    'icon': '📝',
                    'action': 'Tomar nota de detalles de la llamada',
                    'reason': 'Útil si necesita reportar posteriormente'
                },
                {
                    'priority': 'BAJA',
                    'icon': '🤔',
                    'action': 'Confiar en su instinto',
                    'reason': 'Si algo parece sospechoso, probablemente lo es'
                }
            ]
            
            explanation['summary'] = (
                f"⚠️ PRECAUCIÓN: Conversación con riesgo MEDIO ({score*100:.0f}% confianza). "
                f"Identificados {len(explanation['evidence'])} indicadores. "
                f"Manténgase alerta y verifique identidad."
            )
        
        else:
            # BAJO/NORMAL
            explanation['recommendations'] = [
                {
                    'priority': 'BAJA',
                    'icon': '✅',
                    'action': 'La conversación parece legítima',
                    'reason': 'Pocos o ningún indicador de vishing detectado'
                },
                {
                    'priority': 'BAJA',
                    'icon': '🔍',
                    'action': 'Mantenga precauciones estándar',
                    'reason': 'Nunca comparta datos sensibles sin verificar'
                },
                {
                    'priority': 'BAJA',
                    'icon': '🛡️',
                    'action': 'Continúe usando buenas prácticas de seguridad',
                    'reason': 'Prevención es la mejor defensa'
                }
            ]
            
            explanation['summary'] = (
                f"✅ NORMAL: Conversación con riesgo BAJO ({score*100:.0f}% confianza). "
                f"No se detectaron señales significativas de vishing. "
                f"Mantenga precauciones estándar de seguridad."
            )
        
        return explanation
    
    def format_explanation_text(self, explanation):
        """Formatear explicación como texto legible"""
        lines = []
        
        lines.append("=" * 70)
        lines.append(f"VEREDICTO: {explanation['verdict']} (Confianza: {explanation['confidence']})")
        lines.append(f"NIVEL DE RIESGO: {explanation['risk_level']}")
        lines.append("=" * 70)
        
        lines.append(f"\n{explanation['summary']}\n")
        
        if explanation['evidence']:
            lines.append("EVIDENCIA DETECTADA:")
            for i, evidence in enumerate(explanation['evidence'], 1):
                lines.append(f"{i}. {evidence['icon']} [{evidence['type']}] "
                           f"(Severidad: {evidence['severity']}, "
                           f"Contribución: {evidence['score_contribution']:.1f}%)")
                lines.append(f"   {evidence['detail']}")
        
        lines.append("\nRECOMENDACIONES:")
        for i, rec in enumerate(explanation['recommendations'], 1):
            lines.append(f"{i}. {rec['icon']} [{rec['priority']}] {rec['action']}")
            lines.append(f"   → {rec['reason']}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)

class VishingScorer:
    """Sistema de puntuación multinivel para detección de vishing"""
    def __init__(self):
        # Pesos balanceados por tipo de análisis
        self.weights = {
            'keywords': 0.25,      # Palabras clave específicas
            'ml_model': 0.20,      # Modelo ML existente
            'sentiment': 0.15,     # Análisis de sentimiento
            'linguistic': 0.20,    # Patrones lingüísticos (futuro)
            'temporal': 0.10,      # Análisis temporal (futuro)
            'acoustic': 0.10       # Features de audio (futuro)
        }
        
        # Thresholds multinivel
        self.thresholds = {
            'critical': 0.75,   # 75% - Fraude crítico
            'high': 0.50,       # 50% - Riesgo alto
            'medium': 0.35,     # 35% - Riesgo medio
            'low': 0.20         # 20% - Riesgo bajo
        }
        
        print("[OK] VishingScorer inicializado con sistema multinivel")
    
    def compute_vishing_score(self, features):
        """
        Calcular score de vishing combinando múltiples features
        
        Args:
            features (dict): Diccionario con scores de diferentes análisis
                - 'keywords': float (0-1) - Score de keywords
                - 'ml_model': float (0-1) - Probabilidad del modelo ML
                - 'sentiment': float (0-1) - Score de sentimiento negativo
                - 'linguistic': float (0-1) - Patrones lingüísticos
                - 'temporal': float (0-1) - Análisis temporal
                - 'acoustic': float (0-1) - Features acústicas
        
        Returns:
            dict: {
                'score': float (0-1) - Score final normalizado
                'confidence': float (0-1) - Confianza basada en features activas
                'breakdown': dict - Desglose por tipo de feature
                'risk_level': str - Nivel de riesgo
                'is_vishing': bool - Clasificación binaria
            }
        """
        score = 0
        confidence = 0
        breakdown = {}
        
        # Calcular score ponderado solo con features disponibles
        for feature_name, weight in self.weights.items():
            if feature_name in features and features[feature_name] is not None:
                feature_value = features[feature_name]
                # Asegurar que el valor está en rango [0, 1]
                feature_value = max(0.0, min(1.0, float(feature_value)))
                
                weighted_score = feature_value * weight
                score += weighted_score
                confidence += weight
                
                breakdown[feature_name] = {
                    'value': round(feature_value, 3),
                    'weight': weight,
                    'contribution': round(weighted_score, 3)
                }
        
        # Normalizar por confidence (suma de pesos de features activas)
        final_score = score / confidence if confidence > 0 else 0
        
        # Clasificar nivel de riesgo
        risk_level = self._classify_risk(final_score)
        
        # Decisión binaria
        is_vishing = final_score >= self.thresholds['medium']
        
        return {
            'score': round(final_score, 3),
            'confidence': round(confidence, 3),
            'breakdown': breakdown,
            'risk_level': risk_level,
            'is_vishing': is_vishing,
            'percentage': round(final_score * 100, 1)
        }
    
    def _classify_risk(self, score):
        """Clasificar nivel de riesgo basado en thresholds"""
        if score >= self.thresholds['critical']:
            return 'CRÍTICO'
        elif score >= self.thresholds['high']:
            return 'ALTO'
        elif score >= self.thresholds['medium']:
            return 'MEDIO'
        elif score >= self.thresholds['low']:
            return 'BAJO'
        else:
            return 'NORMAL'
    
    def get_explanation(self, vishing_result):
        """Generar explicación humana del resultado"""
        explanation = []
        
        # Explicar score principal
        explanation.append(f"Score de vishing: {vishing_result['percentage']}%")
        explanation.append(f"Nivel de riesgo: {vishing_result['risk_level']}")
        explanation.append(f"Confianza del análisis: {vishing_result['confidence']*100:.0f}%")
        
        # Desglosar contribuciones
        if vishing_result['breakdown']:
            explanation.append("\nContribuciones por análisis:")
            for feature, data in sorted(vishing_result['breakdown'].items(), 
                                       key=lambda x: x[1]['contribution'], 
                                       reverse=True):
                if data['contribution'] > 0:
                    explanation.append(
                        f"  • {feature}: {data['value']*100:.1f}% "
                        f"(peso: {data['weight']*100:.0f}%, "
                        f"contribución: {data['contribution']*100:.1f}%)"
                    )
        
        return '\n'.join(explanation)

class LinguisticAnalyzer:
    """
    Analizador de patrones lingüísticos para detección de vishing
    Detecta estructuras del lenguaje más allá de keywords específicas
    MEJORA 3: Análisis Lingüístico con NLP
    """
    
    def __init__(self):
        import re
        
        # Patrones regex para detectar estructuras lingüísticas
        self.patterns = {
            'imperativos': r'\b(debe|tiene que|necesita|confirme|verifique|dígame|proporcione|envíe|haga|realice|pulse|marque)\b',
            'preguntas_datos': r'(cuál es su|me puede dar|necesito que me diga|dígame su|confirme su|indique su|proporcione su)',
            'tiempo_limitado': r'(\d+\s*(horas?|minutos?|días?|segundos?)|antes de|hasta el|plazo de|vence|expira|caduca)',
            'negaciones_riesgo': r'(sin riesgo|garantizado|seguro|100%|totalmente seguro|sin problema|confiable|certificado)',
            'autoridad': r'\b(departamento|ministerio|policía|autoridad|oficial|juzgado|tribunal|gobierno|agencia)\b',
            'amenazas_legales': r'(demanda|multa|sanción|proceso legal|orden judicial|cargo criminal|delito|consecuencias legales)',
            'ofertas_sospechosas': r'(ha ganado|premio|sorteo|lotería|beneficiario|seleccionado|afortunado|gratis)',
            'solicitud_accion': r'(haga clic|descargue|instale|abra el enlace|visite|acceda a|ingrese a)'
        }
        
        # Pesos por tipo de patrón (0-1, qué tan sospechoso es)
        self.pattern_weights = {
            'imperativos': 0.7,        # Comandos directos = presión
            'preguntas_datos': 0.9,    # Solicitar datos = muy sospechoso
            'tiempo_limitado': 0.8,    # Presión temporal = táctica común
            'negaciones_riesgo': 0.6,  # Tranquilizar = manipulación
            'autoridad': 0.8,          # Falsa autoridad = suplantación
            'amenazas_legales': 0.85,  # Amenazas = intimidación
            'ofertas_sospechosas': 0.7, # Ofertas falsas = gancho
            'solicitud_accion': 0.75   # Solicitar acción = phishing
        }
        
        print("[OK] LinguisticAnalyzer inicializado con 8 patrones lingüísticos")
    
    def analyze(self, text):
        """
        Analizar patrones lingüísticos en el texto
        
        Args:
            text (str): Texto a analizar
        
        Returns:
            dict: {
                'scores': dict - Scores individuales por patrón (0-1)
                'total_score': float - Score total normalizado (0-1)
                'flags': list - Lista de patrones detectados con score > 0.5
                'matches': dict - Coincidencias encontradas por patrón
                'risk_level': str - Nivel de riesgo lingüístico
            }
        """
        import re
        
        if not text or not text.strip():
            return self._empty_result()
        
        text_lower = text.lower()
        scores = {}
        matches = {}
        
        # 1. Detectar imperativos (comandos directos)
        imperative_matches = re.findall(self.patterns['imperativos'], text_lower)
        imperative_count = len(imperative_matches)
        scores['imperative'] = min(1.0, (imperative_count / 3) * self.pattern_weights['imperativos'])
        matches['imperative'] = imperative_matches[:5]  # Primeros 5
        
        # 2. Solicitudes de datos personales
        data_request_matches = re.findall(self.patterns['preguntas_datos'], text_lower)
        data_requests = len(data_request_matches)
        scores['data_request'] = min(1.0, (data_requests / 2) * self.pattern_weights['preguntas_datos'])
        matches['data_request'] = data_request_matches[:5]
        
        # 3. Presión temporal
        time_pressure_match = re.search(self.patterns['tiempo_limitado'], text_lower)
        time_pressure = bool(time_pressure_match)
        scores['time_pressure'] = self.pattern_weights['tiempo_limitado'] if time_pressure else 0.0
        matches['time_pressure'] = [time_pressure_match.group()] if time_pressure_match else []
        
        # 4. Negaciones de riesgo (tranquilización sospechosa)
        risk_negation_matches = re.findall(self.patterns['negaciones_riesgo'], text_lower)
        risk_negations = len(risk_negation_matches)
        scores['risk_negation'] = min(1.0, (risk_negations / 2) * self.pattern_weights['negaciones_riesgo'])
        matches['risk_negation'] = risk_negation_matches[:5]
        
        # 5. Apelación a falsa autoridad
        authority_matches = re.findall(self.patterns['autoridad'], text_lower)
        authority_claims = len(authority_matches)
        scores['authority'] = min(1.0, (authority_claims / 2) * self.pattern_weights['autoridad'])
        matches['authority'] = authority_matches[:5]
        
        # 6. Amenazas legales
        legal_threat_matches = re.findall(self.patterns['amenazas_legales'], text_lower)
        legal_threats = len(legal_threat_matches)
        scores['legal_threats'] = min(1.0, (legal_threats / 2) * self.pattern_weights['amenazas_legales'])
        matches['legal_threats'] = legal_threat_matches[:5]
        
        # 7. Ofertas sospechosas (premios, loterías)
        offer_matches = re.findall(self.patterns['ofertas_sospechosas'], text_lower)
        suspicious_offers = len(offer_matches)
        scores['suspicious_offers'] = min(1.0, (suspicious_offers / 2) * self.pattern_weights['ofertas_sospechosas'])
        matches['suspicious_offers'] = offer_matches[:5]
        
        # 8. Solicitud de acción inmediata
        action_matches = re.findall(self.patterns['solicitud_accion'], text_lower)
        action_requests = len(action_matches)
        scores['action_request'] = min(1.0, (action_requests / 2) * self.pattern_weights['solicitud_accion'])
        matches['action_request'] = action_matches[:5]
        
        # 9. Longitud anormal (scripts de vishing suelen ser largos)
        word_count = len(text.split())
        if word_count > 150:
            scores['length_anomaly'] = 1.0
        elif word_count > 100:
            scores['length_anomaly'] = 0.6
        elif word_count > 70:
            scores['length_anomaly'] = 0.3
        else:
            scores['length_anomaly'] = 0.0
        matches['length_anomaly'] = [f"{word_count} palabras"]
        
        # 10. Exceso de preguntas (interrogatorio)
        question_count = text.count('?') + text.count('¿')
        scores['questioning'] = min(1.0, question_count / 4)
        matches['questioning'] = [f"{question_count} preguntas"]
        
        # Calcular score total (promedio de todos los scores)
        total_score = sum(scores.values()) / len(scores)
        
        # Identificar flags (patrones con score significativo)
        flags = [k for k, v in scores.items() if v > 0.3]
        
        # Determinar nivel de riesgo lingüístico
        if total_score >= 0.7:
            risk_level = 'CRÍTICO'
        elif total_score >= 0.5:
            risk_level = 'ALTO'
        elif total_score >= 0.3:
            risk_level = 'MEDIO'
        elif total_score >= 0.15:
            risk_level = 'BAJO'
        else:
            risk_level = 'NORMAL'
        
        return {
            'scores': scores,
            'total_score': total_score,
            'flags': flags,
            'matches': matches,
            'risk_level': risk_level,
            'pattern_count': len(flags),
            'word_count': word_count,
            'question_count': question_count
        }
    
    def _empty_result(self):
        """Resultado vacío cuando no hay texto"""
        return {
            'scores': {},
            'total_score': 0.0,
            'flags': [],
            'matches': {},
            'risk_level': 'NORMAL',
            'pattern_count': 0,
            'word_count': 0,
            'question_count': 0
        }
    
    def get_pattern_info(self):
        """Obtener información de todos los patrones disponibles"""
        return {
            pattern_name: {
                'regex': pattern,
                'weight': self.pattern_weights.get(pattern_name, 0.5),
                'description': self._get_pattern_description(pattern_name)
            }
            for pattern_name, pattern in self.patterns.items()
        }
    
    def _get_pattern_description(self, pattern_name):
        """Descripción de cada patrón"""
        descriptions = {
            'imperativos': 'Comandos directos que presionan al usuario',
            'preguntas_datos': 'Solicitudes explícitas de datos personales',
            'tiempo_limitado': 'Referencias temporales que crean urgencia',
            'negaciones_riesgo': 'Intentos de tranquilizar sospechosamente',
            'autoridad': 'Apelación a autoridades o instituciones',
            'amenazas_legales': 'Amenazas con consecuencias legales',
            'ofertas_sospechosas': 'Ofertas de premios o beneficios no solicitados',
            'solicitud_accion': 'Solicitudes de realizar acciones inmediatas'
        }
        return descriptions.get(pattern_name, 'Patrón lingüístico')

class ConversationAnalyzer:
    """
    Analizador de patrones temporales en la conversación
    Detecta comportamientos sospechosos a lo largo del flujo de diálogo
    MEJORA 4: Análisis Temporal de Conversación
    """
    
    def __init__(self, window_size=10):
        """
        Inicializar analizador de conversación
        
        Args:
            window_size (int): Número de turnos a mantener en el historial
        """
        self.history = deque(maxlen=window_size)
        self.window_size = window_size
        
        # Palabras clave para diferentes análisis
        self.urgency_keywords = [
            'urgente', 'ya', 'ahora', 'inmediato', 'rápido',
            'enseguida', 'pronto', 'cuanto antes', 'de inmediato',
            'sin demora', 'ahora mismo', 'inmediatamente'
        ]
        
        self.data_request_keywords = [
            'dígame', 'confirme', 'verifique', 'proporcione',
            'indique', 'facilite', 'necesito', 'cuál es',
            'me puede dar', 'envíe', 'número', 'código'
        ]
        
        print(f"[OK] ConversationAnalyzer inicializado (ventana: {window_size} turnos)")
    
    def add_turn(self, text, speaker='system'):
        """
        Agregar un turno de conversación al historial
        
        Args:
            text (str): Texto del turno
            speaker (str): Quién habla ('system' o 'user')
        """
        import time
        
        if not text or not text.strip():
            return
        
        turn = {
            'text': text,
            'speaker': speaker,
            'timestamp': time.time(),
            'word_count': len(text.split()),
            'question_count': text.count('?') + text.count('¿'),
            'urgency_score': self._count_keywords(text, self.urgency_keywords),
            'data_requests': self._count_keywords(text, self.data_request_keywords)
        }
        
        self.history.append(turn)
    
    def _count_keywords(self, text, keywords):
        """Contar keywords en el texto (case-insensitive)"""
        text_lower = text.lower()
        return sum(1 for kw in keywords if kw in text_lower)
    
    def analyze_patterns(self):
        """
        Analizar patrones temporales en la conversación
        
        Returns:
            dict: {
                'scores': dict - Scores individuales por patrón
                'total_score': float - Score total normalizado (0-1)
                'flags': list - Patrones detectados
                'metrics': dict - Métricas adicionales
                'risk_level': str - Nivel de riesgo temporal
            }
        """
        if len(self.history) < 2:
            return self._empty_result()
        
        flags = []
        scores = {}
        metrics = {}
        
        # 1. ESCALADA DE URGENCIA
        # Detectar si las palabras de urgencia aumentan con el tiempo
        urgency_trend = [turn['urgency_score'] for turn in self.history]
        
        if len(urgency_trend) >= 3:
            first_half_avg = sum(urgency_trend[:len(urgency_trend)//2]) / max(len(urgency_trend)//2, 1)
            second_half_avg = sum(urgency_trend[len(urgency_trend)//2:]) / max(len(urgency_trend)//2, 1)
            
            # Si la urgencia aumenta significativamente
            if second_half_avg > first_half_avg + 1:
                scores['urgency_escalation'] = min(1.0, (second_half_avg - first_half_avg) / 3)
                flags.append('ESCALADA_URGENCIA')
                metrics['urgency_increase'] = second_half_avg - first_half_avg
        
        # 2. MONOPOLIZACIÓN DE LA CONVERSACIÓN
        # Turnos excesivamente largos (scripts preparados)
        word_counts = [t['word_count'] for t in self.history]
        avg_length = sum(word_counts) / len(word_counts)
        max_length = max(word_counts)
        
        if avg_length > 50:
            scores['monopolization'] = min(1.0, avg_length / 100)
            flags.append('MONOPOLIO_CONVERSACION')
            metrics['avg_turn_length'] = avg_length
        
        if max_length > 100:
            scores['long_script'] = min(1.0, max_length / 150)
            flags.append('TURNO_MUY_LARGO')
            metrics['max_turn_length'] = max_length
        
        # 3. EXCESO DE PREGUNTAS
        # Interrogatorio (solicitar mucha información)
        total_questions = sum(t['question_count'] for t in self.history)
        avg_questions = total_questions / len(self.history)
        
        if total_questions > len(self.history) * 2:  # Más de 2 preguntas por turno
            scores['excessive_questions'] = min(1.0, avg_questions / 4)
            flags.append('EXCESO_PREGUNTAS')
            metrics['total_questions'] = total_questions
        
        # 4. REPETICIÓN DE SOLICITUDES
        # El estafador repite las mismas solicitudes si el usuario no colabora
        texts = [t['text'].lower() for t in self.history]
        unique_texts = len(set(texts))
        unique_ratio = unique_texts / len(texts) if len(texts) > 0 else 1.0
        
        if unique_ratio < 0.6 and len(self.history) >= 4:  # Mucha repetición
            scores['repetition'] = 1.0 - unique_ratio
            flags.append('REPETICION_ALTA')
            metrics['unique_ratio'] = unique_ratio
        
        # 5. INSISTENCIA EN DATOS
        # Aumento de solicitudes de datos personales
        data_request_trend = [turn['data_requests'] for turn in self.history]
        total_data_requests = sum(data_request_trend)
        
        if total_data_requests > len(self.history):  # Más de 1 solicitud por turno
            scores['data_insistence'] = min(1.0, total_data_requests / (len(self.history) * 2))
            flags.append('INSISTENCIA_DATOS')
            metrics['data_requests'] = total_data_requests
        
        # 6. RITMO ACELERADO
        # Turnos muy seguidos sin dar tiempo a pensar
        if len(self.history) >= 3:
            timestamps = [t['timestamp'] for t in self.history]
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            if avg_interval < 5:  # Menos de 5 segundos entre turnos (presión)
                scores['rapid_pace'] = min(1.0, 5 / max(avg_interval, 1))
                flags.append('RITMO_ACELERADO')
                metrics['avg_interval_seconds'] = avg_interval
        
        # 7. VOLUMEN DE CONVERSACIÓN
        # Conversaciones de vishing tienden a ser más largas
        if len(self.history) >= 8:
            scores['long_conversation'] = min(1.0, len(self.history) / self.window_size)
            flags.append('CONVERSACION_LARGA')
            metrics['turn_count'] = len(self.history)
        
        # Calcular score total (promedio ponderado)
        if scores:
            # Dar más peso a los patrones más críticos
            weights = {
                'urgency_escalation': 1.2,
                'data_insistence': 1.3,
                'monopolization': 1.0,
                'excessive_questions': 1.1,
                'repetition': 0.9,
                'long_script': 1.0,
                'rapid_pace': 0.8,
                'long_conversation': 0.7
            }
            
            weighted_sum = sum(scores.get(k, 0) * weights.get(k, 1.0) for k in scores.keys())
            total_weight = sum(weights.get(k, 1.0) for k in scores.keys())
            total_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            total_score = 0.0
        
        # Determinar nivel de riesgo temporal
        if total_score >= 0.7:
            risk_level = 'CRÍTICO'
        elif total_score >= 0.5:
            risk_level = 'ALTO'
        elif total_score >= 0.3:
            risk_level = 'MEDIO'
        elif total_score >= 0.15:
            risk_level = 'BAJO'
        else:
            risk_level = 'NORMAL'
        
        return {
            'scores': scores,
            'total_score': total_score,
            'flags': flags,
            'metrics': metrics,
            'risk_level': risk_level,
            'pattern_count': len(flags),
            'turn_count': len(self.history)
        }
    
    def _empty_result(self):
        """Resultado vacío cuando no hay suficiente historial"""
        return {
            'scores': {},
            'total_score': 0.0,
            'flags': [],
            'metrics': {},
            'risk_level': 'NORMAL',
            'pattern_count': 0,
            'turn_count': len(self.history)
        }
    
    def clear_history(self):
        """Limpiar historial de conversación"""
        self.history.clear()
        print("[INFO] Historial de conversación limpiado")
    
    def get_conversation_summary(self):
        """Obtener resumen de la conversación"""
        if not self.history:
            return {
                'turn_count': 0,
                'total_words': 0,
                'total_questions': 0,
                'avg_turn_length': 0
            }
        
        return {
            'turn_count': len(self.history),
            'total_words': sum(t['word_count'] for t in self.history),
            'total_questions': sum(t['question_count'] for t in self.history),
            'avg_turn_length': sum(t['word_count'] for t in self.history) / len(self.history),
            'oldest_turn_age_seconds': self.history[0]['timestamp'] - self.history[-1]['timestamp'] if len(self.history) > 1 else 0
        }

class IncongruenceDetector:
    """
    Detector de incongruencias y señales mixtas en el texto
    Identifica contradicciones sospechosas (amabilidad + urgencia, etc.)
    MEJORA 5: Detector de Incongruencias
    """
    
    def __init__(self):
        # Palabras para detectar amabilidad
        self.politeness_words = [
            'por favor', 'disculpe', 'gracias', 'muchas gracias',
            'señor', 'señora', 'buenos días', 'buenas tardes',
            'buenas noches', 'le agradezco', 'si es tan amable',
            'con permiso', 'perdone', 'estimado', 'apreciado'
        ]
        
        # Palabras de urgencia
        self.urgency_words = [
            'urgente', 'ya', 'ahora', 'inmediato', 'rápido',
            'antes de', 'cuanto antes', 'de inmediato', 'enseguida',
            'sin demora', 'ahora mismo', 'pronto', 'inmediatamente',
            'tiempo limitado', 'último momento', 'expira'
        ]
        
        # Palabras de amenaza
        self.threat_words = [
            'bloqueado', 'suspendido', 'multa', 'demanda', 'sanción',
            'cancelado', 'desactivado', 'inhabilitado', 'proceso legal',
            'consecuencias', 'penalización', 'cargo criminal', 'delito',
            'orden judicial', 'embargo', 'congelado', 'restricción'
        ]
        
        # Palabras de solicitud de datos sensibles
        self.data_request_words = [
            'cvv', 'pin', 'clave', 'contraseña', 'password',
            'número de tarjeta', 'fecha de vencimiento', 'código de seguridad',
            'token', 'otp', 'cédula', 'dni', 'nif', 'pasaporte',
            'cuenta bancaria', 'número de cuenta', 'iban'
        ]
        
        # Palabras de tranquilización/reassurance
        self.reassurance_words = [
            'no se preocupe', 'tranquilo', 'tranquila', 'seguro',
            'sin problema', 'sin riesgo', 'garantizado', 'confiable',
            'está protegido', 'no pasa nada', 'todo está bien',
            'es normal', 'rutinario', 'estándar', 'certificado'
        ]
        
        # Palabras de beneficio/ganancia
        self.benefit_words = [
            'ha ganado', 'premio', 'lotería', 'sorteo', 'beneficio',
            'reembolso', 'devolución', 'bonificación', 'descuento',
            'oferta exclusiva', 'gratis', 'sin costo', 'regalo'
        ]
        
        print("[OK] IncongruenceDetector inicializado con 6 categorías de análisis")
    
    def analyze(self, text, sentiment_result):
        """
        Analizar incongruencias y señales mixtas en el texto
        
        Args:
            text (str): Texto a analizar
            sentiment_result (dict): Resultado del análisis de sentimiento
        
        Returns:
            dict: {
                'score': float - Score de incongruencia (0-1)
                'flags': list - Lista de incongruencias detectadas
                'details': dict - Detalles de cada incongruencia
                'risk_level': str - Nivel de riesgo
            }
        """
        if not text or not text.strip():
            return self._empty_result()
        
        flags = []
        details = {}
        scores = {}
        
        text_lower = text.lower()
        
        # Detectar niveles de cada categoría
        politeness_score = self._detect_politeness(text_lower)
        urgency_score = self._detect_urgency(text_lower)
        has_threats = self._has_category(text_lower, self.threat_words)
        has_data_request = self._has_category(text_lower, self.data_request_words)
        has_reassurance = self._has_category(text_lower, self.reassurance_words)
        has_benefits = self._has_category(text_lower, self.benefit_words)
        
        # 1. AMABILIDAD EXCESIVA + URGENCIA (Muy sospechoso)
        # Los estafadores son corteses pero presionan con urgencia
        if politeness_score > 0.6 and urgency_score > 0.6:
            scores['politeness_urgency'] = 0.85
            flags.append('AMABILIDAD_CON_URGENCIA')
            details['politeness_urgency'] = {
                'politeness': politeness_score,
                'urgency': urgency_score,
                'reason': 'Cortesía excesiva combinada con presión temporal'
            }
        
        # 2. SENTIMIENTO POSITIVO + AMENAZAS (Incongruencia emocional)
        # Tono positivo hablando de problemas graves
        if sentiment_result.get('label') == 'POS' and has_threats:
            scores['positive_threats'] = 0.9
            flags.append('SENTIMIENTO_POSITIVO_CON_AMENAZAS')
            details['positive_threats'] = {
                'sentiment': sentiment_result.get('label'),
                'compound': sentiment_result.get('compound', 0),
                'reason': 'Tono positivo al comunicar amenazas o problemas'
            }
        
        # 3. SOLICITUD DE DATOS + TRANQUILIZACIÓN (Bandera roja)
        # Pedir datos sensibles mientras tranquilizan
        if has_data_request and has_reassurance:
            scores['data_reassurance'] = 1.0  # Máxima gravedad
            flags.append('SOLICITUD_DATOS_CON_TRANQUILIZACIÓN')
            details['data_reassurance'] = {
                'reason': 'Solicita datos sensibles mientras intenta tranquilizar'
            }
        
        # 4. AMENAZAS + TRANQUILIZACIÓN (Manipulación emocional)
        # "Su cuenta está bloqueada, pero no se preocupe"
        if has_threats and has_reassurance:
            scores['threat_reassurance'] = 0.8
            flags.append('AMENAZA_CON_TRANQUILIZACIÓN')
            details['threat_reassurance'] = {
                'reason': 'Presenta amenazas pero intenta calmar al usuario'
            }
        
        # 5. BENEFICIOS + URGENCIA (Táctica de presión)
        # "Ha ganado un premio, pero debe reclamarlo YA"
        if has_benefits and urgency_score > 0.5:
            scores['benefit_urgency'] = 0.75
            flags.append('BENEFICIO_CON_URGENCIA')
            details['benefit_urgency'] = {
                'urgency': urgency_score,
                'reason': 'Ofrece beneficios pero presiona para actuar rápido'
            }
        
        # 6. AMABILIDAD + SOLICITUD DE DATOS (Manipulación)
        # Ser muy amable al pedir datos sensibles
        if politeness_score > 0.7 and has_data_request:
            scores['politeness_data'] = 0.8
            flags.append('AMABILIDAD_SOLICITANDO_DATOS')
            details['politeness_data'] = {
                'politeness': politeness_score,
                'reason': 'Amabilidad excesiva al solicitar datos sensibles'
            }
        
        # 7. SENTIMIENTO NEGATIVO + TRANQUILIZACIÓN (Contradicción)
        # Tono negativo pero diciendo "no se preocupe"
        if sentiment_result.get('label') == 'NEG' and has_reassurance:
            scores['negative_reassurance'] = 0.7
            flags.append('NEGATIVO_CON_TRANQUILIZACIÓN')
            details['negative_reassurance'] = {
                'sentiment': sentiment_result.get('label'),
                'compound': sentiment_result.get('compound', 0),
                'reason': 'Tono negativo pero intenta tranquilizar'
            }
        
        # Calcular score total (promedio ponderado)
        if scores:
            total_score = sum(scores.values()) / len(scores)
        else:
            total_score = 0.0
        
        # Determinar nivel de riesgo
        if total_score >= 0.8:
            risk_level = 'CRÍTICO'
        elif total_score >= 0.6:
            risk_level = 'ALTO'
        elif total_score >= 0.4:
            risk_level = 'MEDIO'
        elif total_score >= 0.2:
            risk_level = 'BAJO'
        else:
            risk_level = 'NORMAL'
        
        return {
            'score': total_score,
            'flags': flags,
            'details': details,
            'scores': scores,
            'risk_level': risk_level,
            'incongruence_count': len(flags)
        }
    
    def _detect_politeness(self, text):
        """Detectar nivel de amabilidad (0-1)"""
        count = sum(1 for word in self.politeness_words if word in text)
        return min(1.0, count / 3)
    
    def _detect_urgency(self, text):
        """Detectar nivel de urgencia (0-1)"""
        count = sum(1 for word in self.urgency_words if word in text)
        return min(1.0, count / 2)
    
    def _has_category(self, text, word_list):
        """Verificar si el texto contiene palabras de una categoría"""
        return any(word in text for word in word_list)
    
    def _empty_result(self):
        """Resultado vacío cuando no hay texto"""
        return {
            'score': 0.0,
            'flags': [],
            'details': {},
            'scores': {},
            'risk_level': 'NORMAL',
            'incongruence_count': 0
        }

class VishingKeywords:
    """
    Sistema de detección de keywords contextuales para vishing
    Categoriza keywords por tipo de amenaza con pesos diferenciados
    """
    
    CATEGORIES = {
        'bancarias': {
            'keywords': [
                'banco', 'tarjeta', 'cuenta', 'clave', 'token',
                'cvv', 'pin', 'número de tarjeta', 'verificar datos',
                'bloqueo de cuenta', 'transacción sospechosa',
                'departamento de seguridad', 'fraude detectado',
                'tarjeta bloqueada', 'movimiento inusual', 'actividad sospechosa',
                'confirmar identidad', 'datos bancarios', 'código de seguridad'
            ],
            'weight': 0.9,  # Alta severidad - temas bancarios muy riesgosos
            'threshold': 2,  # Mínimo 2 keywords para activar
            'description': 'Términos bancarios y financieros'
        },
        'urgencia': {
            'keywords': [
                'urgente', 'inmediato', 'ahora mismo', 'ya', 'rápido',
                '24 horas', 'último momento', 'última oportunidad',
                'antes de que', 'se cerrará', 'expira', 'caducidad',
                'tiempo limitado', 'de inmediato', 'cuanto antes',
                'sin demora', 'enseguida', 'pronto vence'
            ],
            'weight': 0.7,  # Alta urgencia = táctica de presión
            'threshold': 2,
            'description': 'Palabras que presionan con urgencia temporal'
        },
        'suplantacion': {
            'keywords': [
                'soy de su banco', 'le llamo de', 'departamento',
                'servicio al cliente', 'soporte técnico',
                'autoridades', 'policía', 'ministerio', 'hacienda',
                'soy del banco', 'llamo del', 'entidad bancaria',
                'oficial', 'representante', 'agente autorizado',
                'equipo de seguridad', 'área de fraude'
            ],
            'weight': 0.85,  # Muy alta - suplantación de identidad
            'threshold': 1,  # Con 1 keyword ya es sospechoso
            'description': 'Suplantación de identidad institucional'
        },
        'datos_sensibles': {
            'keywords': [
                'necesito que me confirme', 'dígame su',
                'verificar su', 'actualizar sus datos',
                'número de documento', 'fecha de nacimiento',
                'contraseña', 'código', 'otp', 'sms',
                'proporcione', 'indíqueme', 'confirme su',
                'número completo', 'dígitos', 'cédula',
                'pasaporte', 'dni', 'nit'
            ],
            'weight': 1.0,  # Máxima severidad - solicitud de datos críticos
            'threshold': 1,
            'description': 'Solicitud de información personal sensible'
        },
        'amenazas': {
            'keywords': [
                'demanda', 'proceso legal', 'denuncia', 'multa',
                'bloqueado', 'suspendido', 'inhabilitado',
                'consecuencias', 'responsable', 'penalización',
                'sanción', 'cancelado', 'problema legal',
                'acciones legales', 'deuda', 'embargo',
                'requerimiento judicial', 'citación'
            ],
            'weight': 0.8,  # Alta severidad - intimidación
            'threshold': 1,
            'description': 'Amenazas o intimidación legal/financiera'
        },
        'financieras': {
            'keywords': [
                'dinero fácil', 'ganancia garantizada', 'sin riesgo',
                'inversión segura', 'multiplica tu dinero',
                'millonario', 'sistema infalible',
                'dinero rápido', 'beneficio asegurado',
                'rentabilidad garantizada', 'ganancias inmediatas',
                'oportunidad única', 'oferta exclusiva',
                'préstamo fácil', 'crédito inmediato'
            ],
            'weight': 0.6,  # Media-alta - estafas financieras clásicas
            'threshold': 1,
            'description': 'Ofertas financieras fraudulentas'
        },
        'verificacion': {
            'keywords': [
                'verificar', 'confirmar', 'validar', 'actualizar',
                'renovar', 'reactivar', 'restaurar',
                'comprobar', 'autenticar', 'certificar',
                'registrar nuevamente', 'volver a ingresar',
                'sincronizar', 'activar de nuevo'
            ],
            'weight': 0.5,  # Media - común en phishing
            'threshold': 3,  # Requiere más keywords (son palabras comunes)
            'description': 'Verbos de verificación (comunes en phishing)'
        }
    }
    
    def __init__(self):
        """Inicializar analizador de keywords contextuales"""
        # Contar total de keywords
        total_keywords = sum(len(cat['keywords']) for cat in self.CATEGORIES.values())
        print(f"[OK] VishingKeywords inicializado con {total_keywords} keywords en {len(self.CATEGORIES)} categorías")
    
    def analyze(self, text):
        """
        Analizar texto buscando keywords categorizadas
        
        Args:
            text (str): Texto a analizar
            
        Returns:
            dict: {
                'categories': dict - Categorías detectadas con matches
                'total_score': float (0-1) - Score total normalizado
                'risk_level': str - Nivel de riesgo clasificado
                'keywords_found': list - Todas las keywords encontradas
                'category_count': int - Número de categorías activadas
            }
        """
        if not text:
            return {
                'categories': {},
                'total_score': 0.0,
                'risk_level': 'NORMAL',
                'keywords_found': [],
                'category_count': 0
            }
        
        text_lower = text.lower()
        detected = {}
        total_score = 0
        all_keywords = []
        
        for category, config in self.CATEGORIES.items():
            # Buscar keywords de esta categoría
            matches = [kw for kw in config['keywords'] if kw in text_lower]
            
            # Verificar si cumple el threshold mínimo
            if len(matches) >= config['threshold']:
                # Calcular score de esta categoría
                # Score aumenta con más matches, hasta un máximo (saturación en 3)
                category_score = config['weight'] * min(1.0, len(matches) / 3)
                
                detected[category] = {
                    'matches': matches,
                    'count': len(matches),
                    'weight': config['weight'],
                    'score': round(category_score, 3),
                    'description': config['description']
                }
                
                total_score += category_score
                all_keywords.extend(matches)
        
        # Normalizar score total (puede superar 1.0 si hay múltiples categorías)
        final_score = min(1.0, total_score)
        
        return {
            'categories': detected,
            'total_score': round(final_score, 3),
            'risk_level': self._classify_risk(final_score),
            'keywords_found': all_keywords,
            'category_count': len(detected)
        }
    
    def _classify_risk(self, score):
        """Clasificar nivel de riesgo basado en score"""
        if score >= 0.8:
            return 'CRÍTICO'
        elif score >= 0.6:
            return 'ALTO'
        elif score >= 0.4:
            return 'MEDIO'
        elif score >= 0.2:
            return 'BAJO'
        else:
            return 'NORMAL'
    
    def get_category_info(self, category_name):
        """Obtener información de una categoría específica"""
        if category_name in self.CATEGORIES:
            cat = self.CATEGORIES[category_name]
            return {
                'name': category_name,
                'description': cat['description'],
                'weight': cat['weight'],
                'threshold': cat['threshold'],
                'keyword_count': len(cat['keywords'])
            }
        return None
    
    def get_all_categories_info(self):
        """Obtener información de todas las categorías"""
        return {
            name: self.get_category_info(name) 
            for name in self.CATEGORIES.keys()
        }

class FraudDetector:
    def __init__(self):
        """Detector de fraude optimizado con keywords contextuales"""
        self.model = None
        self.vectorizer = None
        self.ml_available = False
        
        try:
            # Intentar cargar modelos ML
            print("[INFO] Cargando modelos de fraude...")
            
            try:
                self.model = joblib.load('best_model_lr.joblib')
                print("[OK] Modelo ML cargado")
                self.vectorizer = joblib.load('vectorizer_tfidf.joblib')
                print("[OK] Vectorizador TF-IDF cargado")
                self.ml_available = True
            except Exception as ml_error:
                print(f"[WARNING] No se pudieron cargar modelos ML: {type(ml_error).__name__}")
                print("[INFO] El detector funcionará con keywords contextuales")
                self.ml_available = False
            
            # Inicializar sistema de keywords contextuales (MEJORA 2)
            self.vishing_keywords = VishingKeywords()
            
            # Keywords legacy (mantenidas para compatibilidad)
            self.fraud_keywords = [
                'dinero fácil', 'dinero rapido', 'ganancia garantizada', 'sin riesgo',
                'inversión segura', 'oportunidad única', 'acción urgente', 'acción inmediata',
                'aprovecha ahora', 'oferta limitada', 'multiplica tu dinero', 'ingresos pasivos',
                'trabajo desde casa', 'gana dinero online', 'millonario en meses',
                'sistema infalible', 'fórmula secreta', 'estrategia ganadora'
            ]
            
            # Configuración de threshold
            self.fraud_threshold = 60  # Reducido de 80% a 60% (más sensible)
            
            print("[OK] Detector de fraude cargado correctamente")
            
        except Exception as e:
            print(f"[ERROR] Error cargando detector de fraude: {e}")
            raise
    
    def analyze_text(self, text):
        """
        Analizar texto para detectar fraude usando keywords contextuales
        
        Returns:
            dict: {
                'is_fraud': bool - Detección binaria
                'probability': float (0-100) - Probabilidad de fraude
                'keywords_found': list - Keywords legacy encontradas
                'keyword_analysis': dict - Análisis contextual detallado
                'status': str - Estado del análisis
                'ml_available': bool - Si ML está disponible
            }
        """
        if not text:
            return {
                'is_fraud': False,
                'probability': 0,
                'keywords_found': [],
                'keyword_analysis': None,
                'status': 'no_text',
                'ml_available': self.ml_available
            }
        
        try:
            # ========== ANÁLISIS CON KEYWORDS CONTEXTUALES (MEJORA 2) ==========
            keyword_analysis = self.vishing_keywords.analyze(text)
            
            # Análisis legacy de keywords (para compatibilidad)
            text_lower = text.lower()
            keywords_found = [kw for kw in self.fraud_keywords if kw in text_lower]
            
            # ========== PROBABILIDAD DE FRAUDE ==========
            # Priorizar análisis contextual sobre legacy
            if keyword_analysis['total_score'] > 0:
                # Usar score contextual (0-1) convertido a porcentaje
                keyword_probability = keyword_analysis['total_score'] * 100
            else:
                # Fallback a keywords legacy
                keyword_probability = min(len(keywords_found) * 20, 100)
            
            # Si ML está disponible, combinar con keywords contextuales
            if self.ml_available and self.model and self.vectorizer:
                # Análisis ML
                text_vectorized = self.vectorizer.transform([text])
                ml_probability = float(self.model.predict_proba(text_vectorized)[0][1]) * 100
                
                # Combinar ML (70%) + Keywords Contextuales (30%)
                fraud_probability = (ml_probability * 0.7 + keyword_probability * 0.3)
            else:
                # Solo keywords contextuales
                fraud_probability = keyword_probability
            
            # Determinar si es fraude
            is_fraud = fraud_probability >= self.fraud_threshold
            
            return {
                'is_fraud': is_fraud,
                'probability': round(fraud_probability, 2),
                'keywords_found': keywords_found,  # Legacy keywords
                'keyword_analysis': keyword_analysis,  # Nuevo análisis contextual
                'status': 'analyzed' if self.ml_available else 'keywords_only',
                'threshold': self.fraud_threshold,
                'ml_available': self.ml_available
            }
            
        except Exception as e:
            print(f"[ERROR] Error analizando texto: {e}")
            return {
                'is_fraud': False,
                'probability': 0,
                'keywords_found': [],
                'keyword_analysis': None,
                'status': 'error',
                'ml_available': self.ml_available
            }

class AudioTranscriptor:
    def __init__(self):
        """Inicializar transcriptor de audio modular"""
        
        # Inicializar gestor de motores
        self.engine_manager = TranscriptionEngineManager()
        
        # Inicializar gestor de sentimientos
        print("[INFO] Inicializando gestor de análisis de sentimientos...")
        self.sentiment_manager = SentimentEngineManager()
        print("[OK] Gestor de sentimientos inicializado")
        
        # Configuración de audio por defecto (Perfil 1 - Oficina tranquila)
        self.audio_config = {
            'energy_threshold': 600,
            'dynamic_energy_threshold': True,
            'dynamic_energy_adjustment_damping': 0.18,
            'dynamic_energy_adjustment_ratio': 1.5,
            'pause_threshold': 0.6,
            'non_speaking_duration': 0.3,
            'listen_timeout': 3,
            'phrase_time_limit': 10,
            'language': 'es-ES',
            # VAD (webrtcvad)
            'vad_enabled': False,
            'vad_aggressiveness': 2,   # 0..3
            'vad_padding_ms': 250,     # pre/post relleno en ms
            'vad_frame_ms': 20,        # 10/20/30 ms soportados
            'vad_min_segment_ms': 250, # duración mínima de segmento
            'sample_rate': 16000,      # 16k mono para VAD
            'device_index': None       # índice de dispositivo de audio
        }
        
        # Perfiles predefinidos
        self.audio_profiles = {
            'office': {
                'name': 'Oficina Tranquila',
                'description': 'Alta precisión, sin ruido de fondo',
                'config': {
                    'energy_threshold': 600,
                    'dynamic_energy_threshold': True,
                    'dynamic_energy_adjustment_damping': 0.18,
                    'dynamic_energy_adjustment_ratio': 1.5,
                    'pause_threshold': 0.6,
                    'non_speaking_duration': 0.3,
                    'listen_timeout': 3,
                    'phrase_time_limit': 10,
                    'vad_enabled': False,
                    'vad_aggressiveness': 2,
                    'vad_padding_ms': 250,
                    'vad_frame_ms': 20,
                    'vad_min_segment_ms': 250,
                    'sample_rate': 16000
                }
            },
            'callcenter': {
                'name': 'Call Center',
                'description': 'Ruido variable, baja latencia con VAD',
                'config': {
                    'energy_threshold': 400,
                    'dynamic_energy_threshold': True,
                    'dynamic_energy_adjustment_damping': 0.15,
                    'dynamic_energy_adjustment_ratio': 1.5,
                    'pause_threshold': 0.5,
                    'non_speaking_duration': 0.3,
                    'listen_timeout': 2,
                    'phrase_time_limit': 8,
                    'vad_enabled': True,
                    'vad_aggressiveness': 2,
                    'vad_padding_ms': 250,
                    'vad_frame_ms': 20,
                    'vad_min_segment_ms': 250,
                    'sample_rate': 16000
                }
            },
            'voip': {
                'name': 'Telefónico/VoIP',
                'description': 'Optimizado para llamadas (8kHz)',
                'config': {
                    'energy_threshold': 350,
                    'dynamic_energy_threshold': True,
                    'dynamic_energy_adjustment_damping': 0.20,
                    'dynamic_energy_adjustment_ratio': 1.5,
                    'pause_threshold': 0.5,
                    'non_speaking_duration': 0.3,
                    'listen_timeout': 3,
                    'phrase_time_limit': 8,
                    'vad_enabled': True,
                    'vad_aggressiveness': 3,
                    'vad_padding_ms': 250,
                    'vad_frame_ms': 20,
                    'vad_min_segment_ms': 250,
                    'sample_rate': 8000
                }
            }
        }
        
        # Estado del sistema
        self.is_listening = False
        self.is_changing_profile = False  # Bandera para cambio de perfil
        self.microphone = None
        self.audio_queue = queue.Queue()
        self.listen_thread = None
        
        # Detector de fraude
        self.fraud_detector = FraudDetector()
        
        # Sistema de scoring multinivel de vishing
        self.vishing_scorer = VishingScorer()
        
        # Sistema de thresholds dinámicos (MEJORA 6)
        self.adaptive_threshold = AdaptiveThreshold()
        
        # Analizador acústico (MEJORA 7)
        self.acoustic_analyzer = AcousticAnalyzer()
        
        # Generador de explicaciones (MEJORA 8)
        self.explainable_detector = ExplainableVishingDetector()
        
        # Analizador lingüístico (MEJORA 3)
        self.linguistic_analyzer = LinguisticAnalyzer()
        
        # Analizador de conversación (MEJORA 4)
        self.conversation_analyzer = ConversationAnalyzer(window_size=10)
        
        # Detector de incongruencias (MEJORA 5)
        self.incongruence_detector = IncongruenceDetector()
        
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
        
        print("[CONFIG] Configuracion de audio aplicada:", self.audio_config)
        
        # Inicializar micrófono
        self._initialize_microphone()
    
    def _setup_default_engine(self):
        """Configurar motor por defecto"""
        config = {'audio': self.audio_config}
        if self.engine_manager.set_engine('deepspeech', config):
            print("[OK] Motor DeepSpeech configurado como predeterminado")
        else:
            print("[WARNING] Error configurando motor predeterminado")
    
    def _initialize_microphone(self):
        """Inicializar micrófono con configuración optimizada"""
        try:
            print("[AUDIO] Buscando microfonos disponibles...")
            
            # Configurar sample rate y chunk size según VAD
            srate = int(self.audio_config.get('sample_rate', 16000))
            frame_ms = int(self.audio_config.get('vad_frame_ms', 20))
            chunk = max(160, int(srate * frame_ms / 1000))  # ej: 320 @16kHz/20ms
            
            # Inicializar micrófono con parámetros optimizados
            device_idx = self.audio_config.get('device_index', None)
            self.microphone = sr.Microphone(
                device_index=device_idx,
                sample_rate=srate,
                chunk_size=chunk
            )
            
            # Configurar recognizer con el motor actual
            if self.engine_manager.current_engine:
                recognizer = self.engine_manager.current_engine.recognizer
                
                # Aplicar configuración de audio completa
                recognizer.energy_threshold = self.audio_config.get('energy_threshold', 600)
                recognizer.dynamic_energy_threshold = self.audio_config.get('dynamic_energy_threshold', True)
                recognizer.dynamic_energy_adjustment_damping = self.audio_config.get('dynamic_energy_adjustment_damping', 0.18)
                recognizer.dynamic_energy_adjustment_ratio = self.audio_config.get('dynamic_energy_adjustment_ratio', 1.5)
                recognizer.pause_threshold = self.audio_config.get('pause_threshold', 0.6)
                recognizer.non_speaking_duration = self.audio_config.get('non_speaking_duration', 0.3)
                
                print(f"[OK] Microfono inicializado: {srate}Hz, chunk={chunk}")
                print(f"[CONFIG] Umbral energia: {recognizer.energy_threshold}")
                print(f"[CONFIG] VAD: {'Activado' if self.audio_config.get('vad_enabled') else 'Desactivado'}")
                
                # Calibrar micrófono (con timeout para no bloquear servidor)
                try:
                    with self.microphone as source:
                        print("[AUDIO] Calibrando microfono para ruido ambiental...")
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Reducido a 0.5s
                        print(f"[OK] Umbral calibrado: {recognizer.energy_threshold}")
                except Exception as calib_error:
                    print(f"[WARNING] No se pudo calibrar micrófono: {calib_error}")
                    print("[INFO] Usando umbral de energía predeterminado")
            
        except Exception as e:
            print(f"[ERROR] Error inicializando microfono: {e}")
            self.microphone = None
    
    def change_engine(self, engine_id, engine_config=None):
        """Cambiar motor de transcripción"""
        try:
            print(f"[ENGINE] Cambiando motor de transcripcion a: {engine_id}")
            
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
                print(f"[OK] Motor cambiado exitosamente a: {engine_id}")
                
                # Reinicializar micrófono con nuevo motor
                self._initialize_microphone()
                
                # Reanudar transcripción si estaba activa
                if was_listening:
                    self.start_listening()
                
                return True
            else:
                print(f"[ERROR] No se pudo cambiar al motor: {engine_id}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error cambiando motor: {e}")
            return False
    
    def get_available_engines(self):
        """Obtener motores disponibles"""
        return self.engine_manager.get_available_engines()
    
    def get_current_engine_info(self):
        """Obtener información del motor actual"""
        return self.engine_manager.get_current_engine_info()
    
    def load_audio_profile(self, profile_name):
        """Cargar un perfil de audio predefinido"""
        if profile_name not in self.audio_profiles:
            print(f"[ERROR] Perfil '{profile_name}' no encontrado")
            return False
        
        try:
            print(f"[AUDIO] Cargando perfil de audio: {profile_name}")
            
            # Activar bandera de cambio de perfil
            self.is_changing_profile = True
            
            # Detener transcripción si está activa (sin emitir notificaciones de error)
            was_listening = self.is_listening
            if was_listening:
                self.is_listening = False  # Detener silenciosamente
                if self.listen_thread:
                    self.listen_thread.join(timeout=2)
                print("[TRANSCRIPTION] Transcripcion detenida para cambio de perfil")
            
            profile = self.audio_profiles[profile_name]
            
            # Actualizar configuración de audio
            self.audio_config.update(profile['config'])
            
            # Mantener language y device_index si existen
            if 'language' not in profile['config']:
                profile['config']['language'] = self.audio_config.get('language', 'es-ES')
            if 'device_index' not in profile['config']:
                profile['config']['device_index'] = self.audio_config.get('device_index', None)
            
            # Reinicializar micrófono con nueva configuración
            self._initialize_microphone()
            
            print(f"[OK] Perfil '{profile['name']}' cargado exitosamente")
            print(f"[INFO] {profile['description']}")
            
            # Desactivar bandera de cambio de perfil
            self.is_changing_profile = False
            
            # Reanudar transcripción si estaba activa
            if was_listening:
                self.start_listening()
            
            return True
            
        except Exception as e:
            self.is_changing_profile = False
            print(f"[ERROR] Error cargando perfil: {e}")
            return False
    
    def get_audio_profiles(self):
        """Obtener lista de perfiles disponibles"""
        return {
            profile_id: {
                'name': profile['name'],
                'description': profile['description']
            }
            for profile_id, profile in self.audio_profiles.items()
        }
    
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
            
            print(f"[CONFIG] Configuracion de audio actualizada: {new_config}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error actualizando configuracion: {e}")
            return False
    
    def _convert_to_json_serializable(self, obj):
        """Convertir objetos numpy y otros tipos a tipos serializables en JSON"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_json_serializable(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    def start_listening(self):
        """Iniciar transcripción en tiempo real"""
        if self.is_listening:
            return
        
        if not self.microphone or not self.engine_manager.current_engine:
            print("[ERROR] Microfono o motor no disponible")
            return
        
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        print("[TRANSCRIPTION] Iniciando transcripcion en tiempo real...")
    
    def stop_listening(self):
        """Detener transcripción"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        print("[TRANSCRIPTION] Transcripcion detenida")
    
    def _listen_loop(self):
        """Loop principal de escucha"""
        if not self.engine_manager.current_engine or not self.microphone:
            # No emitir error si estamos cambiando de perfil
            if not self.is_changing_profile:
                print("[ERROR] No hay motor activo o microfono disponible")
                socketio.emit('error', {'message': 'No hay motor activo o micrófono disponible'})
            return
        
        recognizer = self.engine_manager.current_engine.recognizer
        print(f"[TRANSCRIPTION] Iniciando loop de escucha con motor: {self.engine_manager.current_engine_name}")
        
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
                        
        print("[TRANSCRIPTION] Loop de escucha terminado")
        socketio.emit('listening_status', {'status': 'stopped', 'message': 'Escucha detenida'})
    
    def _process_audio(self, audio_data):
        """Procesar audio transcrito"""
        try:
            print("[AUDIO] Procesando audio capturado...")
            
            # Transcribir usando el motor actual
            text = self.engine_manager.transcribe(audio_data)
            
            if text:
                print(f"[TRANSCRIPTION] Texto transcrito: '{text}'")
                
                # Actualizar estadísticas
                self.stats['total_transcriptions'] += 1
                
                # ========== ANÁLISIS DE SENTIMIENTO ==========
                print("[SENTIMENT] Analizando sentimiento del texto...")
                sentiment_result = self.sentiment_manager.analyze_text(text)
                print(f"[SENTIMENT] Compuesto: {sentiment_result['compound']:.3f}, Etiqueta: {sentiment_result['label']}")
                if sentiment_result.get('emotions'):
                    print(f"[SENTIMENT] Emociones: {sentiment_result['emotions']}")
                
                # ========== ANÁLISIS DE FRAUDE (Keywords + ML) ==========
                fraud_analysis = self.fraud_detector.analyze_text(text)
                
                # Logging de keywords contextuales
                if fraud_analysis.get('keyword_analysis'):
                    kw_analysis = fraud_analysis['keyword_analysis']
                    if kw_analysis['category_count'] > 0:
                        print(f"[KEYWORDS] {kw_analysis['category_count']} categorías detectadas | "
                              f"Score: {kw_analysis['total_score']*100:.1f}% | "
                              f"Nivel: {kw_analysis['risk_level']}")
                        for cat_name, cat_data in kw_analysis['categories'].items():
                            print(f"[KEYWORDS]   • {cat_name}: {cat_data['count']} matches "
                                  f"(score: {cat_data['score']*100:.1f}%)")
                            print(f"[KEYWORDS]     Palabras: {', '.join(cat_data['matches'][:5])}"
                                  f"{'...' if len(cat_data['matches']) > 5 else ''}")
                
                # ========== SISTEMA MULTINIVEL DE VISHING ==========
                # Preparar features para el scoring multinivel
                features = {}
                
                # 1. Keywords: normalizar probabilidad del detector (0-100 → 0-1)
                if fraud_analysis.get('probability') is not None:
                    features['keywords'] = fraud_analysis['probability'] / 100.0
                
                # 2. ML Model: usar probabilidad si modelo está disponible
                if fraud_analysis.get('ml_available') and fraud_analysis.get('probability'):
                    features['ml_model'] = fraud_analysis['probability'] / 100.0
                
                # 3. Sentiment: usar score de riesgo conversacional
                fraud_risk_score = self.sentiment_manager.compute_fraud_risk_score()
                
                # MEJORA 5: Detector de Incongruencias (mejora el análisis de sentimiento)
                print("[INCONGRUENCE] Analizando incongruencias y señales mixtas...")
                incongruence_result = self.incongruence_detector.analyze(text, sentiment_result)
                
                # Combinar sentiment base con detección de incongruencias (60/40)
                combined_sentiment_score = (fraud_risk_score * 0.6) + (incongruence_result['score'] * 0.4)
                features['sentiment'] = combined_sentiment_score
                
                # Logging de incongruencias
                if incongruence_result['incongruence_count'] > 0:
                    print(f"[INCONGRUENCE] {incongruence_result['incongruence_count']} incongruencias detectadas | "
                          f"Score: {incongruence_result['score']*100:.1f}% | "
                          f"Nivel: {incongruence_result['risk_level']}")
                    print(f"[INCONGRUENCE] Flags: {', '.join(incongruence_result['flags'][:5])}")
                
                # 4. Linguistic: MEJORA 3 - Análisis de patrones lingüísticos
                print("[LINGUISTIC] Analizando patrones lingüísticos...")
                linguistic_result = self.linguistic_analyzer.analyze(text)
                features['linguistic'] = linguistic_result['total_score']
                
                # Logging de análisis lingüístico
                if linguistic_result['pattern_count'] > 0:
                    print(f"[LINGUISTIC] {linguistic_result['pattern_count']} patrones detectados | "
                          f"Score: {linguistic_result['total_score']*100:.1f}% | "
                          f"Nivel: {linguistic_result['risk_level']}")
                    print(f"[LINGUISTIC] Flags: {', '.join(linguistic_result['flags'][:5])}")
                
                # 5. Temporal: MEJORA 4 - Análisis temporal de conversación
                print("[TEMPORAL] Agregando turno y analizando patrones temporales...")
                self.conversation_analyzer.add_turn(text, speaker='system')
                temporal_result = self.conversation_analyzer.analyze_patterns()
                features['temporal'] = temporal_result['total_score']
                
                # Logging de análisis temporal
                if temporal_result['pattern_count'] > 0:
                    print(f"[TEMPORAL] {temporal_result['pattern_count']} patrones detectados | "
                          f"Score: {temporal_result['total_score']*100:.1f}% | "
                          f"Nivel: {temporal_result['risk_level']}")
                    print(f"[TEMPORAL] Flags: {', '.join(temporal_result['flags'][:5])}")
                    if temporal_result.get('metrics'):
                        print(f"[TEMPORAL] Turnos: {temporal_result['turn_count']}")
                
                # 6. Acoustic: MEJORA 7 - Análisis de features acústicas
                print("[ACOUSTIC] Analizando características acústicas del audio...")
                acoustic_result = self.acoustic_analyzer.analyze_audio(
                    audio_data=audio_data,
                    text=text,
                    sample_rate=16000
                )
                features['acoustic'] = acoustic_result['score']
                
                # Logging de análisis acústico
                if acoustic_result['analysis_success']:
                    print(f"[ACOUSTIC] Score: {acoustic_result['percentage']}% | "
                          f"Nivel: {acoustic_result['risk_level']} | "
                          f"Flags: {acoustic_result['flag_count']}")
                    if acoustic_result['flags']:
                        print(f"[ACOUSTIC] Flags detectadas: {', '.join(acoustic_result['flags'][:5])}")
                    if acoustic_result.get('features'):
                        feats = acoustic_result['features']
                        if 'speaking_rate' in feats and feats['speaking_rate'] > 0:
                            print(f"[ACOUSTIC] Velocidad: {feats['speaking_rate']:.1f} pal/seg | "
                                  f"Silencios: {feats['silence_ratio']*100:.1f}%")
                else:
                    print(f"[ACOUSTIC] ⚠️ Análisis acústico falló, usando score=0.0")
                
                # Calcular score de vishing multinivel
                vishing_result = self.vishing_scorer.compute_vishing_score(features)
                
                # MEJORA 6: Aplicar thresholds dinámicos
                # Clasificar con contexto adaptativo basado en keywords detectadas
                adaptive_classification = self.adaptive_threshold.classify(
                    score=vishing_result['score'],
                    context='auto',
                    text=text,
                    keyword_analysis=fraud_analysis
                )
                
                # Actualizar resultado con clasificación adaptativa
                vishing_result['adaptive'] = {
                    'classification': adaptive_classification[0],
                    'risk_level': adaptive_classification[1],
                    'threshold_used': adaptive_classification[2],
                    'security_context': adaptive_classification[3]
                }
                
                # Logging detallado
                print(f"[VISHING] Score: {vishing_result['percentage']}% | "
                      f"Nivel: {vishing_result['risk_level']} | "
                      f"Confianza: {vishing_result['confidence']*100:.0f}%")
                print(f"[ADAPTIVE] Clasificación: {vishing_result['adaptive']['classification']} | "
                      f"Contexto: {vishing_result['adaptive']['security_context']} | "
                      f"Threshold: {vishing_result['adaptive']['threshold_used']*100:.0f}%")
                
                if vishing_result['is_vishing'] or vishing_result['adaptive']['classification'] == 'FRAUDE':
                    self.stats['fraud_detected'] += 1
                    print(f"[VISHING] ⚠️ VISHING DETECTADO ⚠️")
                    print(f"[VISHING] Desglose:")
                    for feature, data in vishing_result['breakdown'].items():
                        print(f"[VISHING]   • {feature}: {data['value']*100:.1f}% "
                              f"(contribución: {data['contribution']*100:.1f}%)")
                
                # Crear resultado completo
                result = {
                    'text': text,
                    'timestamp': datetime.now().isoformat(),
                    
                    # Análisis tradicional (legacy)
                    'fraud_analysis': fraud_analysis,
                    'fraud_risk_score': fraud_risk_score,
                    
                    # Nuevo sistema multinivel
                    'vishing_analysis': vishing_result,
                    'is_fraud': vishing_result['is_vishing'],
                    'combined_fraud_probability': vishing_result['percentage'],
                    
                    # Análisis de sentimiento
                    'sentiment_analysis': sentiment_result,
                    
                    # Detector de incongruencias (MEJORA 5)
                    'incongruence_analysis': incongruence_result,
                    
                    # Análisis lingüístico (MEJORA 3)
                    'linguistic_analysis': linguistic_result,
                    
                    # Análisis temporal (MEJORA 4)
                    'temporal_analysis': temporal_result,
                    
                    # Análisis acústico (MEJORA 7)
                    'acoustic_analysis': acoustic_result,
                    
                    # Explicación humana (MEJORA 8)
                    'explanation': self.explainable_detector.generate_explanation(
                        vishing_result=vishing_result,
                        fraud_analysis=fraud_analysis,
                        sentiment_result=sentiment_result,
                        linguistic_result=linguistic_result,
                        temporal_result=temporal_result,
                        acoustic_result=acoustic_result,
                        incongruence_result=incongruence_result,
                        adaptive_result=vishing_result.get('adaptive', None)
                    ),
                    
                    # Metadatos
                    'engine_info': self.get_current_engine_info(),
                    'sentiment_engine': self.sentiment_manager.get_current_engine_name(),
                    'audio_config': self.audio_config,
                    'success': True
                }
                
                # Guardar en historial
                self.transcription_history.append(result)
                
                # Convertir a tipos serializables antes de emitir
                result_serializable = self._convert_to_json_serializable(result)
                
                # Emitir resultado via SocketIO
                socketio.emit('transcription_result', result_serializable)
                
                print("[OK] Resultado enviado via SocketIO")
                
            else:
                print("[WARNING] No se pudo transcribir el audio (silencio o ruido)")
                # Emitir información de debug
                debug_info = {
                    'message': 'Audio capturado pero no se pudo transcribir (posiblemente silencio)',
                    'timestamp': datetime.now().isoformat(),
                    'engine_info': self.get_current_engine_info()
                }
                socketio.emit('transcription_debug', self._convert_to_json_serializable(debug_info))
                
        except Exception as e:
            error_msg = f"Error procesando audio: {e}"
            print(f"[ERROR] {error_msg}")
            
            # Emitir error específico
            error_info = {
                'message': error_msg,
                'timestamp': datetime.now().isoformat(),
                'engine_info': self.get_current_engine_info()
            }
            socketio.emit('transcription_error', self._convert_to_json_serializable(error_info))

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
                        
                        # Validar duración mínima del segmento
                        min_ms = self.audio_config.get('vad_min_segment_ms', 250)
                        segment_duration_ms = (len(segment_bytes) / bytes_per_sample) / sample_rate * 1000
                        
                        if segment_duration_ms >= min_ms:
                            # Convertir a AudioData (PCM16 mono)
                            audio_data = sr.AudioData(segment_bytes, sample_rate, bytes_per_sample)
                            threading.Thread(
                                target=self._process_audio,
                                args=(audio_data,),
                                daemon=True
                            ).start()
                        else:
                            print(f"⏭️ Segmento descartado (demasiado corto: {segment_duration_ms:.0f}ms < {min_ms}ms)")
                        
                        voiced_frames = []
                        ring_buffer.clear()
                        triggered = False

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
        .profile-card { transition: all 0.3s; border: 2px solid #dee2e6; }
        .profile-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); border-color: #ffc107; }
        .profile-card.active { border: 3px solid #ffc107; background: #fffbf0; }
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
                <p class="text-white-50">Múltiples motores: DeepSpeech • Whisper • Silero</p>
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
                                title="Cada motor tiene diferentes fortalezas: DeepSpeech (equilibrado), Whisper (alta precisión), Silero (rápido). Haz clic en una tarjeta para cambiar de motor">
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
                        <!-- Perfiles Predefinidos -->
                        <div class="row mb-4">
                            <div class="col-12">
                                <h6 class="mb-3">
                                    <i class="fas fa-magic"></i> Perfiles Predefinidos
                                    <button type="button" class="btn btn-sm btn-outline-info ms-2" 
                                            data-bs-toggle="tooltip" data-bs-placement="top" 
                                            title="Selecciona un perfil optimizado según tu entorno. Cada perfil ajusta automáticamente todos los parámetros de audio para un caso de uso específico">
                                        <i class="fas fa-info-circle"></i>
                                    </button>
                                </h6>
                                <div class="row" id="audio-profiles-container">
                                    <!-- Se llenan dinámicamente desde JavaScript -->
                                    <div class="col-md-4 mb-3">
                                        <div class="card h-100 profile-card active" id="profile-office" onclick="loadAudioProfile('office')" data-profile="office" style="cursor: pointer; transition: all 0.3s;">
                                            <div class="card-body text-center">
                                                <h6>
                                                    <i class="fas fa-building"></i> Oficina Tranquila
                                                    <span class="badge bg-warning ms-2 active-badge">ACTIVO</span>
                                                </h6>
                                                <p class="small text-muted mb-0">Alta precisión, sin ruido de fondo</p>
                                                <span class="badge bg-success mt-2">Recomendado</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <div class="card h-100 profile-card" id="profile-callcenter" onclick="loadAudioProfile('callcenter')" data-profile="callcenter" style="cursor: pointer; transition: all 0.3s;">
                                            <div class="card-body text-center">
                                                <h6>
                                                    <i class="fas fa-headset"></i> Call Center
                                                    <span class="badge bg-warning ms-2 active-badge" style="display: none;">ACTIVO</span>
                                                </h6>
                                                <p class="small text-muted mb-0">Ruido variable, baja latencia con VAD</p>
                                                <span class="badge bg-primary mt-2">VAD Activado</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <div class="card h-100 profile-card" id="profile-voip" onclick="loadAudioProfile('voip')" data-profile="voip" style="cursor: pointer; transition: all 0.3s;">
                                            <div class="card-body text-center">
                                                <h6>
                                                    <i class="fas fa-phone"></i> Telefónico/VoIP
                                                    <span class="badge bg-warning ms-2 active-badge" style="display: none;">ACTIVO</span>
                                                </h6>
                                                <p class="small text-muted mb-0">Optimizado para llamadas (8kHz)</p>
                                                <span class="badge bg-warning mt-2">8kHz</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <hr class="my-4">
                        
                        <!-- Controles Avanzados -->
                        <div class="row mb-3">
                            <div class="col-12">
                                <h6>
                                    <i class="fas fa-cog"></i> Controles Avanzados
                                    <button class="btn btn-sm btn-outline-secondary ms-2" onclick="toggleAdvancedControls()">
                                        <i class="fas fa-chevron-down" id="advanced-chevron"></i> Mostrar/Ocultar
                                    </button>
                                </h6>
                            </div>
                        </div>
                        
                        <div id="advanced-controls" style="display: none;">
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
                        <!-- Fin de Controles Avanzados -->
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
                
                <!-- Análisis de Sentimiento -->
                <div class="card mt-3">
                    <div class="card-header">
                        <h6><i class="fas fa-brain"></i> Análisis de Sentimiento</h6>
                    </div>
                    <div class="card-body">
                        <div id="sentiment-info">
                            <p class="text-muted">Esperando transcripción...</p>
                        </div>
                        <div id="sentiment-metrics" class="mt-2">
                            <!-- Métricas de conversación -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Sección de Motores de Sentimiento -->
        <div class="container mt-4">
            <div class="card">
                <div class="card-header bg-info text-white">
                    <h5><i class="fas fa-brain"></i> Motores de Análisis de Sentimiento</h5>
                    <small>Selecciona el modelo para analizar emociones y sentimientos en tiempo real</small>
                </div>
                <div class="card-body">
                    <div class="row" id="sentiment-engine-grid">
                        <!-- Los motores se cargarán dinámicamente aquí -->
                        <div class="col-12 text-center">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Cargando motores...</span>
                            </div>
                            <p class="mt-2">Cargando motores de sentimiento...</p>
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

        console.log('🔵 [SENTIMENT] Definiendo función loadSentimentEngines...');
        
        // Cargar motores de sentimiento disponibles
        function loadSentimentEngines() {
            console.log('🔵 [SENTIMENT] Iniciando loadSentimentEngines()...');
            
            const grid = document.getElementById('sentiment-engine-grid');
            if (!grid) {
                console.error('❌ [SENTIMENT] No se encontró sentiment-engine-grid');
                return;
            }
            
            console.log('🔵 [SENTIMENT] Grid encontrado, haciendo fetch...');
            
            fetch('/api/sentiment_engines')
                .then(response => {
                    console.log(`🔵 [SENTIMENT] Response status: ${response.status}`);
                    return response.json();
                })
                .then(data => {
                    console.log('🔵 [SENTIMENT] Data:', data);
                    
                    grid.innerHTML = ''; // Limpiar spinner
                    
                    if (!data.engines || !Array.isArray(data.engines) || data.engines.length === 0) {
                        console.error('❌ [SENTIMENT] No engines:', data);
                        grid.innerHTML = '<div class="col-12"><p class="text-danger">No se encontraron motores</p></div>';
                        return;
                    }
                    
                    console.log(`✅ [SENTIMENT] ${data.engines.length} motores encontrados`);
                    
                    data.engines.forEach((engine, index) => {
                        console.log(`� [SENTIMENT] Creando card para: ${engine.name}`);
                        
                        const col = document.createElement('div');
                        col.className = 'col-md-3 mb-3';
                        
                        col.innerHTML = `
                            <div class="card engine-card h-100 ${engine.is_active ? 'active' : ''}" 
                                 onclick="selectSentimentEngine('${engine.id}')" 
                                 style="cursor: pointer;">
                                <div class="card-body text-center">
                                    <h6 class="card-title">
                                        <i class="fas fa-brain"></i> ${engine.name}
                                        ${engine.is_active ? '<span class="badge bg-success ms-2">Activo</span>' : ''}
                                    </h6>
                                    <p class="card-text small">${engine.description}</p>
                                    <p class="card-text small"><i class="fas fa-database"></i> ${engine.model_size}</p>
                                    <p class="card-text small"><i class="fas fa-tachometer-alt"></i> ${engine.speed}</p>
                                </div>
                            </div>
                        `;
                        
                        grid.appendChild(col);
                    });
                    
                    console.log('✅ [SENTIMENT] Todas las tarjetas creadas');
                })
                .catch(error => {
                    console.error('❌ [SENTIMENT] Error:', error);
                    grid.innerHTML = `<div class="col-12"><p class="text-danger">Error: ${error.message}</p></div>`;
                });
        }
        
        // Seleccionar motor de sentimiento
        function selectSentimentEngine(engineId) {
            debugLog(`[SENTIMENT] Seleccionando motor: ${engineId}`);
            
            fetch('/api/change_sentiment_engine', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ engine_id: engineId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    debugLog(`[OK] Motor de sentimiento cambiado a: ${data.engine}`);
                    loadSentimentEngines(); // Recargar para actualizar el estado activo
                } else {
                    debugLog(`[ERROR] Error cambiando motor: ${data.error}`);
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                debugLog(`[ERROR] Error cambiando motor de sentimiento: ${error}`);
                alert('Error cambiando motor: ' + error.message);
            });
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

        let advancedControlsVisible = false;
        
        function toggleAdvancedControls() {
            const controls = document.getElementById('advanced-controls');
            const chevron = document.getElementById('advanced-chevron');
            
            if (advancedControlsVisible) {
                controls.style.display = 'none';
                chevron.className = 'fas fa-chevron-right';
            } else {
                controls.style.display = 'block';
                chevron.className = 'fas fa-chevron-down';
            }
            advancedControlsVisible = !advancedControlsVisible;
        }

        function loadAudioProfile(profileName) {
            debugLog('🎨 Cargando perfil de audio: ' + profileName);
            
            // Marcar perfil como activo inmediatamente
            document.querySelectorAll('.profile-card').forEach(card => {
                card.classList.remove('active');
                const badge = card.querySelector('.active-badge');
                if (badge) badge.style.display = 'none';
            });
            
            const clickedCard = event.target.closest('.profile-card');
            clickedCard.classList.add('active');
            const activeBadge = clickedCard.querySelector('.active-badge');
            if (activeBadge) activeBadge.style.display = 'inline-block';
            
            fetch('/api/load_audio_profile', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({profile_name: profileName})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification(`✅ Perfil "${profileName}" cargado exitosamente`, 'success');
                    debugLog('✅ Perfil cargado: ' + profileName);
                    
                    // Recargar configuración de audio en la UI
                    loadCurrentAudioConfig();
                } else {
                    showNotification('❌ Error cargando perfil: ' + data.error, 'error');
                    // Quitar la marca active si falló
                    clickedCard.classList.remove('active');
                    if (activeBadge) activeBadge.style.display = 'none';
                }
            })
            .catch(error => {
                debugLog('❌ Error cargando perfil: ' + error.message);
                showNotification('❌ Error cargando perfil', 'error');
                // Quitar la marca active si falló
                clickedCard.classList.remove('active');
                if (activeBadge) activeBadge.style.display = 'none';
            });
        }

        function loadCurrentAudioConfig() {
            fetch('/api/audio_config')
                .then(response => response.json())
                .then(data => {
                    if (data.config) {
                        const config = data.config;
                        
                        // Actualizar controles avanzados
                        document.getElementById('energy_threshold').value = config.energy_threshold || 600;
                        document.getElementById('energy_value').textContent = config.energy_threshold || 600;
                        document.getElementById('pause_threshold').value = config.pause_threshold || 0.6;
                        document.getElementById('pause_value').textContent = config.pause_threshold || 0.6;
                        document.getElementById('phrase_time_limit').value = config.phrase_time_limit || 10;
                        document.getElementById('phrase_value').textContent = config.phrase_time_limit || 10;
                        document.getElementById('listen_timeout').value = config.listen_timeout || 3;
                        document.getElementById('timeout_value').textContent = config.listen_timeout || 3;
                        document.getElementById('language').value = config.language || 'es-ES';
                        document.getElementById('dynamic_energy_threshold').checked = config.dynamic_energy_threshold || false;
                        
                        // VAD fields
                        const vadEnabledEl = document.getElementById('vad_enabled');
                        const vadAggEl = document.getElementById('vad_aggressiveness');
                        const vadPadEl = document.getElementById('vad_padding_ms');
                        const vadAggValEl = document.getElementById('vad_agg_value');
                        const vadPadValEl = document.getElementById('vad_padding_value');
                        
                        if (vadEnabledEl) vadEnabledEl.checked = !!config.vad_enabled;
                        if (vadAggEl) { 
                            vadAggEl.value = (config.vad_aggressiveness ?? 2); 
                            if (vadAggValEl) vadAggValEl.textContent = vadAggEl.value; 
                        }
                        if (vadPadEl) { 
                            vadPadEl.value = (config.vad_padding_ms ?? 250); 
                            if (vadPadValEl) vadPadValEl.textContent = vadPadEl.value; 
                        }
                        
                        debugLog('✅ Configuración de audio actualizada en UI');
                    }
                })
                .catch(error => {
                    debugLog('❌ Error cargando configuración: ' + error.message);
                });
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
                    loadEnginesWorking(); // Recargar para actualizar UI
                    
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
            
            // Handler de cambio de motor de sentimiento
            socket.on('sentiment_engine_changed', function(data) {
                debugLog('[SENTIMENT] Motor de sentimiento cambiado: ' + JSON.stringify(data));
                if (data.success) {
                    showNotification('Motor de sentimiento cambiado a ' + data.engine_name, 'success');
                    loadSentimentEngines(); // Recargar para actualizar UI
                }
            });
            
            // Handler de métricas de sentimiento
            socket.on('sentiment_metrics', function(data) {
                debugLog('[SENTIMENT] Métricas recibidas: ' + JSON.stringify(data));
                
                const metricsDiv = document.getElementById('sentiment-metrics');
                if (metricsDiv && data.metrics && data.metrics.count > 0) {
                    const metrics = data.metrics;
                    const fraudRisk = data.fraud_risk_score || 0;
                    
                    let trendIcon = '→';
                    let trendColor = 'text-secondary';
                    if (metrics.recent_trend === 'improving') {
                        trendIcon = '↗';
                        trendColor = 'text-success';
                    } else if (metrics.recent_trend === 'declining') {
                        trendIcon = '↘';
                        trendColor = 'text-danger';
                    }
                    
                    metricsDiv.innerHTML = `
                        <hr>
                        <h6>Métricas de Conversación</h6>
                        <small>
                            <div class="row">
                                <div class="col-6">
                                    <strong>Total:</strong> ${metrics.count}<br>
                                    <strong>Promedio:</strong> ${metrics.mean_sentiment.toFixed(3)}<br>
                                    <strong>Volatilidad:</strong> ${metrics.volatility.toFixed(3)}
                                </div>
                                <div class="col-6">
                                    <strong>Positivo:</strong> ${(metrics.pos_ratio * 100).toFixed(0)}%<br>
                                    <strong>Negativo:</strong> ${(metrics.neg_ratio * 100).toFixed(0)}%<br>
                                    <strong>Neutral:</strong> ${(metrics.neu_ratio * 100).toFixed(0)}%
                                </div>
                            </div>
                            <div class="mt-2">
                                <strong>Tendencia:</strong> <span class="${trendColor}">${trendIcon} ${metrics.recent_trend}</span><br>
                                <strong>Riesgo Fraude:</strong> <span class="${fraudRisk > 0.7 ? 'text-danger' : fraudRisk > 0.4 ? 'text-warning' : 'text-success'}">${(fraudRisk * 100).toFixed(1)}%</span>
                            </div>
                        </small>
                    `;
                }
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
                
                // Determinar clase de borde según nivel de riesgo del vishing
                let fraudClass = 'border-success bg-success bg-opacity-10';
                let badgeHTML = '<span class="badge bg-success">✅ NORMAL</span>';
                let probabilityText = '';
                
                if (data.vishing_analysis) {
                    const vishing = data.vishing_analysis;
                    probabilityText = `${vishing.percentage}%`;
                    
                    if (vishing.is_vishing) {
                        if (vishing.risk_level === 'CRÍTICO') {
                            fraudClass = 'border-danger bg-danger bg-opacity-25';
                            badgeHTML = `<span class="badge bg-danger">🚨 CRÍTICO</span>`;
                        } else if (vishing.risk_level === 'ALTO') {
                            fraudClass = 'border-danger bg-danger bg-opacity-10';
                            badgeHTML = `<span class="badge bg-danger">⚠️ ALTO</span>`;
                        } else if (vishing.risk_level === 'MEDIO') {
                            fraudClass = 'border-warning bg-warning bg-opacity-10';
                            badgeHTML = `<span class="badge bg-warning text-dark">⚠️ MEDIO</span>`;
                        }
                    } else if (vishing.risk_level === 'BAJO') {
                        fraudClass = 'border-info bg-info bg-opacity-10';
                        badgeHTML = `<span class="badge bg-info">ℹ️ BAJO</span>`;
                    }
                } else if (data.fraud_analysis) {
                    // Fallback al sistema antiguo
                    probabilityText = `${data.fraud_analysis.probability}%`;
                    if (data.fraud_analysis.is_fraud) {
                        fraudClass = 'border-danger bg-danger bg-opacity-10';
                        badgeHTML = `<span class="badge bg-danger">⚠️ FRAUDE</span>`;
                    }
                }
                
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
                            ${badgeHTML}<br><small>${probabilityText}</small>
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
                
                // Actualizar info de fraude (ahora con sistema multinivel de vishing)
                const fraudInfo = document.getElementById('fraud-info');
                if (fraudInfo && data.vishing_analysis) {
                    const vishing = data.vishing_analysis;
                    
                    // Determinar color según nivel de riesgo
                    let alertClass = 'alert-success';
                    let icon = '✅';
                    
                    if (vishing.risk_level === 'CRÍTICO') {
                        alertClass = 'alert-danger';
                        icon = '🚨';
                    } else if (vishing.risk_level === 'ALTO') {
                        alertClass = 'alert-danger';
                        icon = '⚠️';
                    } else if (vishing.risk_level === 'MEDIO') {
                        alertClass = 'alert-warning';
                        icon = '⚠️';
                    } else if (vishing.risk_level === 'BAJO') {
                        alertClass = 'alert-info';
                        icon = 'ℹ️';
                    }
                    
                    if (vishing.is_vishing) {
                        // MEJORA 6: Mostrar clasificación adaptativa
                        let adaptiveHTML = '';
                        if (vishing.adaptive) {
                            const contextIcons = {
                                'high_security': '🔒',
                                'medium_security': '🔐',
                                'low_security': '🔓'
                            };
                            const contextIcon = contextIcons[vishing.adaptive.security_context] || '🔐';
                            const thresholdPercent = (vishing.adaptive.threshold_used * 100).toFixed(0);
                            
                            adaptiveHTML = `
                                <div class="alert alert-info mt-2" style="padding: 8px; font-size: 13px;">
                                    <strong>📊 Clasificación Adaptativa:</strong><br>
                                    ${contextIcon} Contexto: <strong>${vishing.adaptive.security_context.replace('_', ' ').toUpperCase()}</strong><br>
                                    📈 Clasificación: <strong>${vishing.adaptive.classification}</strong> (threshold: ${thresholdPercent}%)<br>
                                    <small>El sistema ajustó el umbral según el contexto de seguridad detectado</small>
                                </div>
                            `;
                        }
                        
                        // Construir desglose de contribuciones
                        let breakdownHTML = '';
                        if (vishing.breakdown) {
                            breakdownHTML = '<small><strong>Desglose:</strong><br>';
                            for (const [feature, data] of Object.entries(vishing.breakdown)) {
                                const featureNames = {
                                    'keywords': 'Keywords',
                                    'ml_model': 'ML Model',
                                    'sentiment': 'Sentimiento',
                                    'linguistic': 'Lingüística',
                                    'temporal': 'Temporal',
                                    'acoustic': 'Acústica'
                                };
                                const displayName = featureNames[feature] || feature;
                                const value = (data.value * 100).toFixed(1);
                                const contrib = (data.contribution * 100).toFixed(1);
                                breakdownHTML += `• ${displayName}: ${value}% (→ ${contrib}%)<br>`;
                            }
                            breakdownHTML += '</small>';
                        }
                        
                        // Añadir información de keywords contextuales si está disponible
                        let keywordCategoriesHTML = '';
                        if (data.fraud_analysis && data.fraud_analysis.keyword_analysis && 
                            data.fraud_analysis.keyword_analysis.categories) {
                            const kwAnalysis = data.fraud_analysis.keyword_analysis;
                            if (kwAnalysis.category_count > 0) {
                                keywordCategoriesHTML = '<small><strong>Categorías detectadas:</strong><br>';
                                for (const [catName, catData] of Object.entries(kwAnalysis.categories)) {
                                    const categoryNames = {
                                        'bancarias': '🏦 Bancarias',
                                        'urgencia': '⏰ Urgencia',
                                        'suplantacion': '🎭 Suplantación',
                                        'datos_sensibles': '🔐 Datos Sensibles',
                                        'amenazas': '⚠️ Amenazas',
                                        'financieras': '💰 Financieras',
                                        'verificacion': '✓ Verificación'
                                    };
                                    const displayName = categoryNames[catName] || catName;
                                    keywordCategoriesHTML += `• ${displayName}: ${catData.count} palabras<br>`;
                                }
                                keywordCategoriesHTML += '</small>';
                            }
                        }
                        
                        // Añadir información de análisis lingüístico (MEJORA 3)
                        let linguisticHTML = '';
                        if (data.linguistic_analysis && data.linguistic_analysis.pattern_count > 0) {
                            const ling = data.linguistic_analysis;
                            linguisticHTML = '<small><strong>Patrones lingüísticos:</strong><br>';
                            
                            // Mostrar flags principales (máximo 5)
                            const flagNames = {
                                'imperative': '⚡ Imperativos',
                                'data_request': '🔑 Solicitud de datos',
                                'time_pressure': '⏱️ Presión temporal',
                                'authority': '👔 Falsa autoridad',
                                'legal_threats': '⚖️ Amenazas legales',
                                'suspicious_offers': '🎁 Ofertas sospechosas',
                                'questioning': '❓ Interrogatorio',
                                'action_request': '👆 Solicitud de acción',
                                'risk_negation': '🛡️ Tranquilización',
                                'length_anomaly': '📏 Longitud anormal'
                            };
                            
                            ling.flags.slice(0, 5).forEach(flag => {
                                const displayName = flagNames[flag] || flag;
                                linguisticHTML += `• ${displayName}<br>`;
                            });
                            
                            if (ling.flags.length > 5) {
                                linguisticHTML += `• ... y ${ling.flags.length - 5} más<br>`;
                            }
                            
                            linguisticHTML += '</small>';
                        }
                        
                        // Añadir información de análisis temporal (MEJORA 4)
                        let temporalHTML = '';
                        if (data.temporal_analysis && data.temporal_analysis.pattern_count > 0) {
                            const temp = data.temporal_analysis;
                            temporalHTML = '<small><strong>Patrones temporales:</strong><br>';
                            
                            // Mostrar flags principales (máximo 5)
                            const tempFlagNames = {
                                'ESCALADA_URGENCIA': '📈 Escalada de urgencia',
                                'MONOPOLIO_CONVERSACION': '🗣️ Monopolio conversación',
                                'TURNO_MUY_LARGO': '📝 Turno muy largo',
                                'EXCESO_PREGUNTAS': '❓ Exceso de preguntas',
                                'REPETICION_ALTA': '🔄 Alta repetición',
                                'INSISTENCIA_DATOS': '🔐 Insistencia en datos',
                                'RITMO_ACELERADO': '⚡ Ritmo acelerado',
                                'CONVERSACION_LARGA': '⏳ Conversación larga'
                            };
                            
                            temp.flags.slice(0, 5).forEach(flag => {
                                const displayName = tempFlagNames[flag] || flag;
                                temporalHTML += `• ${displayName}<br>`;
                            });
                            
                            if (temp.flags.length > 5) {
                                temporalHTML += `• ... y ${temp.flags.length - 5} más<br>`;
                            }
                            
                            // Agregar métrica de turnos
                            if (temp.turn_count > 1) {
                                temporalHTML += `<em>(${temp.turn_count} turnos analizados)</em><br>`;
                            }
                            
                            temporalHTML += '</small>';
                        }
                        
                        // Añadir información de incongruencias (MEJORA 5)
                        let incongruenceHTML = '';
                        if (data.incongruence_analysis && data.incongruence_analysis.incongruence_count > 0) {
                            const incong = data.incongruence_analysis;
                            incongruenceHTML = '<small><strong>Incongruencias:</strong><br>';
                            
                            // Mostrar flags principales (máximo 5)
                            const incongFlagNames = {
                                'AMABILIDAD_CON_URGENCIA': '⚠️ Amabilidad + Urgencia',
                                'SENTIMIENTO_POSITIVO_CON_AMENAZAS': '🔴 Positivo + Amenazas',
                                'SOLICITUD_DATOS_CON_TRANQUILIZACIÓN': '🚨 Datos + Tranquilización',
                                'AMENAZA_CON_TRANQUILIZACIÓN': '⚡ Amenaza + Calma',
                                'BENEFICIO_CON_URGENCIA': '🎁 Beneficio + Urgencia',
                                'AMABILIDAD_SOLICITANDO_DATOS': '😊 Amabilidad + Datos',
                                'NEGATIVO_CON_TRANQUILIZACIÓN': '😟 Negativo + Calma'
                            };
                            
                            incong.flags.slice(0, 5).forEach(flag => {
                                const displayName = incongFlagNames[flag] || flag;
                                incongruenceHTML += `• ${displayName}<br>`;
                            });
                            
                            if (incong.flags.length > 5) {
                                incongruenceHTML += `• ... y ${incong.flags.length - 5} más<br>`;
                            }
                            
                            incongruenceHTML += '</small>';
                        }
                        
                        // Añadir información de análisis acústico (MEJORA 7)
                        let acousticHTML = '';
                        if (data.acoustic_analysis && data.acoustic_analysis.analysis_success && 
                            data.acoustic_analysis.flag_count > 0) {
                            const acoustic = data.acoustic_analysis;
                            acousticHTML = '<small><strong>🎤 Acústica:</strong><br>';
                            
                            // Mostrar flags acústicas principales (máximo 4)
                            const acousticFlagNames = {
                                'VELOCIDAD_EXCESIVA': '⚡ Velocidad excesiva',
                                'VELOCIDAD_MUY_LENTA': '🐌 Velocidad muy lenta',
                                'HABLA_ROBOTICA': '🤖 Habla robótica',
                                'FRICCION_VOCAL_ALTA': '😰 Fricción vocal alta',
                                'PAUSAS_MINIMAS': '💨 Pausas mínimas',
                                'PAUSAS_EXCESIVAS': '⏸️ Pausas excesivas',
                                'VOZ_MONOTONA': '😑 Voz monótona',
                                'SEGMENTO_UNICO': '📢 Segmento único'
                            };
                            
                            acoustic.flags.slice(0, 4).forEach(flag => {
                                const displayName = acousticFlagNames[flag] || flag;
                                acousticHTML += `• ${displayName}<br>`;
                            });
                            
                            if (acoustic.flags.length > 4) {
                                acousticHTML += `• ... y ${acoustic.flags.length - 4} más<br>`;
                            }
                            
                            // Mostrar features acústicas clave si están disponibles
                            if (acoustic.features) {
                                const feats = acoustic.features;
                                if (feats.speaking_rate && feats.speaking_rate > 0) {
                                    acousticHTML += `<em>(${feats.speaking_rate.toFixed(1)} pal/seg, `;
                                    acousticHTML += `${(feats.silence_ratio * 100).toFixed(0)}% pausas)</em><br>`;
                                }
                            }
                            
                            acousticHTML += '</small>';
                        }
                        
                        // Añadir Dashboard de Explicabilidad (MEJORA 8)
                        let explanationHTML = '';
                        if (data.explanation) {
                            const exp = data.explanation;
                            
                            // Construir HTML del dashboard de explicación
                            explanationHTML = '<div class="mt-3 border-top pt-2">';
                            explanationHTML += '<h6>📊 Dashboard de Explicabilidad</h6>';
                            
                            // 1. RESUMEN EJECUTIVO
                            explanationHTML += `<div class="alert alert-${exp.risk_level === 'CRÍTICO' ? 'danger' : exp.risk_level === 'ALTO' ? 'warning' : exp.risk_level === 'MEDIO' ? 'info' : 'success'} p-2 mb-2">`;
                            explanationHTML += `<small>${exp.summary}</small>`;
                            explanationHTML += '</div>';
                            
                            // 2. EVIDENCIA DETECTADA
                            if (exp.evidence && exp.evidence.length > 0) {
                                explanationHTML += '<div class="mb-2"><small><strong>🔍 Evidencia Detectada:</strong></small><ul class="mb-1">';
                                exp.evidence.forEach((ev, idx) => {
                                    const severityBadge = ev.severity === 'ALTA' ? 'danger' : ev.severity === 'MEDIA' ? 'warning' : 'secondary';
                                    explanationHTML += '<li style="font-size: 0.85rem;">';
                                    explanationHTML += `${ev.icon} <span class="badge badge-${severityBadge}">${ev.type}</span> `;
                                    explanationHTML += `<em>(${ev.score_contribution.toFixed(1)}%)</em><br>`;
                                    explanationHTML += `<small class="text-muted">${ev.detail}</small>`;
                                    explanationHTML += '</li>';
                                });
                                explanationHTML += '</ul></div>';
                            }
                            
                            // 3. RECOMENDACIONES ACCIONABLES
                            if (exp.recommendations && exp.recommendations.length > 0) {
                                explanationHTML += '<div class="mb-2"><small><strong>💡 Recomendaciones:</strong></small><ol class="mb-1">';
                                exp.recommendations.slice(0, 3).forEach(rec => {
                                    const priorityColor = rec.priority === 'CRÍTICA' ? 'danger' : rec.priority === 'ALTA' ? 'warning' : rec.priority === 'MEDIA' ? 'info' : 'secondary';
                                    explanationHTML += '<li style="font-size: 0.85rem;">';
                                    explanationHTML += `${rec.icon} <span class="badge badge-${priorityColor}">${rec.priority}</span> `;
                                    explanationHTML += `<strong>${rec.action}</strong><br>`;
                                    explanationHTML += `<small class="text-muted">→ ${rec.reason}</small>`;
                                    explanationHTML += '</li>';
                                });
                                if (exp.recommendations.length > 3) {
                                    explanationHTML += `<li><small class="text-muted">... y ${exp.recommendations.length - 3} recomendaciones más</small></li>`;
                                }
                                explanationHTML += '</ol></div>';
                            }
                            
                            explanationHTML += '</div>';
                        }
                        
                        fraudInfo.innerHTML = `
                            <div class="alert ${alertClass} p-2 mb-2">
                                <strong>${icon} VISHING DETECTADO - ${vishing.risk_level}</strong><br>
                                <small>Score: ${vishing.percentage}%</small><br>
                                <small>Confianza: ${(vishing.confidence * 100).toFixed(0)}%</small><br>
                                ${breakdownHTML}
                                ${keywordCategoriesHTML}
                                ${linguisticHTML}
                                ${temporalHTML}
                                ${incongruenceHTML}
                                ${acousticHTML}
                                ${data.fraud_analysis.keywords_found && data.fraud_analysis.keywords_found.length > 0 ? 
                                    `<small><strong>Keywords:</strong> ${data.fraud_analysis.keywords_found.join(', ')}</small>` : ''}
                            </div>
                            ${adaptiveHTML}
                            ${explanationHTML}
                        `;
                    } else {
                        fraudInfo.innerHTML = `
                            <div class="alert ${alertClass} p-2 mb-2">
                                <strong>${icon} ${vishing.risk_level === 'NORMAL' ? 'Texto Normal' : 'Riesgo ' + vishing.risk_level}</strong><br>
                                <small>Score: ${vishing.percentage}%</small><br>
                                <small>Confianza: ${(vishing.confidence * 100).toFixed(0)}%</small>
                            </div>
                        `;
                    }
                } else if (fraudInfo && data.is_fraud !== undefined) {
                    // Fallback al sistema antiguo si no hay vishing_analysis
                    if (data.is_fraud) {
                        fraudInfo.innerHTML = `
                            <div class="alert alert-danger p-2 mb-2">
                                <strong>⚠️ FRAUDE DETECTADO</strong><br>
                                <small>Probabilidad combinada: ${data.combined_fraud_probability.toFixed(1)}%</small><br>
                                <small>Keywords: ${data.fraud_analysis.probability}%</small><br>
                                <small>Sentimiento: ${(data.fraud_risk_score * 100).toFixed(1)}%</small>
                                ${data.fraud_analysis.keywords_found && data.fraud_analysis.keywords_found.length > 0 ? 
                                    `<br><small>Palabras clave: ${data.fraud_analysis.keywords_found.join(', ')}</small>` : ''}
                            </div>
                        `;
                    } else {
                        fraudInfo.innerHTML = `
                            <div class="alert alert-success p-2 mb-2">
                                <strong>✅ Texto Normal</strong><br>
                                <small>Riesgo combinado: ${data.combined_fraud_probability.toFixed(1)}%</small>
                            </div>
                        `;
                    }
                }
                
                // Actualizar análisis de sentimiento
                const sentimentInfo = document.getElementById('sentiment-info');
                if (sentimentInfo && data.sentiment_analysis) {
                    const sentiment = data.sentiment_analysis;
                    let sentimentColor = 'text-success';
                    let sentimentIcon = '😊';
                    
                    if (sentiment.label === 'NEG') {
                        sentimentColor = 'text-danger';
                        sentimentIcon = '😠';
                    } else if (sentiment.label === 'NEU') {
                        sentimentColor = 'text-secondary';
                        sentimentIcon = '😐';
                    }
                    
                    let emotionsHtml = '';
                    if (sentiment.emotions && Object.keys(sentiment.emotions).length > 0) {
                        emotionsHtml = '<div class="mt-2 small">';
                        for (const [emotion, score] of Object.entries(sentiment.emotions)) {
                            if (score > 0.1) {
                                emotionsHtml += `<span class="badge bg-secondary me-1">${emotion}: ${(score * 100).toFixed(0)}%</span>`;
                            }
                        }
                        emotionsHtml += '</div>';
                    }
                    
                    sentimentInfo.innerHTML = `
                        <div class="p-2">
                            <div class="${sentimentColor}">
                                <strong>${sentimentIcon} ${sentiment.label}</strong>
                                <span class="ms-2">Score: ${sentiment.compound.toFixed(3)}</span>
                            </div>
                            <small class="text-muted">Motor: ${data.sentiment_engine || 'N/A'}</small>
                            ${emotionsHtml}
                        </div>
                    `;
                }
                
                // Solicitar métricas de sentimiento actualizadas
                if (socket && socket.emit) {
                    socket.emit('request_sentiment_metrics');
                }
            });
            
        } catch (error) {
            console.error('❌ Error inicializando Socket.IO: ' + error.message);
        }

        // Cargar datos iniciales
        document.addEventListener('DOMContentLoaded', function() {
            debugLog('🚀 DOMContentLoaded ejecutándose...');
            
            // Inicializar tooltips de Bootstrap (si está disponible)
            try {
                if (typeof bootstrap !== 'undefined') {
                    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
                    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                        return new bootstrap.Tooltip(tooltipTriggerEl);
                    });
                    console.log('✅ Bootstrap tooltips inicializados');
                } else {
                    console.log('⚠️ Bootstrap no disponible, omitiendo tooltips');
                }
            } catch (error) {
                console.log('⚠️ Error inicializando tooltips:', error.message);
            }
            
            // Cargar engines con delay para asegurar que la página esté lista
            setTimeout(function() {
                debugLog('🚀 Iniciando carga de engines...');
                console.log('🔵 [SENTIMENT] ¿Función existe?', typeof loadSentimentEngines);
                loadEnginesWorking();
                console.log('🔵 [SENTIMENT] Llamando a loadSentimentEngines...');
                loadSentimentEngines(); // Cargar motores de sentimiento
                console.log('🔵 [SENTIMENT] loadSentimentEngines llamada');
            }, 1000);
            
            // Cargar configuración de audio actual usando la nueva función
            loadCurrentAudioConfig();
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
                    
                    // Actualizar el indicador de motor actual
                    const currentEngineElement = document.getElementById('current-engine');
                    if (currentEngineElement && data.current_engine) {
                        currentEngineElement.textContent = data.current_engine.display_name || data.current_engine.engine_id || 'Ninguno';
                    }
                    
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

@app.route('/api/audio_profiles')
def get_audio_profiles():
    """API para obtener perfiles de audio disponibles"""
    try:
        return jsonify({
            'success': True,
            'profiles': transcriptor.get_audio_profiles()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load_audio_profile', methods=['POST'])
def load_audio_profile():
    """API para cargar un perfil de audio"""
    try:
        data = request.get_json()
        profile_name = data.get('profile_name')
        
        if not profile_name:
            return jsonify({'success': False, 'error': 'profile_name requerido'})
        
        success = transcriptor.load_audio_profile(profile_name)
        return jsonify({
            'success': success,
            'message': f'Perfil {profile_name} cargado' if success else 'Error cargando perfil'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sentiment_engines')
def get_sentiment_engines():
    """API para obtener motores de sentimiento disponibles"""
    try:
        print("[API] Obteniendo motores de sentimiento...")
        engines = []
        for engine_id, engine_obj in transcriptor.sentiment_manager.engines.items():
            engine_data = {
                'id': engine_id,
                'name': engine_obj.display_name,
                'description': engine_obj.description,
                'model_size': engine_obj.model_size,
                'speed': engine_obj.speed,
                'is_active': engine_id == transcriptor.sentiment_manager.current_engine_name
            }
            print(f"[API] Motor: {engine_id} -> {engine_data}")
            engines.append(engine_data)
        
        response_data = {
            'success': True,
            'engines': engines,
            'current_engine': transcriptor.sentiment_manager.current_engine_name
        }
        print(f"[API] Devolviendo {len(engines)} motores")
        print(f"[API] Response: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        error_msg = str(e)
        print(f"[API ERROR] Error en sentiment_engines: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/vishing_scorer_info')
def get_vishing_scorer_info():
    """API para obtener información del sistema de scoring de vishing"""
    try:
        print("[API] Obteniendo información del VishingScorer...")
        
        scorer_info = {
            'success': True,
            'weights': transcriptor.vishing_scorer.weights,
            'thresholds': transcriptor.vishing_scorer.thresholds,
            'description': {
                'keywords': 'Detección de palabras clave sospechosas',
                'ml_model': 'Modelo de Machine Learning (LogisticRegression + TF-IDF)',
                'sentiment': 'Análisis de sentimiento conversacional',
                'linguistic': 'Patrones lingüísticos (MEJORA 3 - ACTIVA)',
                'temporal': 'Análisis temporal de conversación (MEJORA 4 - ACTIVA)',
                'acoustic': 'Características acústicas del audio (futuro)'
            },
            'risk_levels': {
                'CRÍTICO': f'>= {transcriptor.vishing_scorer.thresholds["critical"]*100}%',
                'ALTO': f'>= {transcriptor.vishing_scorer.thresholds["high"]*100}%',
                'MEDIO': f'>= {transcriptor.vishing_scorer.thresholds["medium"]*100}%',
                'BAJO': f'>= {transcriptor.vishing_scorer.thresholds["low"]*100}%',
                'NORMAL': f'< {transcriptor.vishing_scorer.thresholds["low"]*100}%'
            }
        }
        
        print(f"[API] VishingScorer info: {scorer_info}")
        return jsonify(scorer_info)
    except Exception as e:
        error_msg = str(e)
        print(f"[API ERROR] Error en vishing_scorer_info: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/linguistic_analyzer_info')
def get_linguistic_analyzer_info():
    """API para obtener información del analizador lingüístico (MEJORA 3)"""
    try:
        print("[API] Obteniendo información del LinguisticAnalyzer...")
        
        pattern_info = transcriptor.linguistic_analyzer.get_pattern_info()
        
        analyzer_info = {
            'success': True,
            'patterns': pattern_info,
            'pattern_count': len(pattern_info),
            'description': 'Detecta patrones lingüísticos sospechosos más allá de keywords específicas',
            'features': [
                'Comandos imperativos (debe, necesita, confirme)',
                'Solicitudes de datos personales',
                'Presión temporal (24 horas, urgente)',
                'Negaciones de riesgo (100% seguro, sin problema)',
                'Apelación a autoridad (policía, ministerio)',
                'Amenazas legales (multa, demanda)',
                'Ofertas sospechosas (ha ganado, premio)',
                'Solicitud de acción (haga clic, descargue)',
                'Longitud anormal del texto',
                'Exceso de preguntas'
            ]
        }
        
        print(f"[API] LinguisticAnalyzer info: {analyzer_info['pattern_count']} patrones disponibles")
        return jsonify(analyzer_info)
    except Exception as e:
        error_msg = str(e)
        print(f"[API ERROR] Error en linguistic_analyzer_info: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/conversation_analyzer_info')
def get_conversation_analyzer_info():
    """API para obtener información del analizador de conversación (MEJORA 4)"""
    try:
        print("[API] Obteniendo información del ConversationAnalyzer...")
        
        summary = transcriptor.conversation_analyzer.get_conversation_summary()
        current_analysis = transcriptor.conversation_analyzer.analyze_patterns()
        
        analyzer_info = {
            'success': True,
            'window_size': transcriptor.conversation_analyzer.window_size,
            'current_summary': summary,
            'current_analysis': {
                'total_score': current_analysis['total_score'],
                'risk_level': current_analysis['risk_level'],
                'pattern_count': current_analysis['pattern_count'],
                'turn_count': current_analysis['turn_count'],
                'flags': current_analysis['flags']
            },
            'description': 'Analiza patrones temporales y comportamentales en el flujo de conversación',
            'patterns_detected': [
                'Escalada de urgencia (aumento de palabras de presión)',
                'Monopolización de conversación (turnos muy largos)',
                'Exceso de preguntas (interrogatorio)',
                'Alta repetición (insistencia en mismas solicitudes)',
                'Insistencia en datos (solicitudes constantes de información)',
                'Ritmo acelerado (poco tiempo entre turnos)',
                'Conversación larga (muchos turnos)',
                'Turno muy largo (scripts preparados)'
            ]
        }
        
        print(f"[API] ConversationAnalyzer info: {current_analysis['turn_count']} turnos, "
              f"{current_analysis['pattern_count']} patrones detectados")
        return jsonify(analyzer_info)
    except Exception as e:
        error_msg = str(e)
        print(f"[API ERROR] Error en conversation_analyzer_info: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/clear_conversation', methods=['POST'])
def clear_conversation():
    """API para limpiar el historial de conversación"""
    try:
        print("[API] Limpiando historial de conversación...")
        transcriptor.conversation_analyzer.clear_history()
        return jsonify({
            'success': True,
            'message': 'Historial de conversación limpiado'
        })
    except Exception as e:
        error_msg = str(e)
        print(f"[API ERROR] Error en clear_conversation: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/incongruence_detector_info')
def get_incongruence_detector_info():
    """API para obtener información del detector de incongruencias (MEJORA 5)"""
    try:
        print("[API] Obteniendo información del IncongruenceDetector...")
        
        detector_info = {
            'success': True,
            'description': 'Detecta contradicciones y señales mixtas sospechosas en el texto',
            'incongruence_types': [
                {
                    'name': 'Amabilidad con Urgencia',
                    'flag': 'AMABILIDAD_CON_URGENCIA',
                    'severity': 0.85,
                    'description': 'Cortesía excesiva combinada con presión temporal'
                },
                {
                    'name': 'Sentimiento Positivo con Amenazas',
                    'flag': 'SENTIMIENTO_POSITIVO_CON_AMENAZAS',
                    'severity': 0.9,
                    'description': 'Tono positivo al comunicar problemas graves'
                },
                {
                    'name': 'Solicitud de Datos con Tranquilización',
                    'flag': 'SOLICITUD_DATOS_CON_TRANQUILIZACIÓN',
                    'severity': 1.0,
                    'description': 'Pide datos sensibles mientras intenta calmar'
                },
                {
                    'name': 'Amenaza con Tranquilización',
                    'flag': 'AMENAZA_CON_TRANQUILIZACIÓN',
                    'severity': 0.8,
                    'description': 'Presenta amenazas pero intenta tranquilizar'
                },
                {
                    'name': 'Beneficio con Urgencia',
                    'flag': 'BENEFICIO_CON_URGENCIA',
                    'severity': 0.75,
                    'description': 'Ofrece beneficios pero presiona para actuar rápido'
                },
                {
                    'name': 'Amabilidad Solicitando Datos',
                    'flag': 'AMABILIDAD_SOLICITANDO_DATOS',
                    'severity': 0.8,
                    'description': 'Amabilidad excesiva al pedir información sensible'
                },
                {
                    'name': 'Negativo con Tranquilización',
                    'flag': 'NEGATIVO_CON_TRANQUILIZACIÓN',
                    'severity': 0.7,
                    'description': 'Tono negativo pero intenta tranquilizar'
                }
            ],
            'word_categories': {
                'politeness': len(transcriptor.incongruence_detector.politeness_words),
                'urgency': len(transcriptor.incongruence_detector.urgency_words),
                'threats': len(transcriptor.incongruence_detector.threat_words),
                'data_requests': len(transcriptor.incongruence_detector.data_request_words),
                'reassurance': len(transcriptor.incongruence_detector.reassurance_words),
                'benefits': len(transcriptor.incongruence_detector.benefit_words)
            }
        }
        
        print(f"[API] IncongruenceDetector info: 7 tipos de incongruencias, "
              f"{sum(detector_info['word_categories'].values())} palabras clave")
        return jsonify(detector_info)
    except Exception as e:
        error_msg = str(e)
        print(f"[API ERROR] Error en incongruence_detector_info: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/adaptive_threshold_info')
def get_adaptive_threshold_info():
    """API para obtener información del sistema de thresholds adaptativos (MEJORA 6)"""
    try:
        print("[API] Obteniendo información del AdaptiveThreshold...")
        
        stats = transcriptor.adaptive_threshold.get_stats()
        
        threshold_info = {
            'success': True,
            'description': 'Sistema de thresholds dinámicos con calibración automática según contexto',
            'security_contexts': [
                {
                    'name': 'high_security',
                    'display_name': 'Alta Seguridad',
                    'icon': '🔒',
                    'description': 'Para temas sensibles (banca, datos personales)',
                    'thresholds': stats['thresholds']['high_security'],
                    'use_cases': ['Transacciones bancarias', 'Datos sensibles', 'Información confidencial']
                },
                {
                    'name': 'medium_security',
                    'display_name': 'Seguridad Media',
                    'icon': '🔐',
                    'description': 'Para conversaciones generales',
                    'thresholds': stats['thresholds']['medium_security'],
                    'use_cases': ['Conversación general', 'Atención al cliente', 'Soporte técnico']
                },
                {
                    'name': 'low_security',
                    'display_name': 'Baja Seguridad',
                    'icon': '🔓',
                    'description': 'Para conversaciones casuales de bajo riesgo',
                    'thresholds': stats['thresholds']['low_security'],
                    'use_cases': ['Conversación casual', 'Amigos/familia', 'Temas cotidianos']
                }
            ],
            'calibration_stats': stats['calibration_stats'],
            'performance_log_size': stats['performance_log_size'],
            'auto_calibration': {
                'enabled': stats['auto_calibration_enabled'],
                'interval': transcriptor.adaptive_threshold.calibration_interval,
                'max_adjustment': transcriptor.adaptive_threshold.max_threshold_adjustment
            },
            'features': [
                '🎯 Detección automática de contexto de seguridad',
                '📊 3 perfiles de thresholds (alto/medio/bajo)',
                '🔄 Auto-calibración cada 100 casos',
                '📈 Ajuste basado en precision/recall',
                '🎚️ Máximo ajuste: ±10% por calibración',
                '📝 Logging de 1000 casos más recientes'
            ]
        }
        
        # Calcular métricas si hay datos
        if stats['calibration_stats']['total_predictions'] > 0:
            cs = stats['calibration_stats']
            tp = cs['true_positives']
            fp = cs['false_positives']
            fn = cs['false_negatives']
            tn = cs['true_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
            
            threshold_info['performance_metrics'] = {
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1_score': round(f1, 3),
                'accuracy': round(accuracy, 3)
            }
        
        print(f"[API] AdaptiveThreshold info: 3 contextos, "
              f"{stats['calibration_stats']['total_predictions']} predicciones, "
              f"{stats['calibration_stats']['calibration_count']} calibraciones")
        
        return jsonify(threshold_info)
    except Exception as e:
        error_msg = str(e)
        print(f"[API ERROR] Error en adaptive_threshold_info: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/acoustic_analyzer_info')
def get_acoustic_analyzer_info():
    """API para obtener información del analizador acústico (MEJORA 7)"""
    try:
        print("[API] Obteniendo información del AcousticAnalyzer...")
        
        analyzer_info = {
            'success': True,
            'description': 'Análisis de características acústicas del audio para detectar patrones de vishing',
            'features': [
                {
                    'name': 'speaking_rate',
                    'display_name': 'Velocidad de Habla',
                    'description': 'Palabras por segundo - Detecta habla apresurada o muy lenta',
                    'threshold': 'Normal: 2.0-4.5 pal/seg',
                    'suspicious': 'Rápida: >4.5 pal/seg | Lenta: <2.0 pal/seg'
                },
                {
                    'name': 'energy',
                    'display_name': 'Energía del Audio',
                    'description': 'Volumen promedio del audio normalizado',
                    'threshold': 'Variable según ambiente',
                    'suspicious': 'Muy uniforme indica voz sintética o script'
                },
                {
                    'name': 'zero_crossing_rate',
                    'display_name': 'Tasa de Cruce por Cero',
                    'description': 'Indica fricción vocal, urgencia o estrés',
                    'threshold': 'Normal: <0.15',
                    'suspicious': 'Alta: >0.15 (estrés vocal)'
                },
                {
                    'name': 'silence_ratio',
                    'display_name': 'Ratio de Silencios',
                    'description': 'Porcentaje de pausas/silencios en el audio',
                    'threshold': 'Normal: 10-40%',
                    'suspicious': 'Pocas: <10% | Muchas: >40%'
                },
                {
                    'name': 'peak_ratio',
                    'display_name': 'Ratio de Picos',
                    'description': 'Picos de energía que indican énfasis natural',
                    'threshold': 'Variable',
                    'suspicious': 'Muy pocos picos = voz monótona'
                },
                {
                    'name': 'speech_segments',
                    'display_name': 'Segmentos de Habla',
                    'description': 'Número de segmentos continuos de habla',
                    'threshold': '≥2 para audio >2 seg',
                    'suspicious': '1 segmento = habla sin pausas naturales'
                }
            ],
            'flags': [
                {
                    'name': 'VELOCIDAD_EXCESIVA',
                    'icon': '⚡',
                    'severity': 0.7,
                    'description': 'Habla muy rápida (>4.5 pal/seg) - Típico de scripts o urgencia artificial'
                },
                {
                    'name': 'VELOCIDAD_MUY_LENTA',
                    'icon': '🐌',
                    'severity': 0.4,
                    'description': 'Habla muy lenta (<2.0 pal/seg) - Posible lectura o dubitación'
                },
                {
                    'name': 'HABLA_ROBOTICA',
                    'icon': '🤖',
                    'severity': 0.8,
                    'description': 'Energía muy uniforme - Típico de voz sintética o lectura de script'
                },
                {
                    'name': 'FRICCION_VOCAL_ALTA',
                    'icon': '😰',
                    'severity': 0.6,
                    'description': 'Alta tasa de cruce por cero - Indica estrés o urgencia vocal'
                },
                {
                    'name': 'PAUSAS_MINIMAS',
                    'icon': '💨',
                    'severity': 0.7,
                    'description': 'Muy pocas pausas (<10%) - Habla apresurada sin respirar'
                },
                {
                    'name': 'PAUSAS_EXCESIVAS',
                    'icon': '⏸️',
                    'severity': 0.5,
                    'description': 'Muchas pausas (>40%) - Posible dubitación o nerviosismo'
                },
                {
                    'name': 'VOZ_MONOTONA',
                    'icon': '😑',
                    'severity': 0.6,
                    'description': 'Pocos picos de energía - Falta de énfasis natural'
                },
                {
                    'name': 'SEGMENTO_UNICO',
                    'icon': '📢',
                    'severity': 0.7,
                    'description': 'Habla continua sin pausas naturales - Típico de lectura de script'
                }
            ],
            'scoring_components': [
                {
                    'name': 'scripted_speech',
                    'weight': 0.35,
                    'description': 'Detección de habla leída o robótica (energía uniforme + pausas mínimas)'
                },
                {
                    'name': 'excessive_speed',
                    'weight': 0.25,
                    'description': 'Velocidad de habla anormal (muy rápida o muy lenta)'
                },
                {
                    'name': 'unnatural_pauses',
                    'weight': 0.20,
                    'description': 'Pausas sospechosas (muy pocas o excesivas)'
                },
                {
                    'name': 'energy_anomaly',
                    'weight': 0.20,
                    'description': 'Energía anómala (muy uniforme = robótico)'
                }
            ],
            'risk_levels': {
                'ALTO': '≥70% - Múltiples indicadores acústicos de vishing',
                'MEDIO': '50-69% - Algunos patrones acústicos sospechosos',
                'BAJO': '30-49% - Patrones acústicos levemente anómalos',
                'NORMAL': '<30% - Audio con características naturales'
            },
            'integration': {
                'weight_in_vishing_scorer': '10%',
                'combines_with': ['keywords', 'ml_model', 'sentiment', 'linguistic', 'temporal'],
                'sample_rate': '16000 Hz',
                'audio_format': 'WAV (int16)'
            }
        }
        
        print(f"[API] AcousticAnalyzer info: 6 features, 8 flags, 4 componentes de scoring")
        return jsonify(analyzer_info)
    except Exception as e:
        error_msg = str(e)
        print(f"[API ERROR] Error en acoustic_analyzer_info: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/explainable_detector_info')
def explainable_detector_info():
    """API para obtener información del ExplainableVishingDetector (MEJORA 8)"""
    try:
        detector_info = {
            'name': 'ExplainableVishingDetector',
            'version': '1.0.0',
            'description': 'Generador de explicaciones humanas para resultados de detección de vishing',
            'purpose': 'Convertir análisis técnicos en explicaciones claras y recomendaciones accionables',
            'mejora': 8,
            'status': 'ACTIVO',
            
            'evidence_types': [
                {
                    'type': 'KEYWORDS',
                    'icon': '🔑',
                    'severity': 'ALTA',
                    'description': 'Palabras clave sospechosas detectadas en categorías de fraude',
                    'source': 'VishingKeywords + ML Model'
                },
                {
                    'type': 'ML_MODEL',
                    'icon': '🤖',
                    'severity': 'ALTA',
                    'description': 'Probabilidad de fraude según modelo de Machine Learning',
                    'source': 'Logistic Regression (sklearn)'
                },
                {
                    'type': 'SENTIMENT',
                    'icon': '😰',
                    'severity': 'MEDIA',
                    'description': 'Análisis de sentimiento y emociones (miedo, negatividad)',
                    'source': 'SentimentManager (TextBlob/VADER/roBERTa)'
                },
                {
                    'type': 'LINGUISTIC',
                    'icon': '📝',
                    'severity': 'MEDIA',
                    'description': 'Patrones lingüísticos asociados con vishing',
                    'source': 'LinguisticAnalyzer'
                },
                {
                    'type': 'TEMPORAL',
                    'icon': '⏱️',
                    'severity': 'MEDIA',
                    'description': 'Anomalías en el flujo temporal de la conversación',
                    'source': 'ConversationAnalyzer'
                },
                {
                    'type': 'ACOUSTIC',
                    'icon': '🎤',
                    'severity': 'BAJA',
                    'description': 'Características acústicas sospechosas en el audio',
                    'source': 'AcousticAnalyzer'
                },
                {
                    'type': 'INCONGRUENCE',
                    'icon': '⚠️',
                    'severity': 'ALTA',
                    'description': 'Contradicciones e inconsistencias detectadas',
                    'source': 'IncongruenceDetector'
                }
            ],
            
            'recommendation_levels': [
                {
                    'score_range': '≥75%',
                    'classification': 'FRAUDE',
                    'priority': 'CRÍTICA',
                    'main_action': '🚨 TERMINAR LA LLAMADA INMEDIATAMENTE',
                    'description': 'Múltiples indicadores de vishing - Alto riesgo de fraude',
                    'recommendation_count': 5
                },
                {
                    'score_range': '60-74%',
                    'classification': 'SOSPECHOSO',
                    'priority': 'ALTA',
                    'main_action': '⚠️ Proceder con EXTREMA cautela',
                    'description': 'Patrones sospechosos detectados - Verificar identidad',
                    'recommendation_count': 5
                },
                {
                    'score_range': '45-59%',
                    'classification': 'MONITOREAR',
                    'priority': 'MEDIA',
                    'main_action': '🔍 Mantenerse alerta y escéptico',
                    'description': 'Algunas características sospechosas - Hacer preguntas',
                    'recommendation_count': 5
                },
                {
                    'score_range': '<45%',
                    'classification': 'LEGÍTIMO',
                    'priority': 'BAJA',
                    'main_action': '✅ Conversación parece legítima',
                    'description': 'Pocos/ningún indicador de vishing - Precauciones estándar',
                    'recommendation_count': 3
                }
            ],
            
            'output_structure': {
                'verdict': 'Clasificación final (FRAUDE/SOSPECHOSO/MONITOREAR/LEGÍTIMO)',
                'confidence': 'Porcentaje de confianza del análisis',
                'risk_level': 'Nivel de riesgo (CRÍTICO/ALTO/MEDIO/BAJO/NORMAL)',
                'security_context': 'Contexto de seguridad adaptativo',
                'evidence': 'Array de evidencias con tipo, severidad, detalle y contribución',
                'breakdown': 'Desglose técnico de cada componente del análisis',
                'recommendations': 'Array de recomendaciones con prioridad, acción y razón',
                'summary': 'Resumen ejecutivo en lenguaje natural'
            },
            
            'integration': {
                'aggregates_from': [
                    'VishingScorer',
                    'VishingKeywords',
                    'SentimentManager',
                    'LinguisticAnalyzer',
                    'ConversationAnalyzer',
                    'AcousticAnalyzer',
                    'IncongruenceDetector',
                    'AdaptiveThreshold'
                ],
                'output_format': 'JSON',
                'ui_display': 'Dashboard de Explicabilidad en interfaz web',
                'api_available': True
            },
            
            'example_explanation': {
                'verdict': 'FRAUDE',
                'confidence': '85%',
                'risk_level': 'CRÍTICO',
                'evidence_count': 5,
                'top_evidence': [
                    '🔑 KEYWORDS (25.0%) - Detectadas 3 categorías sospechosas',
                    '🤖 ML_MODEL (20.0%) - Modelo ML detecta 78% probabilidad',
                    '😰 SENTIMENT (15.0%) - Sentimiento NEGATIVO con miedo',
                    '📝 LINGUISTIC (20.0%) - 4 patrones lingüísticos sospechosos',
                    '⏱️ TEMPORAL (10.0%) - 3 anomalías conversacionales'
                ],
                'top_recommendation': '🚨 TERMINAR LA LLAMADA INMEDIATAMENTE'
            }
        }
        
        print(f"[API] ExplainableDetector info: 7 tipos de evidencia, 4 niveles de riesgo")
        return jsonify(detector_info)
    except Exception as e:
        error_msg = str(e)
        print(f"[API ERROR] Error en explainable_detector_info: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/change_sentiment_engine', methods=['POST'])
def change_sentiment_engine():
    """API para cambiar motor de sentimiento"""
    try:
        data = request.get_json()
        engine_id = data.get('engine_id')
        
        if not engine_id:
            return jsonify({'success': False, 'error': 'engine_id requerido'})
        
        print(f"[SENTIMENT] Cambiando motor a: {engine_id}")
        success = transcriptor.sentiment_manager.set_engine(engine_id)
        
        if success:
            engine_name = transcriptor.sentiment_manager.get_current_engine_name()
            print(f"[SENTIMENT] Motor cambiado exitosamente a: {engine_name}")
            return jsonify({
                'success': True,
                'engine': engine_name,
                'message': f'Motor cambiado a {engine_name}'
            })
        else:
            print(f"[ERROR] No se pudo cambiar al motor: {engine_id}")
            return jsonify({
                'success': False,
                'error': f'No se pudo cambiar al motor {engine_id}'
            })
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Error cambiando motor de sentimiento: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/sentiment_metrics')
def get_sentiment_metrics():
    """API para obtener métricas de sentimiento de la conversación"""
    try:
        metrics = transcriptor.sentiment_manager.get_conversation_metrics()
        fraud_risk = transcriptor.sentiment_manager.compute_fraud_risk_score()
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'fraud_risk_score': fraud_risk
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
        
        print(f"[SOCKETIO] Solicitud de cambio de motor a: {engine_id}")
        
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
            
            print(f"[OK] Motor cambiado exitosamente via SocketIO a: {engine_id}")
        else:
            print(f"[ERROR] No se pudo cambiar al motor: {engine_id}")
            emit('error', {'message': f'No se pudo cambiar al motor {engine_id}'})
            
    except Exception as e:
        error_msg = f"Error cambiando motor: {str(e)}"
        print(f"[ERROR] {error_msg}")
        emit('error', {'message': error_msg})

@socketio.on('change_sentiment_engine')
def handle_change_sentiment_engine(data):
    """Manejar cambio de motor de sentimiento via SocketIO"""
    try:
        engine_id = data.get('engine_id')
        if not engine_id:
            emit('error', {'message': 'ID de motor de sentimiento no especificado'})
            return
        
        print(f"[SOCKETIO] Solicitud de cambio de motor de sentimiento a: {engine_id}")
        
        # Cambiar motor de sentimiento
        if transcriptor.sentiment_manager.set_engine(engine_id):
            engine_name = transcriptor.sentiment_manager.get_current_engine_name()
            current_engine = transcriptor.sentiment_manager.current_engine
            engine_info = {
                'name': engine_name,
                'description': current_engine.description if current_engine else '',
                'model_size': current_engine.model_size if current_engine else '',
                'speed': current_engine.speed if current_engine else ''
            }
            
            emit('sentiment_engine_changed', {
                'success': True,
                'engine_id': engine_id,
                'engine_name': engine_name,
                'engine_info': engine_info
            })
            
            print(f"[OK] Motor de sentimiento cambiado exitosamente via SocketIO a: {engine_name}")
        else:
            print(f"[ERROR] No se pudo cambiar al motor de sentimiento: {engine_id}")
            emit('error', {'message': f'No se pudo cambiar al motor de sentimiento {engine_id}'})
            
    except Exception as e:
        error_msg = f"Error cambiando motor de sentimiento: {str(e)}"
        print(f"[ERROR] {error_msg}")
        emit('error', {'message': error_msg})

@socketio.on('request_sentiment_metrics')
def handle_request_sentiment_metrics():
    """Enviar métricas de sentimiento acumuladas"""
    try:
        metrics = transcriptor.sentiment_manager.get_conversation_metrics()
        fraud_risk = transcriptor.sentiment_manager.compute_fraud_risk_score()
        
        emit('sentiment_metrics', {
            'success': True,
            'metrics': metrics,
            'fraud_risk_score': fraud_risk
        })
        
        print("[SOCKETIO] Métricas de sentimiento enviadas")
    except Exception as e:
        error_msg = f"Error obteniendo métricas: {str(e)}"
        print(f"[ERROR] {error_msg}")
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
    print("[INIT] TRANSCRIPTOR MODULAR SPEECH-TO-TEXT")
    print("[INFO] Motores disponibles: DeepSpeech, Whisper, Silero")
    print("[INFO] Panel de configuracion de audio FUNCIONAL")
    print("[INFO] URL: http://localhost:5003")
    print("="*80)
    
    try:
        socketio.run(app, host='0.0.0.0', port=5003, debug=False)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Deteniendo servidor...")
        transcriptor.stop_listening()
        transcriptor.engine_manager.cleanup()
        print("[OK] Servidor detenido correctamente")
