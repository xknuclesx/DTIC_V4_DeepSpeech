#!/usr/bin/env python3
"""
Gestor de Modelos de Análisis de Sentimiento Offline
Soporta: XLM-RoBERTa, BETO, VADER
"""

import time
from collections import deque
from typing import Optional, Dict, List
import statistics

class BaseSentimentEngine:
    """Clase base para motores de sentimiento"""
    def __init__(self):
        self.name = "Base"
        self.display_name = "Base Engine"
        self.description = ""
        self.available = False
        self.initialized = False
        self.requires_download = False
        self.model_size = "Unknown"
        self.speed = "Unknown"
        self.languages = []
        
    def initialize(self):
        """Inicializar modelo"""
        raise NotImplementedError
        
    def analyze(self, text: str) -> Dict:
        """Analizar sentimiento de texto"""
        raise NotImplementedError
        
    def cleanup(self):
        """Limpiar recursos"""
        pass

class VADEREngine(BaseSentimentEngine):
    """Motor VADER - Rápido y ligero (lexicón)"""
    def __init__(self):
        super().__init__()
        self.name = "vader"
        self.display_name = "VADER"
        self.description = "Lexicon-based (muy rápido, baseline)"
        self.model_size = "< 1 MB"
        self.speed = "~10k texts/sec"
        self.requires_download = False
        self.languages = ["en", "es (limitado)"]
        self.analyzer = None
        
    def initialize(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.available = True
            self.initialized = True
            print(f"[OK] {self.display_name} inicializado correctamente")
            return True
        except Exception as e:
            print(f"[ERROR] Error inicializando {self.display_name}: {e}")
            print("[INFO] Instalar con: pip install vaderSentiment")
            self.available = False
            return False
    
    def analyze(self, text: str) -> Dict:
        if not self.initialized or not text:
            return self._empty_result()
        
        try:
            scores = self.analyzer.polarity_scores(text)
            
            # Determinar label basado en compound
            if scores['compound'] >= 0.05:
                label = 'POS'
            elif scores['compound'] <= -0.05:
                label = 'NEG'
            else:
                label = 'NEU'
            
            return {
                'label': label,
                'score': abs(scores['compound']),
                'compound': scores['compound'],
                'pos': scores['pos'],
                'neg': scores['neg'],
                'neu': scores['neu'],
                'engine': self.name,
                'engine_display': self.display_name,
                'success': True
            }
        except Exception as e:
            print(f"[ERROR] Error en análisis {self.display_name}: {e}")
            return self._empty_result()
    
    def _empty_result(self):
        return {
            'label': 'NEU',
            'score': 0.0,
            'compound': 0.0,
            'pos': 0.0,
            'neg': 0.0,
            'neu': 1.0,
            'engine': self.name,
            'engine_display': self.display_name,
            'success': False
        }

class BETOEngine(BaseSentimentEngine):
    """Motor BETO - Español nativo (España + LATAM)"""
    def __init__(self):
        super().__init__()
        self.name = "beto"
        self.display_name = "BETO"
        self.description = "BERT Español (España + LATAM, alta precisión)"
        self.model_size = "420 MB"
        self.speed = "~300ms/text"
        self.requires_download = True
        self.languages = ["es"]
        self.classifier = None
        
    def initialize(self):
        try:
            from transformers import pipeline
            print(f"[INFO] Descargando modelo {self.display_name} (primera vez puede tardar)...")
            self.classifier = pipeline(
                'sentiment-analysis',
                model='finiteautomata/beto-sentiment-analysis',
                device=-1  # CPU
            )
            self.available = True
            self.initialized = True
            print(f"[OK] {self.display_name} inicializado correctamente")
            return True
        except Exception as e:
            print(f"[ERROR] Error inicializando {self.display_name}: {e}")
            print("[INFO] Instalar con: pip install transformers torch")
            self.available = False
            return False
    
    def analyze(self, text: str) -> Dict:
        if not self.initialized or not text:
            return self._empty_result()
        
        try:
            result = self.classifier(text[:512])[0]  # Limitar a 512 tokens
            
            return {
                'label': result['label'],
                'score': result['score'],
                'compound': self._label_to_compound(result['label'], result['score']),
                'engine': self.name,
                'engine_display': self.display_name,
                'success': True
            }
        except Exception as e:
            print(f"[ERROR] Error en análisis {self.display_name}: {e}")
            return self._empty_result()
    
    def _label_to_compound(self, label: str, score: float) -> float:
        """Convertir label a compound score (-1 a 1)"""
        if label == 'POS':
            return score * 0.8
        elif label == 'NEG':
            return -score * 0.8
        else:  # NEU
            return 0.0
    
    def _empty_result(self):
        return {
            'label': 'NEU',
            'score': 0.0,
            'compound': 0.0,
            'engine': self.name,
            'engine_display': self.display_name,
            'success': False
        }



class XLMRobertaEngine(BaseSentimentEngine):
    """Motor XLM-RoBERTa - Multilingüe (100+ idiomas)"""
    def __init__(self):
        super().__init__()
        self.name = "xlm_roberta"
        self.display_name = "XLM-RoBERTa"
        self.description = "Multilingüe (100+ idiomas, code-switching)"
        self.model_size = "1.1 GB"
        self.speed = "~500ms/text"
        self.requires_download = True
        self.languages = ["100+ idiomas"]
        self.classifier = None
        
    def initialize(self):
        try:
            from transformers import pipeline
            print(f"[INFO] Descargando modelo {self.display_name} (primera vez puede tardar)...")
            self.classifier = pipeline(
                'sentiment-analysis',
                model='cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual',
                device=-1  # CPU
            )
            self.available = True
            self.initialized = True
            print(f"[OK] {self.display_name} inicializado correctamente")
            return True
        except Exception as e:
            print(f"[ERROR] Error inicializando {self.display_name}: {e}")
            print("[INFO] Instalar con: pip install transformers torch")
            self.available = False
            return False
    
    def analyze(self, text: str) -> Dict:
        if not self.initialized or not text:
            return self._empty_result()
        
        try:
            result = self.classifier(text[:512])[0]
            
            # Mapear labels (puede ser positive/negative/neutral o LABEL_X)
            label_map = {
                'positive': 'POS',
                'negative': 'NEG',
                'neutral': 'NEU',
                'LABEL_2': 'POS',
                'LABEL_0': 'NEG',
                'LABEL_1': 'NEU'
            }
            
            mapped_label = label_map.get(result['label'], result['label'])
            
            return {
                'label': mapped_label,
                'score': result['score'],
                'compound': self._label_to_compound(mapped_label, result['score']),
                'engine': self.name,
                'engine_display': self.display_name,
                'success': True
            }
        except Exception as e:
            print(f"[ERROR] Error en análisis {self.display_name}: {e}")
            return self._empty_result()
    
    def _label_to_compound(self, label: str, score: float) -> float:
        """Convertir label a compound score (-1 a 1)"""
        if label == 'POS':
            return score * 0.8
        elif label == 'NEG':
            return -score * 0.8
        else:
            return 0.0
    
    def _empty_result(self):
        return {
            'label': 'NEU',
            'score': 0.0,
            'compound': 0.0,
            'engine': self.name,
            'engine_display': self.display_name,
            'success': False
        }

class SentimentEngineManager:
    """Gestor de motores de análisis de sentimiento"""
    def __init__(self):
        self.engines = {
            'vader': VADEREngine(),
            'beto': BETOEngine(),
            'xlm_roberta': XLMRobertaEngine()
        }
        
        self.current_engine = None
        self.current_engine_name = None
        
        # Buffer de análisis para métricas acumuladas
        self.analysis_history = deque(maxlen=100)
        
        print("[INFO] SentimentEngineManager inicializado")
        
        # Inicializar VADER por defecto (rápido y sin descargas)
        self._initialize_default_engine()
    
    def _initialize_default_engine(self):
        """Inicializar motor por defecto (VADER)"""
        try:
            if self.set_engine('vader'):
                print("[OK] Motor de sentimiento por defecto (VADER) configurado")
        except Exception as e:
            print(f"[WARNING] No se pudo inicializar motor por defecto: {e}")
    
    def set_engine(self, engine_id: str) -> bool:
        """Cambiar motor de sentimiento activo"""
        if engine_id not in self.engines:
            print(f"[ERROR] Motor '{engine_id}' no existe")
            return False
        
        engine = self.engines[engine_id]
        
        # Inicializar si no está listo
        if not engine.initialized:
            print(f"[SENTIMENT] Inicializando motor {engine.display_name}...")
            if not engine.initialize():
                print(f"[ERROR] No se pudo inicializar {engine.display_name}")
                return False
        
        self.current_engine = engine
        self.current_engine_name = engine_id
        print(f"[OK] Motor de sentimiento activo: {engine.display_name}")
        return True
    
    def analyze_text(self, text: str) -> Dict:
        """Analizar texto con motor actual"""
        if not self.current_engine:
            print("[WARNING] No hay motor de sentimiento activo")
            return self._empty_analysis()
        
        if not text or not text.strip():
            return self._empty_analysis()
        
        try:
            start_time = time.time()
            result = self.current_engine.analyze(text)
            analysis_time = (time.time() - start_time) * 1000  # ms
            
            # Agregar metadata
            result['text'] = text
            result['timestamp'] = time.time()
            result['analysis_time_ms'] = round(analysis_time, 2)
            
            # Guardar en historial si fue exitoso
            if result.get('success'):
                self.analysis_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Error analizando texto: {e}")
            return self._empty_analysis()
    
    def get_available_engines(self) -> Dict:
        """Obtener lista de motores disponibles"""
        result = {}
        for engine_id, engine in self.engines.items():
            result[engine_id] = {
                'name': engine.display_name,
                'description': engine.description,
                'available': engine.available,
                'initialized': engine.initialized,
                'model_size': engine.model_size,
                'requires_download': engine.requires_download,
                'languages': engine.languages
            }
        return result
    
    def get_current_engine_info(self) -> Optional[Dict]:
        """Obtener info del motor actual"""
        if not self.current_engine:
            return None
        
        return {
            'engine_id': self.current_engine_name,
            'name': self.current_engine.display_name,
            'description': self.current_engine.description,
            'model_size': self.current_engine.model_size,
            'initialized': self.current_engine.initialized
        }
    
    def get_current_engine_name(self) -> str:
        """Obtener nombre del motor actual"""
        return self.current_engine.display_name if self.current_engine else "None"
    
    def get_conversation_metrics(self) -> Dict:
        """Obtener métricas acumuladas de la conversación"""
        if not self.analysis_history:
            return {
                'count': 0,
                'mean_sentiment': 0.0,
                'neg_ratio': 0.0,
                'pos_ratio': 0.0,
                'neu_ratio': 0.0,
                'volatility': 0.0,
                'recent_trend': 'stable'
            }
        
        compounds = [a['compound'] for a in self.analysis_history if a.get('success')]
        labels = [a['label'] for a in self.analysis_history if a.get('success')]
        
        if not compounds:
            return {'count': 0}
        
        neg_count = sum(1 for l in labels if l == 'NEG')
        pos_count = sum(1 for l in labels if l == 'POS')
        neu_count = sum(1 for l in labels if l == 'NEU')
        total = len(labels)
        
        return {
            'count': total,
            'mean_sentiment': round(statistics.mean(compounds), 3),
            'neg_ratio': round(neg_count / total, 3) if total > 0 else 0,
            'pos_ratio': round(pos_count / total, 3) if total > 0 else 0,
            'neu_ratio': round(neu_count / total, 3) if total > 0 else 0,
            'volatility': round(statistics.pstdev(compounds), 3) if len(compounds) > 1 else 0,
            'recent_trend': self._calculate_trend(compounds[-10:]) if len(compounds) >= 3 else 'stable'
        }
    
    def _calculate_trend(self, recent_scores: List[float]) -> str:
        """Calcular tendencia reciente"""
        if len(recent_scores) < 3:
            return 'stable'
        
        first_half = sum(recent_scores[:len(recent_scores)//2])
        second_half = sum(recent_scores[len(recent_scores)//2:])
        
        if second_half < first_half - 0.2:
            return 'declining'
        elif second_half > first_half + 0.2:
            return 'improving'
        return 'stable'
    
    def compute_fraud_risk_score(self) -> float:
        """Calcular score de riesgo de fraude basado en sentimiento (0-1)"""
        metrics = self.get_conversation_metrics()
        
        if metrics['count'] == 0:
            return 0.0
        
        # Ponderación de factores
        score = 0.0
        
        # Sentimiento negativo alto (40%)
        if metrics['mean_sentiment'] < -0.3:
            score += 0.4 * abs(metrics['mean_sentiment'])
        
        # Ratio de mensajes negativos alto (30%)
        score += 0.3 * metrics['neg_ratio']
        
        # Volatilidad alta (20%)
        score += 0.2 * min(1.0, metrics['volatility'])
        
        # Tendencia declining (10%)
        if metrics['recent_trend'] == 'declining':
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _empty_analysis(self) -> Dict:
        """Resultado vacío"""
        return {
            'label': 'NEU',
            'score': 0.0,
            'compound': 0.0,
            'engine': 'none',
            'engine_display': 'None',
            'success': False
        }
    
    def clear_history(self):
        """Limpiar historial de análisis"""
        self.analysis_history.clear()
        print("[INFO] Historial de sentimientos limpiado")
    
    def cleanup(self):
        """Limpiar recursos"""
        for engine in self.engines.values():
            engine.cleanup()
        print("[INFO] SentimentEngineManager limpiado")
