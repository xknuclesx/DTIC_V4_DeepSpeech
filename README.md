# 🎤 Transcriptor Modular Speech-to-Text# 🎤 Transcriptor Modular Speech-to-Text# 🎤 Transcriptor Modular Speech-to-Text



Sistema avanzado de transcripción de voz a texto con **arquitectura modular** que soporta 4 motores de transcripción diferentes. Incluye detección de fraude en tiempo real con machine learning.## 📦 Instalación de Motores Adicionales



## 🚀 CaracterísticasSistema avanzado de transcripción de voz a texto con **arquitectura modular** que soporta 4 motores de transcripción diferentes. Incluye detección de fraude en tiempo real con machine learning.



- **4 motores de transcripción**: DeepSpeech (Google SR), Whisper, Silero, VoskPara instalar Whisper, Silero y Vosk (opcional):

- **Transcripción en tiempo real** con WebSockets (SocketIO)

- **Detección de fraude con ML** (LogisticRegression + TF-IDF)## 🚀 Características```bash

- **Interfaz web moderna** con Bootstrap 5

- **Configuración flexible** de audio en tiempo real.\instalar_motores.bat

- **Arquitectura modular** fácilmente extensible

- **4 motores de transcripción**: DeepSpeech, Whisper, Silero, Vosk```

## 🔧 Motores Disponibles

- **Transcripción en tiempo real** con SocketIO

| Motor | Velocidad | Precisión | Internet | Recursos |

|-------|-----------|-----------|----------|----------|- **Detección de fraude con ML** (LogisticRegression + TF-IDF)## 📁 Estructura del Proyecto

| **DeepSpeech** (Google SR) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Requiere | Bajo |

| **OpenAI Whisper** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ Offline | Alto |- **Interfaz web moderna** con Bootstrap 5

| **Silero STT** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ Offline | Medio |

| **Vosk** (Kaldi) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ Offline | Bajo |- **Configuración flexible** de audio en tiempo real```



## 🚀 Inicio Rápido- **Arquitectura modular** fácilmente extensibleDTIC_V4_DeepSpeech/



### Instalación├── transcriptor.py              # Servidor principal con interfaz integrada



```bash## 🔧 Motores Disponibles├── engine_manager.py            # Gestor de motores de transcripción

# 1. Clonar repositorio

git clone https://github.com/tu-usuario/DTIC_V4_DeepSpeech.git├── engines/                     # Motores de transcripción

cd DTIC_V4_DeepSpeech

| Motor | Velocidad | Precisión | Internet | Recursos |│   ├── base_engine.py          #   Clase base para motores

# 2. Crear entorno virtual

python -m venv transcriptor_env|-------|-----------|-----------|-----------|-----------|│   ├── deepspeech_engine.py    #   Motor Google Speech Recognition



# 3. Activar entorno (Windows)| **DeepSpeech** (Google SR) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Requiere | Bajo |│   ├── whisper_engine.py       #   Motor OpenAI Whisper

.\transcriptor_env\Scripts\activate

| **OpenAI Whisper** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ Offline | Alto |│   ├── silero_engine.py        #   Motor Silero STT

# 4. Instalar dependencias

pip install -r requirements.txt| **Silero STT** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ Offline | Medio |│   └── vosk_engine.py          #   Motor Vosk/Kaldi

```

| **Vosk** (Kaldi) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ Offline | Bajo |├── models/                      # Modelos de Vosk

### Ejecutar

├── transcriptor_env/            # Entorno virtual Python

```bash

# Opción 1: Script de inicio## 🚀 Inicio Rápido├── best_model_lr.joblib         # Modelo ML detección fraude

.\iniciar_transcriptor.bat

├── vectorizer_tfidf.joblib      # Vectorizador TF-IDF

# Opción 2: Manual

.\transcriptor_env\Scripts\activate### 1. Ejecutar el sistema├── requirements.txt             # Dependencias Python

python transcriptor.py

``````bash└── iniciar_transcriptor.bat     # Script inicio rápido



Abrir navegador en: **http://localhost:5003**# Doble clic en el archivo:```



## 📦 Instalación Completainiciar_transcriptor.bat



Para instrucciones detalladas de instalación, consulta [INSTALL.md](INSTALL.md)## ⚙️ Configuración de Audio



## 🎯 Uso# O desde línea de comandos:



### Interfaz Webtranscriptor_env\Scripts\activate- **Umbral de Energía**: Controla sensibilidad del micrófono



1. **Seleccionar motor** - Haz clic en las tarjetas de motores disponiblespython transcriptor.py- **Pausa Entre Frases**: Tiempo de silencio para detectar fin de frase

2. **Configurar audio** - Ajusta los parámetros según tu ambiente:

   - **Umbral de Energía**: Sensibilidad del micrófono (300-4000)```- **Límite de Frase**: Tiempo máximo de grabación continua

   - **Pausa Entre Frases**: Tiempo de silencio para detectar fin de frase (0.1-2.0s)

   - **Límite de Frase**: Duración máxima de grabación (3-15s)- **Timeout de Escucha**: Tiempo máximo esperando inicio de voz

   - **Idioma**: Selecciona el idioma de transcripción

3. **Iniciar transcripción** - Presiona el botón "Iniciar" y habla### 2. Acceder a la interfaz- **Idioma**: Selección de idioma para mejor precisión

4. **Ver resultados** - Las transcripciones aparecen en tiempo real con análisis de fraude

Abrir navegador en: **http://localhost:5003**

### Detección de Fraude

## 🛡️ Detección de Fraude

El sistema analiza cada transcripción en tiempo real:

### 3. Usar el sistema

- 🟢 **Texto Normal**: Borde verde, probabilidad baja de fraude

- 🔴 **Fraude Detectado**: Borde rojo, probabilidad alta de fraude1. **Seleccionar motor** haciendo clic en las tarjetasEl sistema incluye un detector de fraude entrenado que analiza:

- 📊 **Probabilidad**: Porcentaje de confianza del análisis

2. **Ajustar configuración** de audio si es necesario- **Keywords específicas** relacionadas con fraudes

## 📁 Estructura del Proyecto

3. **Hacer clic en "Iniciar"** para comenzar transcripción- **Patrones de texto** usando TF-IDF y regresión logística

```

DTIC_V4_DeepSpeech/4. **Hablar al micrófono** - los resultados aparecen en tiempo real- **Probabilidad de fraude** en tiempo real

├── transcriptor.py              # Servidor principal con interfaz integrada

├── engine_manager.py            # Gestor de motores de transcripción

├── engines/                     # Módulo de motores

│   ├── base_engine.py          # Clase abstracta base## 📦 Instalación de Motores Adicionales## 🔧 Requisitos del Sistema

│   ├── deepspeech_engine.py    # Motor Google Speech Recognition

│   ├── whisper_engine.py       # Motor OpenAI Whisper

│   ├── silero_engine.py        # Motor Silero STT

│   └── vosk_engine.py          # Motor Vosk/KaldiPara instalar Whisper, Silero y Vosk (opcional):- **Python 3.8+**

├── models/                      # Modelos de Vosk (descarga separada)

├── transcriptor_env/            # Entorno virtual Python```bash- **Windows 10/11** (PowerShell)

├── requirements.txt             # Dependencias básicas

├── iniciar_transcriptor.bat     # Script de inicio rápido.\instalar_motores.bat- **Micrófono** funcional

├── README.md                    # Este archivo

├── INSTALL.md                   # Guía de instalación detallada```- **4GB RAM** mínimo (8GB recomendado para Whisper)

├── CONTRIBUTING.md              # Guía de contribución

└── LICENSE                      # Licencia MIT- **Conexión a internet** (solo para DeepSpeech)

```

## 📁 Estructura del Proyecto

**Nota**: Los modelos de ML (`best_model_lr.joblib` y `vectorizer_tfidf.joblib`) se deben descargar por separado debido a su tamaño.

## 📊 Dependencias Principales

## ⚙️ Configuración de Audio

```

### Parámetros Ajustables

DTIC_V4_DeepSpeech/- Flask 3.0.0 + Flask-SocketIO 5.3.6

- **Umbral de Energía**: Controla la sensibilidad del micrófono

  - Valores altos = menos sensible (menos ruido de fondo)├── transcriptor.py              # Servidor principal con interfaz integrada- SpeechRecognition 3.10.0

  - Valores bajos = más sensible (capta sonidos más suaves)

  ├── engine_manager.py            # Gestor de motores de transcripción- PyAudio 0.2.14

- **Pausa Entre Frases**: Tiempo de silencio para detectar fin de frase

  - Valores bajos = respuesta más rápida├── engines/                     # Motores de transcripción- scikit-learn 1.6.1

  - Valores altos = espera más tiempo antes de procesar

│   ├── base_engine.py          #   Clase base para motores- [Opcional] openai-whisper, silero-vad, vosk

- **Límite de Frase**: Tiempo máximo de grabación continua

  - Útil para frases muy largas│   ├── deepspeech_engine.py    #   Motor Google Speech Recognition



- **Timeout de Escucha**: Tiempo máximo esperando inicio de voz│   ├── whisper_engine.py       #   Motor OpenAI Whisper## 🐛 Solución de Problemas

  - Si no detecta voz en este tiempo, para la grabación

│   ├── silero_engine.py        #   Motor Silero STT

- **Ajuste Dinámico**: Adaptación automática al ruido del ambiente

  - Útil en lugares con ruido variable│   └── vosk_engine.py          #   Motor Vosk/Kaldi### Micrófono no detectado



## 🔧 Requisitos del Sistema├── models/                      # Modelos de Vosk- Verificar permisos de micrófono en Windows



- **Python 3.8+**├── transcriptor_env/            # Entorno virtual Python- Comprobar que no esté siendo usado por otra aplicación

- **Windows 10/11** (con PowerShell)

- **Micrófono** funcional├── best_model_lr.joblib         # Modelo ML detección fraude

- **4GB RAM** mínimo (8GB recomendado para Whisper)

- **Conexión a internet** (solo para motor DeepSpeech)├── vectorizer_tfidf.joblib      # Vectorizador TF-IDF### Motor no disponible



## 📊 Dependencias Principales├── requirements.txt             # Dependencias Python- Ejecutar `.\instalar_motores.bat` para instalar dependencias



```└── iniciar_transcriptor.bat     # Script inicio rápido- Verificar conexión a internet para DeepSpeech

Flask==3.0.0

Flask-SocketIO==5.3.6```

SpeechRecognition==3.10.0

PyAudio==0.2.14### Error de transcripción

scikit-learn==1.6.1

joblib==1.3.2## ⚙️ Configuración de Audio- Verificar nivel de ruido ambiental

numpy==1.24.3

```- Ajustar umbral de energía en configuración



**Dependencias opcionales** (motores adicionales):- **Umbral de Energía**: Controla sensibilidad del micrófono- Cambiar de motor de transcripción

- `openai-whisper` - Motor Whisper

- `torch` + `torchaudio` - Requerido para Whisper y Silero- **Pausa Entre Frases**: Tiempo de silencio para detectar fin de frase

- `vosk` - Motor Vosk/Kaldi

- **Límite de Frase**: Tiempo máximo de grabación continua## 📝 Logs y Debug

## 🐛 Solución de Problemas

- **Timeout de Escucha**: Tiempo máximo esperando inicio de voz

### Micrófono no detectado

- Verificar permisos de micrófono en Windows- **Idioma**: Selección de idioma para mejor precisiónEl sistema incluye logging detallado:

- Comprobar que no esté siendo usado por otra aplicación

- Reiniciar el sistema- **Panel de debug** en la interfaz web



### Motor no disponible## 🛡️ Detección de Fraude- **Consola del servidor** con información técnica

- Ejecutar instalación de dependencias adicionales

- Verificar conexión a internet para DeepSpeech- **Estados de SocketIO** en tiempo real

- Descargar modelos necesarios (Vosk)

El sistema incluye un detector de fraude entrenado que analiza:

### Error de transcripción

- Verificar nivel de ruido ambiental- **Keywords específicas** relacionadas con fraudes---

- Ajustar umbral de energía en configuración

- Cambiar de motor de transcripción- **Patrones de texto** usando TF-IDF y regresión logística



### PyAudio no se instala- **Probabilidad de fraude** en tiempo real**Desarrollado por**: Equipo DTIC  

```powershell

# Windows: Instalar desde pipwin**Versión**: 4.0  

pip install pipwin

pipwin install pyaudio## 🔧 Requisitos del Sistema**Última actualización**: Septiembre 2025ma avanzado de transcripción de voz a texto con **arquitectura modular** que soporta 4 motores de transcripción diferentes. Incluye detección de fraude en tiempo real con machine learning.

```



## 🤝 Contribuir

- **Python 3.8+**## 🚀 Características

¡Las contribuciones son bienvenidas! Por favor lee [CONTRIBUTING.md](CONTRIBUTING.md) para conocer el proceso.

- **Windows 10/11** (PowerShell)

### Pasos rápidos:

- **Micrófono** funcional- **4 motores de transcripción**: DeepSpeech, Whisper, Silero, Vosk

1. Fork el proyecto

2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)- **4GB RAM** mínimo (8GB recomendado para Whisper)- **Transcripción en tiempo real** con SocketIO

3. Commit tus cambios (`git commit -m 'feat: agregar nueva funcionalidad'`)

4. Push a la rama (`git push origin feature/nueva-funcionalidad`)- **Conexión a internet** (solo para DeepSpeech)- **Detección de fraude con ML** (LogisticRegression + TF-IDF)

5. Abre un Pull Request

- **Interfaz web moderna** con Bootstrap 5

## 📝 Licencia

## 📊 Dependencias Principales- **Configuración flexible** de audio en tiempo real

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

- **Arquitectura modular** fácilmente extensible

## 👥 Autores

- Flask 3.0.0 + Flask-SocketIO 5.3.6

- **Equipo DTIC** - *Desarrollo inicial*

- SpeechRecognition 3.10.0## 🔧 Motores Disponibles

## 🙏 Agradecimientos

- PyAudio 0.2.14

- [SpeechRecognition](https://github.com/Uberi/speech_recognition) por la biblioteca base

- [OpenAI Whisper](https://github.com/openai/whisper) por el modelo de alta precisión- scikit-learn 1.6.1| Motor | Velocidad | Precisión | Internet | Recursos |

- [Vosk](https://alphacephei.com/vosk/) por el motor offline

- [Silero](https://github.com/snakers4/silero-models) por los modelos STT- [Opcional] openai-whisper, silero-vad, vosk|-------|-----------|-----------|-----------|-----------|



## 📞 Soporte| **DeepSpeech** (Google SR) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Requiere | Bajo |



Si encuentras algún problema o tienes preguntas:## 🐛 Solución de Problemas| **OpenAI Whisper** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ Offline | Alto |



1. Revisa la sección de [Issues](https://github.com/tu-usuario/DTIC_V4_DeepSpeech/issues)| **Silero STT** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ Offline | Medio |

2. Lee [INSTALL.md](INSTALL.md) para problemas de instalación

3. Abre un nuevo issue si no encuentras solución### Micrófono no detectado| **Vosk** (Kaldi) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ Offline | Bajo |



---- Verificar permisos de micrófono en Windows



**Versión**: 4.0  - Comprobar que no esté siendo usado por otra aplicación## 🚀 Inicio Rápido

**Última actualización**: Octubre 2025



🎤 **¡Listo para transcribir con múltiples motores de última generación!**

### Motor no disponible### 1. Ejecutar el sistema

- Ejecutar `.\instalar_motores.bat` para instalar dependencias```bash

- Verificar conexión a internet para DeepSpeech# Doble clic en el archivo:

iniciar_transcriptor.bat

### Error de transcripción

- Verificar nivel de ruido ambiental# O desde línea de comandos:

- Ajustar umbral de energía en configuracióntranscriptor_env\Scripts\activate

- Cambiar de motor de transcripciónpython transcriptor.py

```

## 📝 Logs y Debug

### 2. Acceder a la interfaz

El sistema incluye logging detallado:Abrir navegador en: **http://localhost:5003**

- **Panel de debug** en la interfaz web

- **Consola del servidor** con información técnica### 3. Usar el sistema

- **Estados de SocketIO** en tiempo real1. **Seleccionar motor** haciendo clic en las tarjetas

2. **Ajustar configuración** de audio si es necesario

---3. **Hacer clic en "Iniciar"** para comenzar transcripción

4. **Hablar al micrófono** - los resultados aparecen en tiempo real

**Desarrollado por**: Equipo DTIC  

**Versión**: 4.0  ## � Instalación de Motores Adicionales

**Última actualización**: Septiembre 2025
Para instalar Whisper, Silero y Vosk (opcional):
```bash
.\instalar_motores.bat
```

### ⚙️ **Instalación Manual**
```powershell
# Activar entorno virtual
.\transcriptor_env\Scripts\activate

# Instalar motores adicionales
pip install -r requirements_full.txt
```

---

## 🎯 Uso del Sistema

### **1. Inicio Rápido**
```powershell
.\iniciar.bat
```

### **2. Interfaz Web**
1. Abrir navegador en: **`http://localhost:5002`**
2. **Seleccionar Motor**: Elegir entre los 4 motores disponibles
3. **Configurar Audio**: Ajustar parámetros según ambiente
4. **Iniciar Transcripción**: Presionar "Iniciar" y hablar
5. **Ver Resultados**: Transcripciones en tiempo real con análisis de fraude

### **3. Características de la Interfaz**

#### **🎛️ Panel de Selección de Motores**
- Tarjetas visuales para cada motor
- Indicadores de estado (✅ disponible / ⚠️ no instalado)
- Iconos de conectividad (🌐 internet / 💾 offline)
- Cambio dinámico sin interrumpir transcripción

#### **🔧 Configuración de Audio en Tiempo Real**
- **Umbral de Energía**: Sensibilidad del micrófono (300-4000)
- **Pausa Entre Frases**: Tiempo para detectar final (0.1-2.0s)
- **Límite de Frase**: Duración máxima de grabación (3-15s)
- **Timeout de Escucha**: Tiempo de espera sin audio (1-5s)
- **Idioma**: Selección de idioma de transcripción
- **Ajuste Dinámico**: Adaptación automática al ruido

#### **🛡️ Detección de Fraude Integrada**
- **Machine Learning**: Modelo entrenado con 60% de threshold
- **Keywords**: Detección de palabras clave sospechosas
- **Análisis en Tiempo Real**: Cada transcripción es analizada
- **Indicadores Visuales**: 🔴 Fraude detectado | 🟢 Texto normal

---

## 🔧 Configuración Avanzada

### **Configuración Específica por Motor**

#### **OpenAI Whisper**
```python
{
    'use_faster_whisper': True,    # Usar faster-whisper (recomendado)
    'model_size': 'base',          # tiny, base, small, medium, large
    'device': 'cpu'                # cpu, cuda
}
```

#### **Silero STT**
```python
{
    'model_language': 'es',        # es, en, de, uk, uz
    'device': 'cpu'                # cpu, cuda
}
```

#### **Vosk (Kaldi)**
```python
{
    'model_path': '/path/to/model', # Ruta al modelo descargado
    'model_language': 'es'          # Idioma del modelo
}
```

### **Descarga de Modelos Vosk**
Para usar Vosk, descarga modelos desde: [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)

---

## 📊 Comparativa de Rendimiento

| Característica | DeepSpeech | Whisper | Silero | Vosk |
|----------------|------------|---------|--------|------|
| **Velocidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Precisión** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Internet** | ✅ Requerido | ❌ Opcional | ❌ No | ❌ No |
| **Tamaño** | 10MB | 1-3GB | 100MB | 50MB-1GB |
| **Idiomas** | 20+ | 90+ | 5 | 20+ |
| **Configuración** | Fácil | Media | Media | Avanzada |

---

## 📁 Estructura del Proyecto

```
DTIC_V4_DeepSpeech/
├── 🎯 transcriptor.py              # Sistema principal modular
├── 🔧 engine_manager.py            # Gestor de motores
├── 📁 engines/                     # Módulo de motores
│   ├── base_engine.py              # Clase abstracta base
│   ├── deepspeech_engine.py        # Google Speech Recognition
│   ├── whisper_engine.py           # OpenAI Whisper + Faster-Whisper
│   ├── silero_engine.py            # Silero STT
│   └── vosk_engine.py              # Vosk (Kaldi)
├── 🤖 best_model_lr.joblib         # Modelo ML para fraude
├── 🔤 vectorizer_tfidf.joblib      # Vectorizador TF-IDF
├── 📦 requirements.txt             # Dependencias básicas
├── 📦 requirements_full.txt        # Dependencias completas
├── 🚀 iniciar.bat                  # Launcher del sistema
├── 📦 instalar_motores.bat         # Instalador automático
└── 📁 transcriptor_env/            # Entorno virtual Python
```

---

## 🛠️ Solución de Problemas

### **❌ Error: Motor no se puede cargar**
```powershell
# Instalar dependencias completas
.\instalar_motores.bat

# O manualmente
pip install -r requirements_full.txt
```

### **❌ Error: PyAudio no funciona**
```powershell
# Windows: Reinstalar PyAudio
pip uninstall pyaudio
pip install pyaudio
```

### **❌ Error: Vosk sin modelos**
1. Descargar modelo desde: https://alphacephei.com/vosk/models
2. Extraer en carpeta `models/`
3. Configurar ruta en la interfaz

### **❌ Error: CUDA no disponible**
```powershell
# Instalar PyTorch solo CPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 🌐 API y Endpoints

### **URLs del Sistema**
- **Aplicación Principal**: `http://localhost:5002`
- **Panel de Control**: Interfaz web completa

### **API REST Endpoints**
- `GET /api/engines` - Lista de motores disponibles
- `POST /api/change_engine` - Cambiar motor activo
- `GET /api/audio_config` - Configuración actual de audio
- `POST /start_listening` - Iniciar transcripción
- `POST /stop_listening` - Detener transcripción
- `POST /update_audio_config` - Actualizar configuración

### **WebSocket Events**
- `transcription_result` - Resultado con análisis de fraude
- `engine_changed` - Notificación de cambio de motor
- `engine_error` - Errores en tiempo real

---

## 🔒 Detección de Fraude

### **Características del Sistema**
- **Modelo ML**: Regresión logística entrenada
- **Threshold**: 60% de probabilidad para clasificar como fraude
- **Keywords**: Detección de palabras clave sospechosas
- **Tiempo Real**: Análisis inmediato de cada transcripción

### **Ejemplos de Detección**
- ✅ **Normal**: "Buenos días, ¿cómo está usted hoy?"
- ⚠️ **Fraude**: "Dinero fácil sin riesgo, ganancia garantizada"

### **Indicadores Visuales**
- 🟢 **Texto Normal**: Borde verde, probabilidad baja
- 🔴 **Fraude Detectado**: Borde rojo, probabilidad alta
- 📊 **Porcentaje**: Probabilidad de fraude mostrada

---

## 🚀 Características Técnicas

### **Arquitectura Modular**
- **Strategy Pattern**: Intercambio dinámico de algoritmos
- **Factory Pattern**: Creación automática de motores
- **Observer Pattern**: Notificaciones en tiempo real

### **Tecnologías Utilizadas**
- **Backend**: Python Flask + SocketIO
- **Frontend**: Bootstrap 5 + JavaScript
- **Audio**: SpeechRecognition + PyAudio
- **ML**: scikit-learn + joblib
- **Motores**: Google SR, Whisper, Silero, Vosk

### **Gestión de Recursos**
- **Inicialización Lazy**: Motores se cargan solo cuando se usan
- **Fallback Automático**: Si un motor falla, continúa con otro
- **Gestión de Memoria**: Limpieza automática de recursos

---

## 📄 Dependencias

### **Básicas** (requirements.txt)
```
Flask==3.1.2
Flask-SocketIO==5.5.1
SpeechRecognition==3.14.3
PyAudio==0.2.14
joblib==1.5.2
scikit-learn==1.6.1
numpy==2.3.2
scipy==1.16.1
```

### **Completas** (requirements_full.txt)
```
# Básicas + Motores adicionales
openai-whisper>=20231117
faster-whisper>=1.0.0
torch>=2.0.0
torchaudio>=2.0.0
silero>=0.4.1
vosk>=0.3.45
librosa>=0.10.0
soundfile>=0.12.0
transformers>=4.30.0
```

---

## 🎯 Instrucciones de Uso Rápido

### **Para Usuario Básico**
1. **Ejecutar**: `.\iniciar.bat`
2. **Abrir**: `http://localhost:5002`
3. **Configurar**: Ajustar audio según ambiente
4. **Usar**: Presionar "Iniciar" y hablar

### **Para Usuario Avanzado**
1. **Instalar Motores**: `.\instalar_motores.bat`
2. **Seleccionar Motor**: Elegir según necesidades
3. **Configurar**: Ajustar parámetros específicos
4. **Personalizar**: Descargar modelos adicionales

### **Para Desarrollador**
1. **Estudiar**: Arquitectura en `/engines/`
2. **Extender**: Crear nuevos motores
3. **Integrar**: Usar API REST/WebSocket
4. **Personalizar**: Modificar detección de fraude

---

## 📞 Soporte

- **Sistema Listo**: Funciona inmediatamente con motor básico
- **Instalación Simple**: Script automático para motores adicionales
- **Documentación**: Comentarios en código para desarrollo
- **Arquitectura Limpia**: Fácil de extender y mantener

---

**🎤 El sistema está listo para transcribir con múltiples motores de última generación!**