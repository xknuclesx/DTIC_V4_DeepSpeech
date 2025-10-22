# 📦 Guía de Instalación

## Requisitos Previos

- **Python 3.8 o superior**
- **Windows 10/11** (con PowerShell)
- **Micrófono** funcional
- **4GB RAM** mínimo (8GB recomendado para Whisper)

## Instalación Paso a Paso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/DTIC_V4_DeepSpeech.git
cd DTIC_V4_DeepSpeech
```

### 2. Crear Entorno Virtual

```powershell
# Crear entorno virtual
python -m venv transcriptor_env

# Activar entorno virtual
.\transcriptor_env\Scripts\activate
```

### 3. Instalar Dependencias Básicas

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Descargar Modelos de Machine Learning

Los modelos de detección de fraude (`best_model_lr.joblib` y `vectorizer_tfidf.joblib`) deben ser descargados por separado debido a su tamaño.

**Opción 1: Descargar desde releases**
- Ir a la sección de [Releases](https://github.com/tu-usuario/DTIC_V4_DeepSpeech/releases)
- Descargar `ml_models.zip`
- Extraer en la raíz del proyecto

**Opción 2: Entrenar tus propios modelos**
```python
# Ejecutar script de entrenamiento (si está disponible)
python train_fraud_detector.py
```

### 5. (Opcional) Instalar Motores Adicionales

Para instalar Whisper, Silero y Vosk:

```powershell
# Instalar dependencias adicionales
pip install openai-whisper
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install vosk

# Para Silero (experimental)
pip install omegaconf
```

### 6. (Opcional) Descargar Modelos de Vosk

Si deseas usar el motor Vosk:

1. Visita https://alphacephei.com/vosk/models
2. Descarga el modelo para español: `vosk-model-small-es-0.42`
3. Extrae el contenido en la carpeta `models/`

```powershell
# Estructura esperada:
# models/
#   └── vosk-model-small-es-0.42/
#       ├── am/
#       ├── conf/
#       ├── graph/
#       └── ivector/
```

## Verificación de Instalación

```powershell
# Activar entorno
.\transcriptor_env\Scripts\activate

# Verificar instalación
python -c "import flask, flask_socketio, speech_recognition; print('✅ Instalación correcta')"
```

## Ejecutar el Sistema

```powershell
# Activar entorno virtual
.\transcriptor_env\Scripts\activate

# Ejecutar servidor
python transcriptor.py
```

Abrir navegador en: **http://localhost:5003**

## Problemas Comunes

### Error: PyAudio no se instala

```powershell
# Windows: Instalar desde archivo wheel
pip install pipwin
pipwin install pyaudio
```

### Error: No se encuentra el micrófono

1. Verificar permisos de micrófono en Configuración de Windows
2. Comprobar que no esté siendo usado por otra aplicación
3. Reiniciar el sistema

### Error: Modelos ML no encontrados

Asegúrate de tener los archivos:
- `best_model_lr.joblib`
- `vectorizer_tfidf.joblib`

En la raíz del proyecto.

## Desinstalación

```powershell
# Desactivar entorno virtual
deactivate

# Eliminar entorno virtual
Remove-Item -Recurse -Force transcriptor_env

# Eliminar archivos compilados
Get-ChildItem -Recurse __pycache__ | Remove-Item -Recurse -Force
```

---

**¿Necesitas ayuda?** Abre un issue en el repositorio de GitHub.
