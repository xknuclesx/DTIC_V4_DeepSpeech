# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir al Transcriptor Modular Speech-to-Text!

## Cómo Contribuir

### 1. Fork del Repositorio

1. Haz fork del proyecto
2. Clona tu fork localmente:
```bash
git clone https://github.com/tu-usuario/DTIC_V4_DeepSpeech.git
cd DTIC_V4_DeepSpeech
```

### 2. Crear una Rama

```bash
git checkout -b feature/nueva-funcionalidad
# o
git checkout -b fix/correccion-bug
```

### 3. Configurar Entorno de Desarrollo

```powershell
# Crear y activar entorno virtual
python -m venv transcriptor_env
.\transcriptor_env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar dependencias de desarrollo (si existen)
pip install -r requirements-dev.txt
```

### 4. Realizar Cambios

- Escribe código limpio y documentado
- Sigue las convenciones de Python (PEP 8)
- Agrega comentarios donde sea necesario
- Actualiza la documentación si es relevante

### 5. Probar los Cambios

```powershell
# Ejecutar pruebas (si existen)
python -m pytest tests/

# Probar manualmente
python transcriptor.py
```

### 6. Commit y Push

```bash
git add .
git commit -m "feat: descripción clara de los cambios"
git push origin feature/nueva-funcionalidad
```

### 7. Crear Pull Request

1. Ve a tu fork en GitHub
2. Haz clic en "Pull Request"
3. Describe los cambios realizados
4. Espera revisión

## Convenciones de Código

### Estilo de Código

- Usa **4 espacios** para indentación
- Límite de **100 caracteres** por línea
- Nombres de variables en `snake_case`
- Nombres de clases en `PascalCase`
- Constantes en `UPPER_CASE`

### Mensajes de Commit

Usa el formato de [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` Nueva funcionalidad
- `fix:` Corrección de bug
- `docs:` Cambios en documentación
- `style:` Formato, sin cambios de código
- `refactor:` Refactorización de código
- `test:` Agregar o modificar tests
- `chore:` Tareas de mantenimiento

**Ejemplos:**
```
feat: agregar motor de transcripción Whisper
fix: corregir detección de micrófono en Windows
docs: actualizar README con instrucciones de instalación
```

## Agregar Nuevos Motores de Transcripción

Si deseas agregar un nuevo motor de transcripción:

### 1. Crear Clase del Motor

Crea un archivo en `engines/nuevo_engine.py`:

```python
from engines.base_engine import BaseTranscriptionEngine
import speech_recognition as sr

class NuevoEngine(BaseTranscriptionEngine):
    def __init__(self):
        super().__init__()
        self.engine_name = "Nuevo Motor"
        self.requires_internet = False  # o True
    
    def initialize(self) -> bool:
        """Inicializar el motor"""
        try:
            # Tu lógica de inicialización
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Error inicializando: {e}")
            return False
    
    def transcribe_audio(self, audio_data) -> str:
        """Transcribir audio"""
        try:
            # Tu lógica de transcripción
            text = "texto transcrito"
            return text
        except Exception as e:
            print(f"Error transcribiendo: {e}")
            return None
    
    def get_engine_info(self) -> dict:
        """Información del motor"""
        return {
            'name': self.engine_name,
            'version': '1.0.0',
            'initialized': self.is_initialized,
            'requires_internet': self.requires_internet
        }
```

### 2. Registrar en el Gestor

Edita `engine_manager.py` y agrega tu motor en `_load_engines()`:

```python
'nuevo_motor': {
    'module': 'engines.nuevo_engine',
    'class': 'NuevoEngine',
    'display_name': 'Nuevo Motor',
    'description': 'Descripción del nuevo motor',
    'requires_internet': False
}
```

### 3. Documentar

- Actualiza el README con información del nuevo motor
- Agrega instrucciones de instalación de dependencias
- Documenta configuraciones específicas

## Reportar Bugs

Para reportar un bug, crea un issue con:

1. **Descripción clara** del problema
2. **Pasos para reproducir** el bug
3. **Comportamiento esperado** vs **comportamiento actual**
4. **Información del sistema**:
   - Versión de Python
   - Sistema operativo
   - Versión del proyecto
5. **Logs o mensajes de error** (si hay)

## Solicitar Funcionalidades

Para solicitar una nueva funcionalidad:

1. Crea un issue con la etiqueta `enhancement`
2. Describe claramente la funcionalidad
3. Explica por qué sería útil
4. Sugiere una posible implementación (opcional)

## Código de Conducta

- Sé respetuoso con otros contribuidores
- Acepta críticas constructivas
- Enfócate en lo mejor para el proyecto
- Ayuda a otros miembros de la comunidad

## Preguntas

Si tienes dudas, puedes:

- Abrir un issue con la etiqueta `question`
- Contactar a los mantenedores del proyecto

---

**¡Gracias por contribuir!** 🎉
