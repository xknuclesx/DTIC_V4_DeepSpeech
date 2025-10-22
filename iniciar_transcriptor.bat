@echo off
echo ========================================
echo   Transcriptor Modular Speech-to-Text
echo ========================================
echo.

REM Verificar si el entorno virtual existe
if not exist "transcriptor_env\Scripts\activate.bat" (
    echo [ERROR] Entorno virtual no encontrado.
    echo [INFO] Por favor, ejecuta primero la instalacion:
    echo        python -m venv transcriptor_env
    echo        .\transcriptor_env\Scripts\activate
    echo        pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activar entorno virtual
echo [INFO] Activando entorno virtual...
call transcriptor_env\Scripts\activate.bat

REM Verificar que Python este disponible
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado en el entorno virtual.
    pause
    exit /b 1
)

REM Verificar que los modelos ML existan
if not exist "best_model_lr.joblib" (
    echo [ADVERTENCIA] Modelo ML de fraude no encontrado.
    echo [INFO] El sistema funcionara sin deteccion de fraude.
    echo.
)

REM Ejecutar el servidor
echo [INFO] Iniciando servidor en http://localhost:5003
echo [INFO] Presiona Ctrl+C para detener el servidor
echo.
python transcriptor.py

REM Si el servidor se detiene
echo.
echo [INFO] Servidor detenido.
pause
