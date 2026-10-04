@echo off
cd /d "%~dp0"
echo Avvio NIR Spectrum Manager...
py -m streamlit run app_streamlit.py
pause
