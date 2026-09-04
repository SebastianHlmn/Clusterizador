@echo off
setlocal
cd /d %~dp0
if not exist .venv (
  py -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run clusterizador_unidades_fiscales.py --server.address 0.0.0.0 --server.port 8501
endlocal
