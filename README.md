# Carrera_empresas_2025
Dashboard Carrera de las Empresas 2025
Este proyecto es un dashboard interactivo de análisis de rendimiento para los participantes de la Carrera de las Empresas 2025.

Permite a cualquier corredor:
  - Analizar su rendimiento individual
  - Compararse con su categoría, su empresa o el conjunto de corredores
  - Visualizar distribuciones, percentiles y rankings de forma clara e intuitiva

# App en producción:
https://dashcarreraempresas2025.streamlit.app/

## Funcionalidades principales
### 1. Identificación del corredor
  - Introducción manual del nombre completo
  - Búsqueda tolerante a mayúsculas, tildes y nombres parciales
  - Detección de ambigüedad (varios corredores con nombre similar)

### 2. Análisis individual
  - Tiempo personal
  - Percentil dentro de:
  - Categoría (sexo + distancia)
  - General (misma distancia)
  - Empresa
  - Curva acumulada (CDF) interactiva

### 3. Distribuciones de tiempos
  - Histogramas y curvas suavizadas
  - Visualización en formato HH:MM:SS
  - Segmentación por:
    - Sexo
    - Distancia (5k / 10k)

### 4. Comparativas
  - Comparación entre empresas
  -Empresa vs global
  - Ranking interno dentro de la empresa
  - Top 10 + posición personal

# Detalles técnicos
  - Datos cargados de forma segura desde Google Sheets
  - Uso de st.secrets para proteger el acceso a la fuente de datos
  - Cacheo para mejorar rendimiento
  - Visualizaciones interactivas con Plotly

# Privacidad y uso de datos

La app expone datos publicos de https://www.carreradelasempresas.com/

No se permite la descarga masiva de datos

El objetivo es análisis agregado y visual, no redistribución de datos

# Tecnologías utilizadas

  - Python
  - Streamlit
  - Pandas / NumPy
  - Plotly
  - SciPy
  - Google Sheets (como backend de datos)

# Autor

Proyecto desarrollado por Santiago Puertas

Con fines analíticos, educativos y de visualización de datos deportivos.
