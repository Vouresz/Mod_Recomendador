# Sistema de Recomendación Híbrido de Cursos Académicos

Sistema inteligente de recomendación de cursos que combina múltiples técnicas de Machine Learning para sugerir cursos personalizados a estudiantes basándose en su historial académico, rendimiento y relaciones entre cursos.

## Tabla de Contenidos

- [Características](#características)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
- [API REST](#api-rest)
- [Componentes del Sistema](#componentes-del-sistema)
- [Formato de Datos](#formato-de-datos)
- [Entrenamiento](#entrenamiento)
- [Evaluación](#evaluación)

## Características

- **Sistema Híbrido Multi-Modelo**: Combina Knowledge Graphs, Collaborative Filtering y Content-Based Filtering
- **Priorización Inteligente**: Identifica automáticamente cursos obligatorios reprobados y los prioriza
- **Verificación de Prerequisites**: Valida que el estudiante cumpla con los prerequisitos necesarios
- **Análisis de Líneas de Carrera**: Considera el desempeño del estudiante en diferentes áreas
- **API REST Completa**: Interfaz HTTP para integración con aplicaciones frontend
- **Explicabilidad**: Proporciona razones detalladas de cada recomendación

## Arquitectura del Sistema

El sistema está construido sobre tres modelos principales que se fusionan mediante una red neuronal:

```
┌──────────────────────────────────────────────────────┐
│                 SISTEMA HÍBRIDO                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐   │
│  │   Knowledge  │  │ Collaborative│  │  Content  │   │
│  │     Graph    │  │  Filtering   │  │   Based   │   │
│  │   (Node2Vec) │  │     (ALS)    │  │  (Líneas) │   │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘   │
│         │                 │                │         │
│         └─────────────────┼────────────────┘         │
│                           ▼                          │
│                  ┌─────────────────┐                 │
│                  │  MLP Fusión     │                 │
│                  │  (Deep Learning)│                 │
│                  └─────────────────┘                 │
│                           │                          │
│                           ▼                          │
│                  Score Final + Reglas                │
└──────────────────────────────────────────────────────┘
```

### Flujo de Recomendación

1. **Entrada**: ID del estudiante + Top K recomendaciones solicitadas
2. **Filtrado**: Excluir cursos ya aprobados, verificar prerequisites
3. **Scoring Multi-Modelo**:
   - Knowledge Graph: Embeddings de cursos relacionados
   - Collaborative Filtering: Preferencias de estudiantes similares
   - Content-Based: Similitud con líneas de carrera del estudiante
4. **Fusión**: Red neuronal combina los scores
5. **Priorización**: Boost para cursos obligatorios reprobados
6. **Salida**: Top K cursos ordenados con explicaciones

## Requisitos

### Software

- Python 3.8+
- pip (gestor de paquetes)

### Dependencias Python

Ver `requirements.txt`:


## Instalación


### Crear entorno conda

```bash
conda create -n recomendador python=3.8
conda activate recomendador
```

### Instalar dependencias

Usando pip:
```bash
pip install -r requirements.txt
```

###  Preparar datos

Colocar los archivos de datos en el directorio `data/`:

- `courses.csv`: Catálogo de cursos
- `courses_taken.csv`: Historial académico de estudiantes

## Estructura del Proyecto

```
Mod_Recomendador/
├── data/
│   ├── courses.csv              # Catálogo de cursos
│   └── courses_taken.csv        # Historial de estudiantes
├── configs/
│   └── config.yaml              # Configuración del sistema
├── models/                      # Modelos entrenados (generado)
│   ├── kg_model.pkl
│   ├── cf_model.pkl
│   ├── content_model.pkl
│   ├── hybrid_model.pt
│   └── preprocessor.pkl
├── data_loader.py               # Carga y validación de datos
├── preprocess.py                # Preprocesamiento de datos
├── kg_builder.py                # Construcción del Knowledge Graph
├── cf_model.py                  # Collaborative Filtering (ALS)
├── content_model.py             # Content-Based Filtering
├── hybrid_model.py              # Modelo híbrido (MLP)
├── recommend.py                 # Sistema de recomendación final
├── utils.py                     # Funciones auxiliares
├── train.py                     # Pipeline de entrenamiento
├── demo.py                      # Demostración del sistema
├── get_recs.py                  # Script para obtener recomendaciones
├── api.py                       # API REST
├── test_api.py                  # Cliente de prueba para API
├── requirements.txt             # Dependencias Python
└── README.md                    # Este archivo
```

## Uso

### Activar Entorno Conda

Antes de ejecutar cualquier comando, asegúrate de activar el entorno:

```bash
conda activate recomendador
```

### Entrenar el Sistema

```bash
python train.py
```

Esto ejecutará el pipeline completo:
1. Carga de datos
2. Preprocesamiento
3. Construcción del Knowledge Graph
4. Entrenamiento de Collaborative Filtering
5. Preparación de Content-Based
6. Entrenamiento del modelo híbrido
7. Guardado de modelos

### Demo Interactivo

```bash
python demo.py
```

Muestra el funcionamiento completo del sistema con ejemplos.

### Obtener Recomendaciones

```bash
python get_recs.py --top_k 5
```

Por defecto, el sistema analiza al estudiante `ALUMNO_REAL`.

Parámetros:
- `student_id`: ID del estudiante (opcional, default: ALUMNO_REAL)
- `--top_k`: Número de recomendaciones (default: 5)
- `--models_dir`: Directorio de modelos (default: models)

### Iniciar API REST

```bash
python api.py --host 0.0.0.0 --port 5000
```

Parámetros:
- `--host`: Host del servidor (default: 0.0.0.0)
- `--port`: Puerto del servidor (default: 5000)
- `--models-dir`: Directorio de modelos (default: models)
- `--data-dir`: Directorio de datos (default: data)
- `--debug`: Modo debug

## API REST

La API REST proporciona endpoints completos para interactuar con el sistema.

### Endpoints Principales

#### Health Check
```bash
GET /api/health
```

#### Estudiantes
```bash
GET  /api/students                        # Listar estudiantes
GET  /api/students/{id}                   # Info de estudiante
GET  /api/students/{id}/history           # Historial académico
GET  /api/students/{id}/recommendations   # Recomendaciones
```

#### Cursos
```bash
GET  /api/courses                         # Listar cursos
GET  /api/courses/{code}                  # Info de curso
GET  /api/courses/{code}/students         # Estudiantes del curso
```

#### Recomendación
```bash
POST /api/recommend                       # Recomendación personalizada
```

#### Sistema
```bash
GET  /api/stats                           # Estadísticas del sistema
GET  /api/lineas                          # Líneas de carrera
```

### Ejemplo de Uso

```bash
# Obtener recomendaciones para ALUMNO_REAL
curl "http://localhost:5000/api/students/ALUMNO_REAL/recommendations?top_k=5"
```

Respuesta:
```json
{
  "student_id": "ALUMNO_REAL",
  "top_k": 5,
  "recommendations": [
    {
      "course_code": "BMA20",
      "course_name": "Matemáticas Avanzadas",
      "score": 15.2341,
      "lineas_carrera": ["Base", "Matemáticas"],
      "is_failed": true,
      "is_obligatory": true,
      "priority": 1,
      "reasons": {
        "content_similarity": 0.856,
        "collaborative_score": 3.124,
        "lineas_performance": 0.720,
        "kg_neighbors": [["BMA18", "prerequisito"]],
        "prerequisites": ["BMA18"],
        "prerequisites_met": true
      }
    }
  ]
}
```

Ver documentación completa en `API_DOCUMENTATION.md`.

## Componentes del Sistema

### 1. DataLoader (`data_loader.py`)

Carga y valida los datos de entrada.

**Funciones principales:**
- `load_courses()`: Carga catálogo de cursos
- `load_courses_taken()`: Carga historial de estudiantes
- `get_student_history()`: Obtiene historial completo de un estudiante
- `get_course_info()`: Información detallada de un curso
- `validate_data_consistency()`: Valida consistencia de datos

### 2. DataPreprocessor (`preprocess.py`)

Preprocesa datos para los modelos.

**Funcionalidades:**
- Construcción de índices estudiante-curso
- Codificación de líneas de carrera (MultiLabelBinarizer)
- Construcción de matriz de interacciones
- Generación de perfiles de estudiantes

### 3. KnowledgeGraphBuilder (`kg_builder.py`)

Construye y entrena el Knowledge Graph.

**Características:**
- Grafo heterogéneo: Estudiantes, Cursos, Líneas de Carrera
- Relaciones: BELONGS_TO, HAS_PREREQ, TOOK
- Node2Vec: Embeddings mediante random walks
- Vecinos contextuales para explicabilidad

### 4. CollaborativeFilteringModel (`cf_model.py`)

Filtrado colaborativo usando ALS (Alternating Least Squares).

**Características:**
- Factorización de matriz de interacciones
- Embeddings latentes para usuarios e items
- Predicción de afinidad estudiante-curso

### 5. ContentBasedModel (`content_model.py`)

Filtrado basado en contenido usando líneas de carrera.

**Características:**
- Vectorización de líneas de carrera
- Perfil de estudiante basado en cursos aprobados
- Similitud coseno para recomendaciones

### 6. HybridRecommenderModel (`hybrid_model.py`)

Fusión de modelos mediante Deep Learning.

**Arquitectura:**
```python
Input: [KG_emb, CF_emb, Content_emb] × 2 (estudiante + curso)
       ↓
Linear(input_dim, 128) + ReLU + BatchNorm + Dropout(0.3)
       ↓
Linear(128, 64) + ReLU + BatchNorm + Dropout(0.3)
       ↓
Linear(64, 32) + ReLU
       ↓
Linear(32, 1) → Score
```

### 7. CourseRecommender (`recommend.py`)

Sistema completo de recomendación con reglas de negocio.

**Lógica de Priorización:**
1. **Prioridad 1**: Cursos obligatorios reprobados (+2.0 boost)
2. **Prioridad 2**: Cursos obligatorios no cursados (+1.0 boost)
3. **Prioridad 3**: Cursos electivos y otros

**Reglas:**
- Verificación de prerequisites
- Exclusión de cursos aprobados
- Boost por desempeño en líneas relacionadas
- Explicaciones detalladas por recomendación

## Formato de Datos

### courses.csv

```csv
course_code,course_name,prereq_codes,lineas_carrera
BMA01,Matemáticas I,,Base;Matemáticas
BMA02,Matemáticas II,BMA01,Base;Matemáticas
EE320,Circuitos Eléctricos,BFI03;BMA05,Electrónica;Sistemas
```

**Columnas:**
- `course_code`: Código único del curso
- `course_name`: Nombre del curso
- `prereq_codes`: Prerequisites separados por `;` (vacío si no tiene)
- `lineas_carrera`: Líneas de carrera separadas por `;`

### courses_taken.csv

```csv
alumno,course_code,cycle,grade
ALUMNO_REAL,BMA01,1,15.5
ALUMNO_REAL,BFI01,1,14.2
ALUMNO_REAL,BMA02,2,16.8
```

**Columnas:**
- `alumno`: ID del estudiante
- `course_code`: Código del curso
- `cycle`: Ciclo académico
- `grade`: Nota obtenida (0-20)

## Entrenamiento

### Configuración (`configs/config.yaml`)

```yaml
data:
  data_dir: "data/"
  
training:
  pass_threshold: 11.0    # Nota mínima para aprobar
  
kg:
  embedding_dim: 64       # Dimensión de embeddings
  walk_length: 30         # Longitud de random walks
  num_walks: 200          # Número de walks por nodo
  
cf:
  factors: 64             # Factores latentes en ALS
  
hybrid:
  epochs: 5               # Épocas de entrenamiento
  batch_size: 32          # Tamaño de batch
  learning_rate: 0.001    # Learning rate
  
output:
  models_dir: "models/"
```

### Pipeline de Entrenamiento

```python
from train import TrainingPipeline

pipeline = TrainingPipeline("configs/config.yaml")
pipeline.run_full_pipeline()
```

**Pasos:**
1. Carga de datos → DataLoader
2. Preprocesamiento → Índices y encodings
3. Knowledge Graph → Construcción + Node2Vec
4. Collaborative Filtering → ALS
5. Content-Based → Vectorización de líneas
6. Hybrid Model → Entrenamiento MLP
7. Guardado → Modelos serializados

##  Evaluación

### Métricas del Sistema

El sistema proporciona:

- **Precision@K**: Precisión de las top-K recomendaciones
- **Recall@K**: Cobertura de cursos relevantes
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Coverage**: Porcentaje de cursos recomendables
- **Diversity**: Diversidad de líneas de carrera en recomendaciones

### Análisis de Estudiante

```python
from utils import analyze_student_performance

analysis = analyze_student_performance(data_loader, "ALUMNO_REAL")
# Retorna: pass_rate, avg_grade, lineas_performance, best/worst_linea
```

### Progreso Curricular

```python
from utils import get_curriculum_progress
from recommend import CourseRecommender

progress = get_curriculum_progress(
    data_loader, 
    "ALUMNO_REAL",
    CourseRecommender.OBLIGATORY_COURSES
)
# Retorna: progress_percentage, obligatory_passed/failed/pending
```

### Prueba de API

```bash
python test_api.py
```
