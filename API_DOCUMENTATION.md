# API REST - Sistema de Recomendación Híbrido

API REST para el sistema de recomendación de cursos académicos.


## Endpoints Disponibles

### Base URL

http://localhost:5000/api


## Estudiantes



### GET `/api/students`
Lista todos los estudiantes con paginación.

**Parámetros de Query:**
- `page` (int, opcional): Número de página (default: 1)
- `per_page` (int, opcional): Elementos por página (default: 50)

**Ejemplo:**
```bash
curl "http://localhost:5000/api/students?page=1&per_page=10"
```

**Respuesta:**
```json
{
  "students": ["ALUMNO_001", "ALUMNO_002", "..."],
  "total": 100,
  "page": 1,
  "per_page": 10,
  "total_pages": 10
}
```


---

### GET `/api/students/{student_id}`
Obtiene información detallada de un estudiante específico.

**Parámetros de Path:**
- `student_id` (string): ID del estudiante

**Ejemplo:**
```bash
curl http://localhost:5000/api/students/ALUMNO_REAL
```

**Respuesta:**
```json
{
  "student_id": "ALUMNO_REAL",
  "history": {
    "total_courses": 36,
    "passed_courses": 34,
    "courses": ["BMA01", "BFI01", "..."]
  },
  "performance": {
    "pass_rate": 94.4,
    "avg_grade": 12.45,
    "best_linea": ["Electrónica", 13.5],
    "worst_linea": ["Matemáticas", 11.2],
    "lineas_performance": {
      "Electrónica": 13.5,
      "Matemáticas": 11.2,
      "...": "..."
    }
  },
  "curriculum_progress": {
    "progress_percentage": 85.0,
    "obligatory_passed": 28,
    "obligatory_failed": 1,
    "obligatory_pending": 4,
    "failed_list": ["BMA20"]
  }
}
```





---

### GET `/api/students/{student_id}/history`
Obtiene el historial académico completo de un estudiante.

**Ejemplo:**
```bash
curl http://localhost:5000/api/students/ALUMNO_REAL/history
```

**Respuesta:**
```json
{
  "student_id": "ALUMNO_REAL",
  "all_courses": ["BMA01", "BFI01", "BMA02", "..."],
  "passed_courses": ["BMA01", "BFI01", "..."],
  "grades": {
    "BMA01": 12.2,
    "BFI01": 11.9,
    "BMA02": 10.5
  },
  "by_cycle": {
    "1": [
      {"course_code": "BMA01", "grade": 12.2},
      {"course_code": "BFI01", "grade": 11.9}
    ],
    "2": [
      {"course_code": "BMA02", "grade": 10.5}
    ]
  }
}
```





---

### GET `/api/students/{student_id}/recommendations`
Genera recomendaciones personalizadas para un estudiante.

**Parámetros de Query:**
- `top_k` (int, opcional): Número de recomendaciones (default: 10)

**Ejemplo:**
```bash
curl "http://localhost:5000/api/students/ALUMNO_REAL/recommendations?top_k=5"
```

**Respuesta:**
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
        "kg_neighbors": [
          ["BMA18", "prerequisito"],
          ["BMA15", "misma_linea"]
        ],
        "prerequisites": ["BMA18"],
        "prerequisites_met": true
      }
    },
    {
      "course_code": "CIB02",
      "course_name": "Circuitos Digitales",
      "score": 13.8921,
      "lineas_carrera": ["Electrónica", "Sistemas"],
      "is_failed": false,
      "is_obligatory": true,
      "priority": 2,
      "reasons": {
        "content_similarity": 0.654,
        "collaborative_score": 2.987,
        "lineas_performance": 0.612,
        "kg_neighbors": [],
        "prerequisites": [],
        "prerequisites_met": true
      }
    }
  ]
}
```






---


## Cursos



### GET `/api/courses`
Lista todos los cursos con filtros opcionales.

**Parámetros de Query:**
- `page` (int, opcional): Número de página (default: 1)
- `per_page` (int, opcional): Elementos por página (default: 50)
- `linea` (string, opcional): Filtrar por línea de carrera

**Ejemplo:**
```bash
curl "http://localhost:5000/api/courses?linea=Electrónica&per_page=5"
```

**Respuesta:**
```json
{
  "courses": [
    {
      "course_code": "EE320",
      "course_name": "Circuitos Eléctricos",
      "prereq_codes": ["BFI03", "BMA05"],
      "lineas_carrera": ["Electrónica", "Sistemas"]
    }
  ],
  "total": 150,
  "page": 1,
  "per_page": 5,
  "total_pages": 30
}
```






---

### GET `/api/courses/{course_code}`
Obtiene información detallada de un curso específico.

**Ejemplo:**
```bash
curl http://localhost:5000/api/courses/EE320
```

**Respuesta:**
```json
{
  "course_code": "EE320",
  "course_name": "Circuitos Eléctricos",
  "prereq_codes": ["BFI03", "BMA05"],
  "lineas_carrera": ["Electrónica", "Sistemas"],
  "statistics": {
    "num_students": 85,
    "avg_grade": 12.4,
    "pass_rate": 78.5,
    "difficulty": "Medio"
  },
  "related_courses": [
    {"course_code": "EE410", "relation": "siguiente"},
    {"course_code": "BFI03", "relation": "prerequisito"},
    {"course_code": "EE428", "relation": "misma_linea"}
  ]
}
```






### GET `/api/courses/{course_code}/students`
Obtiene lista de estudiantes que cursaron (y aprobaron) un curso.

**Ejemplo:**
```bash
curl http://localhost:5000/api/courses/BMA01/students
```

**Respuesta:**
```json
{
  "course_code": "BMA01",
  "students": ["ALUMNO_001", "ALUMNO_002", "..."],
  "total": 95
}
```





### POST `/api/recommend`
Genera recomendaciones personalizadas con parámetros adicionales.

**Body (JSON):**
```json
{
  "student_id": "ALUMNO_REAL",
  "top_k": 10
}
```

**Ejemplo:**
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"student_id": "ALUMNO_REAL", "top_k": 5}'
```

**Respuesta:**
```json
{
  "student_id": "ALUMNO_REAL",
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
        "lineas_performance": 0.720
      }
    }
  ]
}
```

---

## Estadísticas y Metadatos

### GET `/api/stats`
Obtiene estadísticas generales del sistema.

**Ejemplo:**
```bash
curl http://localhost:5000/api/stats
```

**Respuesta:**
```json
{
  "system": {
    "total_students": 100,
    "total_courses": 250,
    "total_records": 3500,
    "total_lineas": 15,
    "lineas": ["Base", "Matemáticas", "Electrónica", "..."]
  },
  "models": {
    "kg_embeddings": 365,
    "kg_nodes": 365,
    "kg_edges": 1250,
    "cf_factors": 64,
    "embedding_dim": 64
  },
  "data_quality": {
    "valid": true,
    "issues": [],
    "warnings": []
  }
}
```

---

### GET `/api/lineas`
Lista todas las líneas de carrera disponibles.

**Ejemplo:**
```bash
curl http://localhost:5000/api/lineas
```

**Respuesta:**
```json
{
  "lineas": ["Base", "Matemáticas", "Electrónica", "Sistemas", "..."],
  "total": 15,
  "courses_per_linea": {
    "Base": 35,
    "Matemáticas": 28,
    "Electrónica": 42,
    "Sistemas": 38
  }
}
```

---

### GET `/api/health`
Health check del sistema (no requiere modelos cargados).

**Ejemplo:**
```bash
curl http://localhost:5000/api/health
```

**Respuesta:**
```json
{
  "status": "online",
  "models_loaded": true,
  "version": "1.0.0"
}
```



---
## Notas

- La API usa **CORS** habilitado para permitir requests desde frontend
- Los modelos se cargan automáticamente al iniciar el servidor
- Las recomendaciones se generan en tiempo real usando los modelos cargados
- Todos los endpoints (excepto `/health`) requieren que los modelos estén cargados
---