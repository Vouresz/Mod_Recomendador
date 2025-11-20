"""
Cliente de prueba para la API REST del Sistema de Recomendación

Uso:
  python test_api.py
"""

import requests
import json
from typing import Dict, List


class RecommenderAPIClient:
    """Cliente para interactuar con la API de recomendación"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
    
    def health_check(self) -> Dict:
        """Verifica el estado de la API"""
        response = requests.get(f"{self.api_url}/health")
        return response.json()
    
    def get_students(self, page: int = 1, per_page: int = 50) -> Dict:
        """Obtiene lista de estudiantes"""
        params = {'page': page, 'per_page': per_page}
        response = requests.get(f"{self.api_url}/students", params=params)
        return response.json()
    
    def get_student(self, student_id: str) -> Dict:
        """Obtiene información de un estudiante"""
        response = requests.get(f"{self.api_url}/students/{student_id}")
        return response.json()
    
    def get_student_history(self, student_id: str) -> Dict:
        """Obtiene historial académico"""
        response = requests.get(f"{self.api_url}/students/{student_id}/history")
        return response.json()
    
    def get_recommendations(self, student_id: str, top_k: int = 10) -> Dict:
        """Obtiene recomendaciones para un estudiante"""
        params = {'top_k': top_k}
        response = requests.get(
            f"{self.api_url}/students/{student_id}/recommendations",
            params=params
        )
        return response.json()
    
    def get_courses(self, page: int = 1, per_page: int = 50, linea: str = None) -> Dict:
        """Obtiene lista de cursos"""
        params = {'page': page, 'per_page': per_page}
        if linea:
            params['linea'] = linea
        response = requests.get(f"{self.api_url}/courses", params=params)
        return response.json()
    
    def get_course(self, course_code: str) -> Dict:
        """Obtiene información de un curso"""
        response = requests.get(f"{self.api_url}/courses/{course_code}")
        return response.json()
    
    def recommend_custom(self, student_id: str, top_k: int = 10) -> Dict:
        """Recomendación personalizada vía POST"""
        data = {
            'student_id': student_id,
            'top_k': top_k
        }
        response = requests.post(f"{self.api_url}/recommend", json=data)
        return response.json()
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del sistema"""
        response = requests.get(f"{self.api_url}/stats")
        return response.json()
    
    def get_lineas(self) -> Dict:
        """Obtiene líneas de carrera"""
        response = requests.get(f"{self.api_url}/lineas")
        return response.json()


def print_section(title: str):
    """Imprime sección decorada"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")


def test_api():
    """Prueba todos los endpoints de la API"""
    
    client = RecommenderAPIClient()
    
    print_section("🧪 PRUEBA DE API - SISTEMA DE RECOMENDACIÓN")
    
    # 1. Health Check
    print_section("1️⃣  HEALTH CHECK")
    try:
        health = client.health_check()
        print(f"Estado: {health['status']}")
        print(f"Modelos cargados: {health['models_loaded']}")
        print(f"Versión: {health['version']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Asegúrate de que la API esté corriendo: python api.py")
        return
    
    # 2. Listar estudiantes
    print_section("2️⃣  LISTAR ESTUDIANTES")
    students_data = client.get_students(page=1, per_page=5)
    print(f"Total de estudiantes: {students_data['total']}")
    print(f"Primeros {len(students_data['students'])} estudiantes:")
    for student in students_data['students']:
        print(f"  • {student}")
    
    # Seleccionar estudiante de prueba
    if not students_data['students']:
        print("❌ No hay estudiantes en la base de datos")
        return
    
    test_student = students_data['students'][0]
    print(f"\n✅ Usando estudiante de prueba: {test_student}")
    
    # 3. Información del estudiante
    print_section(f"3️⃣  INFORMACIÓN DE {test_student}")
    student_info = client.get_student(test_student)
    print(f"Cursos cursados: {student_info['history']['total_courses']}")
    print(f"Cursos aprobados: {student_info['history']['passed_courses']}")
    print(f"Tasa de aprobación: {student_info['performance']['pass_rate']:.1f}%")
    print(f"Promedio: {student_info['performance']['avg_grade']:.2f}")
    print(f"\nProgreso curricular: {student_info['curriculum_progress']['progress_percentage']:.1f}%")
    print(f"Obligatorios aprobados: {student_info['curriculum_progress']['obligatory_passed']}")
    print(f"Obligatorios reprobados: {student_info['curriculum_progress']['obligatory_failed']}")
    
    # 4. Historial académico
    print_section(f"4️⃣  HISTORIAL ACADÉMICO DE {test_student}")
    history = client.get_student_history(test_student)
    print(f"Total de cursos: {len(history['all_courses'])}")
    print(f"Cursos aprobados: {len(history['passed_courses'])}")
    print(f"\nÚltimos 5 cursos:")
    for course in history['all_courses'][-5:]:
        grade = history['grades'].get(course, 0)
        status = "✅" if grade >= 10 else "❌"
        print(f"  {status} {course}: {grade}")
    
    # 5. Recomendaciones
    print_section(f"5️⃣  RECOMENDACIONES PARA {test_student}")
    recs = client.get_recommendations(test_student, top_k=5)
    print(f"Top {len(recs['recommendations'])} cursos recomendados:\n")
    
    for i, rec in enumerate(recs['recommendations'], 1):
        tipo = "📌 REPROBADO" if rec['is_failed'] else "⚠️  OBLIGATORIO" if rec['is_obligatory'] else "✓  Electivo"
        print(f"{i}. {rec['course_code']} - {tipo}")
        print(f"   Score: {rec['score']}")
        print(f"   Líneas: {', '.join(rec['lineas_carrera'])}")
        print(f"   Similitud contenido: {rec['reasons']['content_similarity']}")
        print(f"   Score colaborativo: {rec['reasons']['collaborative_score']}")
        print()
    
    # 6. Listar cursos
    print_section("6️⃣  LISTAR CURSOS")
    courses = client.get_courses(page=1, per_page=5)
    print(f"Total de cursos: {courses['total']}")
    print(f"Primeros {len(courses['courses'])} cursos:")
    for course in courses['courses']:
        print(f"  • {course['course_code']}: {course['course_name']}")
    
    # 7. Información de curso específico
    if courses['courses']:
        test_course = courses['courses'][0]['course_code']
        print_section(f"7️⃣  INFORMACIÓN DEL CURSO {test_course}")
        course_info = client.get_course(test_course)
        print(f"Nombre: {course_info['course_name']}")
        print(f"Prerequisitos: {', '.join(course_info['prereq_codes']) if course_info['prereq_codes'] else 'Ninguno'}")
        print(f"Líneas: {', '.join(course_info['lineas_carrera'])}")
        print(f"\nEstadísticas:")
        print(f"  • Estudiantes: {course_info['statistics']['num_students']}")
        print(f"  • Promedio: {course_info['statistics']['avg_grade']}")
        print(f"  • Tasa aprobación: {course_info['statistics']['pass_rate']}%")
        print(f"  • Dificultad: {course_info['statistics']['difficulty']}")
    
    # 8. Estadísticas del sistema
    print_section("8️⃣  ESTADÍSTICAS DEL SISTEMA")
    stats = client.get_stats()
    print("Sistema:")
    print(f"  • Estudiantes: {stats['system']['total_students']}")
    print(f"  • Cursos: {stats['system']['total_courses']}")
    print(f"  • Registros: {stats['system']['total_records']}")
    print(f"  • Líneas de carrera: {stats['system']['total_lineas']}")
    print(f"\nModelos:")
    print(f"  • KG embeddings: {stats['models']['kg_embeddings']}")
    print(f"  • KG nodos: {stats['models']['kg_nodes']}")
    print(f"  • KG aristas: {stats['models']['kg_edges']}")
    print(f"  • CF factores: {stats['models']['cf_factors']}")
    
    # 9. Líneas de carrera
    print_section("9️⃣  LÍNEAS DE CARRERA")
    lineas = client.get_lineas()
    print(f"Total de líneas: {lineas['total']}")
    print("Líneas disponibles:")
    for linea in lineas['lineas']:
        count = lineas['courses_per_linea'][linea]
        print(f"  • {linea}: {count} cursos")
    
    # 10. Recomendación POST
    print_section("🔟 RECOMENDACIÓN VÍA POST")
    custom_recs = client.recommend_custom(test_student, top_k=3)
    print(f"Top 3 recomendaciones para {test_student}:")
    for i, rec in enumerate(custom_recs['recommendations'], 1):
        print(f"{i}. {rec['course_code']} (Score: {rec['score']})")
    
    print_section("✅ TODAS LAS PRUEBAS COMPLETADAS")


if __name__ == '__main__':
    test_api()