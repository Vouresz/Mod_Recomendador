#!/usr/bin/env python3
"""
Script rápido para verificar que los cursos obligatorios reprobados se recomiendan.
"""
from data_loader import DataLoader
from recommend import CourseRecommender
import yaml

# Cargar configuración
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

pass_threshold = config['pass_threshold']

# Cargar datos
data_loader = DataLoader('data/')
data_loader.load_courses()
data_loader.load_courses_taken()

# Crear recomendador (sin entrenar modelos complejos)
recommender = CourseRecommender(data_loader, None, None, None, None, None)

# Obtener primer estudiante
students = data_loader.get_all_students()
student_id = students[0]

print(f"\n=== Verificación de Cursos Obligatorios Reprobados ===")
print(f"Estudiante: {student_id}")
print(f"Threshold de aprobación: {pass_threshold}")

# Obtener historial del estudiante
history = data_loader.get_student_history(student_id)
print(f"\nTotal de cursos llevados: {len(history)}")

# Separar aprobados y reprobados
approved = set()
failed = set()
for course_id, grade in history.items():
    if grade >= pass_threshold:
        approved.add(course_id)
    else:
        failed.add(course_id)

print(f"Aprobados: {len(approved)}")
print(f"Reprobados: {len(failed)}")

# Ver cuáles de los reprobados son obligatorios
obligatory_failed = failed & recommender.OBLIGATORY_COURSES
print(f"\n*** Cursos Obligatorios Reprobados: {len(obligatory_failed)} ***")
if obligatory_failed:
    for course_id in sorted(obligatory_failed):
        grade = history[course_id]
        print(f"  - {course_id}: {grade:.1f}")

# Intentar recomendación simple (sin modelos ML)
print(f"\nIntentando recomendación simple...")

# Candidatos = cursos no tomados + obligatorios reprobados
all_courses = set(data_loader.courses.keys())
not_taken = all_courses - approved - failed
candidates = list(not_taken) + list(obligatory_failed)

print(f"Candidatos (no tomados + obligatorios reprobados): {len(candidates)}")
print(f"  - No tomados: {len(not_taken)}")
print(f"  - Obligatorios reprobados: {len(obligatory_failed)}")

# Mostrar top 10
print(f"\nTop 10 candidatos:")
for i, course_id in enumerate(sorted(candidates)[:10], 1):
    is_obligatory = course_id in recommender.OBLIGATORY_COURSES
    course = data_loader.courses.get(course_id, {})
    tipo = "[OBLIGATORIO]" if is_obligatory else "[Electivo]"
    print(f"  {i}. {course_id} {tipo} - {course.get('name', 'N/A')}")
