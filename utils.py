"""
Utilidades para el sistema de recomendación
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def validate_data(courses_df: pd.DataFrame, courses_taken_df: pd.DataFrame) -> Dict:
    """
    Valida la consistencia de los datos de entrada
    
    Returns:
        Dict con resultados de validación
    """
    issues = []
    warnings = []
    
    # 1. Validar courses.csv
    required_cols_courses = ['course_code', 'course_name', 'prereq_codes', 'lineas_carrera']
    missing_cols = [col for col in required_cols_courses if col not in courses_df.columns]
    if missing_cols:
        issues.append(f"courses.csv falta columnas: {missing_cols}")
    
    # Verificar duplicados
    if courses_df['course_code'].duplicated().any():
        duplicates = courses_df[courses_df['course_code'].duplicated()]['course_code'].tolist()
        issues.append(f"Cursos duplicados en courses.csv: {duplicates}")
    
    # 2. Validar courses_taken.csv
    required_cols_taken = ['alumno', 'course_code', 'ciclo', 'grade']
    missing_cols = [col for col in required_cols_taken if col not in courses_taken_df.columns]
    if missing_cols:
        issues.append(f"courses_taken.csv falta columnas: {missing_cols}")
    
    # 3. Validar consistencia entre archivos
    courses_in_catalog = set(courses_df['course_code'])
    courses_taken = set(courses_taken_df['course_code'])
    
    unknown_courses = courses_taken - courses_in_catalog
    if unknown_courses:
        warnings.append(f"{len(unknown_courses)} cursos en courses_taken.csv no están en courses.csv")
        if len(unknown_courses) <= 10:
            warnings.append(f"  Ejemplos: {list(unknown_courses)[:10]}")
    
    # 4. Validar rangos de notas
    if 'grade' in courses_taken_df.columns:
        invalid_grades = courses_taken_df[
            (courses_taken_df['grade'] < 0) | (courses_taken_df['grade'] > 20)
        ]
        if len(invalid_grades) > 0:
            issues.append(f"{len(invalid_grades)} notas fuera del rango 0-20")
    
    # 5. Validar prerequisitos
    all_prereqs = set()
    for prereqs in courses_df['prereq_codes']:
        if isinstance(prereqs, list):
            all_prereqs.update(prereqs)
    
    invalid_prereqs = all_prereqs - courses_in_catalog
    if invalid_prereqs:
        warnings.append(f"{len(invalid_prereqs)} prerequisitos no existen en el catálogo")
    
    # 6. Estadísticas
    stats = {
        'num_courses': len(courses_df),
        'num_students': courses_taken_df['alumno'].nunique(),
        'num_records': len(courses_taken_df),
        'avg_courses_per_student': len(courses_taken_df) / courses_taken_df['alumno'].nunique(),
        'avg_grade': courses_taken_df['grade'].mean()
    }
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'stats': stats
    }


def print_validation_report(validation: Dict):
    """Imprime reporte de validación"""
    print("\n" + "="*70)
    print("REPORTE DE VALIDACIÓN DE DATOS")
    print("="*70)
    
    if validation['valid']:
        print("✓ Validación exitosa - No se encontraron errores críticos")
    else:
        print("✗ Validación fallida - Se encontraron errores:")
        for issue in validation['issues']:
            print(f"  ✗ {issue}")
    
    if validation['warnings']:
        print("\n⚠ Advertencias:")
        for warning in validation['warnings']:
            print(f"  ⚠ {warning}")
    
    print("\n📊 Estadísticas:")
    stats = validation['stats']
    print(f"  • Cursos en catálogo: {stats['num_courses']}")
    print(f"  • Estudiantes: {stats['num_students']}")
    print(f"  • Registros totales: {stats['num_records']}")
    print(f"  • Promedio cursos/estudiante: {stats['avg_courses_per_student']:.1f}")
    print(f"  • Nota promedio general: {stats['avg_grade']:.2f}")
    
    print("="*70 + "\n")


def analyze_student_performance(data_loader, student_id: str, pass_threshold: float = 10.0):
    """
    Analiza el desempeño de un estudiante
    """
    history = data_loader.get_student_history(student_id, pass_threshold)
    
    total_courses = len(history['all_courses'])
    passed_courses = len(history['passed_courses'])
    failed_courses = total_courses - passed_courses
    
    # Calcular promedio de aprobados
    grades_passed = [g for c, g in history['grades'].items() if g >= pass_threshold]
    avg_passed = np.mean(grades_passed) if grades_passed else 0
    
    # Análisis por línea de carrera
    lineas_performance = {}
    for course in history['passed_courses']:
        course_info = data_loader.get_course_info(course)
        if course_info and course_info['lineas_carrera']:
            for linea in course_info['lineas_carrera']:
                if linea not in lineas_performance:
                    lineas_performance[linea] = []
                grade = history['grades'].get(course, 0)
                lineas_performance[linea].append(grade)
    
    # Calcular promedio por línea
    lineas_avg = {
        linea: np.mean(grades) 
        for linea, grades in lineas_performance.items()
    }
    
    return {
        'student_id': student_id,
        'total_courses': total_courses,
        'passed_courses': passed_courses,
        'failed_courses': failed_courses,
        'pass_rate': (passed_courses / total_courses * 100) if total_courses > 0 else 0,
        'avg_grade_passed': avg_passed,
        'lineas_performance': lineas_avg,
        'best_linea': max(lineas_avg.items(), key=lambda x: x[1]) if lineas_avg else None,
        'worst_linea': min(lineas_avg.items(), key=lambda x: x[1]) if lineas_avg else None
    }


def print_student_analysis(analysis: Dict):
    """Imprime análisis detallado del estudiante"""
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE DESEMPEÑO: {analysis['student_id']}")
    print(f"{'='*70}")
    
    print(f"\n📚 Cursos:")
    print(f"  • Total cursados: {analysis['total_courses']}")
    print(f"  • Aprobados: {analysis['passed_courses']}")
    print(f"  • Reprobados: {analysis['failed_courses']}")
    print(f"  • Tasa de aprobación: {analysis['pass_rate']:.1f}%")
    
    print(f"\n📊 Rendimiento:")
    print(f"  • Promedio (aprobados): {analysis['avg_grade_passed']:.2f}")
    
    if analysis['lineas_performance']:
        print(f"\n🎯 Desempeño por Línea de Carrera:")
        sorted_lineas = sorted(
            analysis['lineas_performance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for linea, avg in sorted_lineas:
            print(f"  • {linea}: {avg:.2f}")
        
        if analysis['best_linea']:
            print(f"\n  ⭐ Mejor línea: {analysis['best_linea'][0]} ({analysis['best_linea'][1]:.2f})")
        if analysis['worst_linea']:
            print(f"  ⚠️  Línea a reforzar: {analysis['worst_linea'][0]} ({analysis['worst_linea'][1]:.2f})")
    
    print(f"{'='*70}\n")


def compare_students(data_loader, student_id1: str, student_id2: str, pass_threshold: float = 10.0):
    """
    Compara dos estudiantes para identificar similitudes
    """
    hist1 = data_loader.get_student_history(student_id1, pass_threshold)
    hist2 = data_loader.get_student_history(student_id2, pass_threshold)
    
    passed1 = set(hist1['passed_courses'])
    passed2 = set(hist2['passed_courses'])
    
    # Cursos en común
    common_courses = passed1 & passed2
    only_student1 = passed1 - passed2
    only_student2 = passed2 - passed1
    
    # Similitud (Jaccard)
    similarity = len(common_courses) / len(passed1 | passed2) if (passed1 | passed2) else 0
    
    return {
        'student1': student_id1,
        'student2': student_id2,
        'common_courses': list(common_courses),
        'only_student1': list(only_student1),
        'only_student2': list(only_student2),
        'similarity': similarity
    }


def export_recommendations_csv(recommendations: List[Dict], output_path: str):
    """
    Exporta recomendaciones a CSV
    """
    df = pd.DataFrame(recommendations)
    df.to_csv(output_path, index=False)
    print(f"✓ Recomendaciones exportadas a: {output_path}")


def get_curriculum_progress(data_loader, student_id: str, obligatory_courses: set, 
                           pass_threshold: float = 10.0):
    """
    Analiza el progreso curricular del estudiante
    """
    history = data_loader.get_student_history(student_id, pass_threshold)
    passed = set(history['passed_courses'])
    all_taken = set(history['all_courses'])
    
    # Obligatorios
    obligatory_passed = passed & obligatory_courses
    obligatory_failed = (all_taken - passed) & obligatory_courses
    obligatory_pending = obligatory_courses - all_taken
    
    # Porcentaje de avance
    total_obligatory = len(obligatory_courses)
    progress = len(obligatory_passed) / total_obligatory * 100 if total_obligatory > 0 else 0
    
    return {
        'student_id': student_id,
        'obligatory_passed': len(obligatory_passed),
        'obligatory_failed': len(obligatory_failed),
        'obligatory_pending': len(obligatory_pending),
        'total_obligatory': total_obligatory,
        'progress_percentage': progress,
        'failed_list': list(obligatory_failed),
        'pending_list': list(obligatory_pending)
    }


def print_curriculum_progress(progress: Dict):
    """Imprime progreso curricular"""
    print(f"\n{'='*70}")
    print(f"PROGRESO CURRICULAR: {progress['student_id']}")
    print(f"{'='*70}")
    
    print(f"\n📋 Cursos Obligatorios:")
    print(f"  • Total obligatorios: {progress['total_obligatory']}")
    print(f"  • Aprobados: {progress['obligatory_passed']}")
    print(f"  • Reprobados: {progress['obligatory_failed']}")
    print(f"  • Pendientes: {progress['obligatory_pending']}")
    print(f"  • Avance: {progress['progress_percentage']:.1f}%")
    
    if progress['failed_list']:
        print(f"\n🔴 Obligatorios Reprobados (prioridad alta):")
        for course in progress['failed_list']:
            print(f"  • {course}")
    
    if progress['pending_list']:
        print(f"\n⏳ Obligatorios Pendientes:")
        for course in progress['pending_list'][:10]:  # Mostrar primeros 10
            print(f"  • {course}")
        if len(progress['pending_list']) > 10:
            print(f"  ... y {len(progress['pending_list']) - 10} más")
    
    print(f"{'='*70}\n")