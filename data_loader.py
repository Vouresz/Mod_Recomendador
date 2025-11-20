import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from pathlib import Path


class DataLoader:
    """Carga y valida los datos de cursos y estudiantes"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.courses = None
        self.courses_taken = None
        
    def load_courses(self) -> pd.DataFrame:
        """
        Carga courses.csv y procesa prerequisitos y líneas de carrera
        
        Returns:
            DataFrame con información de cursos
        """
        courses_path = self.data_dir / "courses.csv"
        if not courses_path.exists():
            raise FileNotFoundError(f"No se encuentra: {courses_path}")
        
        df = pd.read_csv(courses_path)
        
        # Validar columnas requeridas
        required_cols = ['course_code', 'course_name', 'prereq_codes', 'lineas_carrera']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"courses.csv falta columnas: {missing}")
        
        # Procesar prerequisitos (separados por ;)
        df['prereq_codes'] = df['prereq_codes'].fillna('').apply(
            lambda x: [p.strip() for p in str(x).split(';') if p.strip()]
        )
        
        # Procesar líneas de carrera (separadas por ;)
        df['lineas_carrera'] = df['lineas_carrera'].fillna('').apply(
            lambda x: [l.strip() for l in str(x).split(';') if l.strip()]
        )
        
        # Validar códigos únicos
        if df['course_code'].duplicated().any():
            duplicates = df[df['course_code'].duplicated()]['course_code'].tolist()
            raise ValueError(f"Cursos duplicados en courses.csv: {duplicates}")
        
        self.courses = df
        print(f"✓ courses.csv cargado: {len(df)} cursos")
        return df
    
    def load_courses_taken(self) -> pd.DataFrame:
        """
        Carga courses_taken.csv con historial de estudiantes
        
        Returns:
            DataFrame con registros de cursos tomados
        """
        taken_path = self.data_dir / "courses_taken.csv"
        if not taken_path.exists():
            raise FileNotFoundError(f"No se encuentra: {taken_path}")
        
        df = pd.read_csv(taken_path)
        
        # Validar columnas requeridas
        required_cols = ['alumno', 'course_code', 'cycle', 'grade']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"courses_taken.csv falta columnas: {missing}")
        
        # Convertir tipos
        df['cycle'] = df['cycle'].astype(int)
        df['grade'] = df['grade'].astype(float)
        
        # Validar rangos de notas
        invalid_grades = df[(df['grade'] < 0) | (df['grade'] > 20)]
        if len(invalid_grades) > 0:
            print(f"⚠️  Advertencia: {len(invalid_grades)} notas fuera del rango 0-20")
        
        # Ordenar por alumno y ciclo
        df = df.sort_values(['alumno', 'cycle']).reset_index(drop=True)
        
        self.courses_taken = df
        print(f"✓ courses_taken.csv cargado: {len(df)} registros")
        return df
    
    def get_student_history(self, student_id: str, 
                           pass_threshold: float = 10.0) -> Dict:
        """
        Obtiene historial académico completo de un estudiante
        
        Args:
            student_id: ID del estudiante
            pass_threshold: Nota mínima para aprobar (default: 10.0)
            
        Returns:
            Dict con:
                - student_id: ID del estudiante
                - all_courses: Lista de todos los cursos cursados
                - passed_courses: Lista de cursos aprobados
                - grades: Dict {course_code: grade}
                - by_cycle: Dict {ciclo: [courses]}
        """
        if self.courses_taken is None:
            raise RuntimeError("Debes llamar load_courses_taken() primero")
        
        student_data = self.courses_taken[
            self.courses_taken['alumno'] == student_id
        ]
        
        if len(student_data) == 0:
            # Estudiante sin registros
            return {
                'student_id': student_id,
                'all_courses': [],
                'passed_courses': [],
                'grades': {},
                'by_cycle': {}
            }
        
        # Todos los cursos (incluyendo repetidos)
        all_courses = student_data['course_code'].tolist()
        
        # Para cursos repetidos, tomar la mejor nota
        best_grades = student_data.groupby('course_code')['grade'].max()
        
        # Cursos aprobados (con mejor nota >= threshold)
        passed_courses = best_grades[best_grades >= pass_threshold].index.tolist()
        
        # Dict de notas (mejor nota por curso)
        grades = best_grades.to_dict()
        
        # Cursos por ciclo
        by_cycle = {}
        for _, row in student_data.iterrows():
            cycle = row['cycle']
            if cycle not in by_cycle:
                by_cycle[cycle] = []
            by_cycle[cycle].append({
                'course_code': row['course_code'],
                'grade': row['grade']
            })
        
        return {
            'student_id': student_id,
            'all_courses': all_courses,
            'passed_courses': passed_courses,
            'grades': grades,
            'by_cycle': by_cycle
        }
    
    def get_all_students(self) -> List[str]:
        """
        Obtiene lista única de todos los estudiantes
        
        Returns:
            Lista de IDs de estudiantes
        """
        if self.courses_taken is None:
            raise RuntimeError("Debes llamar load_courses_taken() primero")
        
        return sorted(self.courses_taken['alumno'].unique().tolist())
    
    def get_all_courses(self) -> List[str]:
        """
        Obtiene lista única de todos los cursos en el catálogo
        
        Returns:
            Lista de códigos de cursos
        """
        if self.courses is None:
            raise RuntimeError("Debes llamar load_courses() primero")
        
        return self.courses['course_code'].tolist()
    
    def get_course_info(self, course_code: str) -> Dict:
        """
        Obtiene información detallada de un curso específico
        
        Args:
            course_code: Código del curso
            
        Returns:
            Dict con información del curso o None si no existe
        """
        if self.courses is None:
            raise RuntimeError("Debes llamar load_courses() primero")
        
        course = self.courses[self.courses['course_code'] == course_code]
        
        if course.empty:
            return None
        
        course_row = course.iloc[0]
        
        return {
            'course_code': course_code,
            'course_name': course_row.get('course_name', ''),
            'prereq_codes': course_row['prereq_codes'],
            'lineas_carrera': course_row['lineas_carrera']
        }
    
    def get_students_who_took_course(self, course_code: str, 
                                    pass_threshold: float = 10.0) -> List[str]:
        """
        Obtiene estudiantes que tomaron (y aprobaron) un curso
        
        Args:
            course_code: Código del curso
            pass_threshold: Nota mínima para considerar "aprobado"
            
        Returns:
            Lista de IDs de estudiantes
        """
        if self.courses_taken is None:
            raise RuntimeError("Debes llamar load_courses_taken() primero")
        
        # Filtrar por curso y nota
        students = self.courses_taken[
            (self.courses_taken['course_code'] == course_code) &
            (self.courses_taken['grade'] >= pass_threshold)
        ]['alumno'].unique().tolist()
        
        return students
    
    def get_course_statistics(self, course_code: str) -> Dict:
        """
        Obtiene estadísticas de un curso
        
        Args:
            course_code: Código del curso
            
        Returns:
            Dict con estadísticas del curso
        """
        if self.courses_taken is None:
            raise RuntimeError("Debes llamar load_courses_taken() primero")
        
        course_data = self.courses_taken[
            self.courses_taken['course_code'] == course_code
        ]
        
        if len(course_data) == 0:
            return {
                'course_code': course_code,
                'num_students': 0,
                'avg_grade': 0,
                'pass_rate': 0,
                'difficulty': 'N/A'
            }
        
        num_students = len(course_data)
        avg_grade = course_data['grade'].mean()
        pass_rate = (course_data['grade'] >= 10.0).sum() / num_students * 100
        
        # Clasificar dificultad
        if avg_grade >= 14.0:
            difficulty = 'Fácil'
        elif avg_grade >= 11.0:
            difficulty = 'Medio'
        else:
            difficulty = 'Difícil'
        
        return {
            'course_code': course_code,
            'num_students': num_students,
            'avg_grade': avg_grade,
            'pass_rate': pass_rate,
            'difficulty': difficulty
        }
    
    def get_prerequisite_chain(self, course_code: str) -> List[List[str]]:
        """
        Obtiene la cadena completa de prerequisitos de un curso
        
        Args:
            course_code: Código del curso
            
        Returns:
            Lista de niveles, donde cada nivel es una lista de cursos
        """
        if self.courses is None:
            raise RuntimeError("Debes llamar load_courses() primero")
        
        def get_prereqs_recursive(code: str, visited: set) -> List[List[str]]:
            if code in visited:
                return []
            visited.add(code)
            
            course_info = self.get_course_info(code)
            if not course_info or not course_info['prereq_codes']:
                return [[code]]
            
            prereqs = course_info['prereq_codes']
            chains = []
            
            for prereq in prereqs:
                sub_chains = get_prereqs_recursive(prereq, visited.copy())
                chains.extend(sub_chains)
            
            # Agregar el curso actual al final de cada cadena
            for chain in chains:
                chain.append(code)
            
            return chains if chains else [[code]]
        
        chains = get_prereqs_recursive(course_code, set())
        return chains
    
    def validate_data_consistency(self) -> Dict:
        """
        Valida consistencia entre courses.csv y courses_taken.csv
        
        Returns:
            Dict con resultados de validación
        """
        if self.courses is None or self.courses_taken is None:
            raise RuntimeError("Debes cargar ambos archivos primero")
        
        issues = []
        
        # Cursos en courses_taken que no están en courses
        catalog_courses = set(self.courses['course_code'])
        taken_courses = set(self.courses_taken['course_code'])
        
        unknown_courses = taken_courses - catalog_courses
        if unknown_courses:
            issues.append(f"Cursos en courses_taken.csv no están en catálogo: {unknown_courses}")
        
        # Prerequisitos que no existen
        all_prereqs = set()
        for prereqs in self.courses['prereq_codes']:
            all_prereqs.update(prereqs)
        
        invalid_prereqs = all_prereqs - catalog_courses
        if invalid_prereqs:
            issues.append(f"Prerequisitos inválidos: {invalid_prereqs}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'num_courses': len(catalog_courses),
            'num_students': len(self.get_all_students()),
            'num_records': len(self.courses_taken)
        }