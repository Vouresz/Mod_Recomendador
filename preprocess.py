import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix

class DataPreprocessor:
    """Preprocesa datos para modelos de recomendación"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.mlb_lineas = MultiLabelBinarizer()
        self.student_to_idx = {}
        self.idx_to_student = {}
        self.course_to_idx = {}
        self.idx_to_course = {}
        self.lineas_encoded = None
        
    def build_indices(self):
        """Construye índices para estudiantes y cursos"""
        # Índices de estudiantes
        students = self.data_loader.get_all_students()
        for idx, student in enumerate(students):
            self.student_to_idx[student] = idx
            self.idx_to_student[idx] = student
        
        # Índices de cursos
        courses = self.data_loader.get_all_courses()
        for idx, course in enumerate(courses):
            self.course_to_idx[course] = idx
            self.idx_to_course[idx] = course
    
    def build_interaction_matrix(self, 
                                pass_threshold: float = 10.0) -> csr_matrix:
        """Construye matriz estudiante x curso (aprobado=1)"""
        df = self.data_loader.courses_taken
        df_passed = df[df['grade'] >= pass_threshold].copy()
        
        df_passed['student_idx'] = df_passed['alumno'].map(
            self.student_to_idx
        )
        df_passed['course_idx'] = df_passed['course_code'].map(
            self.course_to_idx
        )
        
        # Filtrar valores nulos (estudiantes o cursos no en índices)
        df_passed = df_passed.dropna(subset=['student_idx', 'course_idx'])
        
        rows = df_passed['student_idx'].values.astype(int)
        cols = df_passed['course_idx'].values.astype(int)
        data = np.ones(len(df_passed))
        
        n_students = len(self.student_to_idx)
        n_courses = len(self.course_to_idx)
        
        if len(rows) == 0:
            # Matriz vacía
            matrix = csr_matrix((n_students, n_courses))
        else:
            matrix = csr_matrix(
                (data, (rows, cols)), 
                shape=(n_students, n_courses)
            )
        
        return matrix
    
    def encode_lineas_carrera(self) -> np.ndarray:
        """Crea one-hot encoding de líneas de carrera"""
        courses = self.data_loader.courses
        lineas_list = courses['lineas_carrera'].tolist()
        self.lineas_encoded = self.mlb_lineas.fit_transform(lineas_list)
        return self.lineas_encoded
    
    def get_course_lineas_vector(self, course_code: str) -> np.ndarray:
        """Obtiene vector de líneas de carrera de un curso"""
        if self.lineas_encoded is None:
            self.encode_lineas_carrera()
        
        if course_code not in self.course_to_idx:
            return np.zeros(len(self.mlb_lineas.classes_))
        
        course_idx = self.course_to_idx[course_code]
        vec = self.lineas_encoded[course_idx]
        # support both sparse matrix and numpy array
        if hasattr(vec, 'toarray'):
            return vec.toarray().flatten()
        return np.asarray(vec).flatten()
    
    def get_student_lineas_profile(self, student_id: str) -> np.ndarray:
        """Obtiene perfil de líneas de carrera del estudiante"""
        if self.lineas_encoded is None:
            self.encode_lineas_carrera()
        
        history = self.data_loader.get_student_history(student_id)
        passed_courses = history['passed_courses']
        
        if not passed_courses:
            return np.zeros(len(self.mlb_lineas.classes_))
        
        # Promediar vectores de cursos aprobados
        vectors = []
        for course in passed_courses:
            vec = self.get_course_lineas_vector(course)
            vectors.append(vec)
        
        profile = np.mean(vectors, axis=0)
        return profile / (np.linalg.norm(profile) + 1e-10)  # Normalizar