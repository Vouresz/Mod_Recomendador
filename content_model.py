import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedModel:
    """Modelo de filtrado basado en líneas de carrera"""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.course_vectors = None
        
    def build_course_vectors(self):
        """Construye vectores de líneas para cursos"""
        courses = self.preprocessor.data_loader.get_all_courses()
        self.course_vectors = {}
        
        for course in courses:
            vec = self.preprocessor.get_course_lineas_vector(course)
            self.course_vectors[course] = vec
        
        print(f"[OK] {len(self.course_vectors)} vectores de cursos construidos")
    
    def get_student_profile(self, student_id: str) -> np.ndarray:
        """Construye perfil del estudiante"""
        return self.preprocessor.get_student_lineas_profile(
            student_id
        )
    
    def compute_similarity(self, 
                          student_id: str,
                          course_code: str) -> float:
        """Calcula similitud entre perfil y curso"""
        if self.course_vectors is None:
            self.build_course_vectors()
        
        student_profile = self.get_student_profile(student_id)
        course_vec = self.course_vectors.get(course_code)
        
        if course_vec is None:
            return 0.0
        
        # Evitar vectores nulos
        if np.linalg.norm(student_profile) < 1e-10:
            return 0.0
        if np.linalg.norm(course_vec) < 1e-10:
            return 0.0
        
        sim = cosine_similarity(
            student_profile.reshape(1, -1),
            course_vec.reshape(1, -1)
        )[0][0]
        
        return float(sim)
    
    def get_course_embedding(self, course_code: str) -> np.ndarray:
        """Obtiene embedding de contenido del curso (vector de líneas)"""
        if self.course_vectors is None:
            self.build_course_vectors()
        
        return self.course_vectors.get(
            course_code, 
            np.zeros(len(self.preprocessor.mlb_lineas.classes_))
        )
    
    def recommend(self, student_id: str, 
                 candidates: list, top_k: int = 10):
        """Recomienda cursos por similitud de líneas"""
        scores = []
        
        for course in candidates:
            score = self.compute_similarity(student_id, course)
            scores.append((course, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def save_model(self, path: str):
        """Guarda el modelo de contenido"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'course_vectors': self.course_vectors,
            'mlb_lineas': self.preprocessor.mlb_lineas
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Modelo de contenido guardado en {path}")
    
    def load_model(self, path: str):
        """Carga el modelo de contenido"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.course_vectors = data['course_vectors']
        self.preprocessor.mlb_lineas = data['mlb_lineas']
        
        print(f"✓ Modelo de contenido cargado desde {path}")