
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple

class CollaborativeFilteringModel:
    """Modelo de Collaborative Filtering con ALS"""
    
    def __init__(self, preprocessor, factors: int = 64):
        self.preprocessor = preprocessor
        self.factors = factors
        self.user_factors = None
        self.item_factors = None
        self.interaction_matrix = None
    
    def train(self, pass_threshold: float = 11.0):
        """Entrena el modelo CF con factorización de matriz simplificada"""
        self.interaction_matrix = self.preprocessor.build_interaction_matrix(
            pass_threshold
        )
        
        n_users = self.interaction_matrix.shape[0]
        n_items = self.interaction_matrix.shape[1]
        
        print(f"Entrenando ALS: {n_users} usuarios x {n_items} items...")
        
        # Inicializar factores
        self.user_factors = np.random.randn(n_users, self.factors).astype(np.float32) * 0.01
        self.item_factors = np.random.randn(n_items, self.factors).astype(np.float32) * 0.01
        
        # ALS alternado simplificado
        iterations = 15
        lambda_reg = 0.01
        
        for iteration in range(iterations):
            print(f"  Iteración {iteration + 1}/{iterations}")
            
            # Actualizar factores de usuario
            for u in range(n_users):
                # Obtener items que el usuario interactuó
                items = self.interaction_matrix[u].nonzero()[1]
                if len(items) > 0:
                    # Resolver mínimos cuadrados
                    V = self.item_factors[items]  # [n_interac, factors]
                    A = V.T @ V + lambda_reg * np.eye(self.factors)
                    b = V.T @ np.ones(len(items))
                    self.user_factors[u] = np.linalg.solve(A, b)
            
            # Actualizar factores de item
            for i in range(n_items):
                # Obtener usuarios que interactuaron con el item
                users = self.interaction_matrix[:, i].nonzero()[0]
                if len(users) > 0:
                    # Resolver mínimos cuadrados
                    U = self.user_factors[users]  # [n_interac, factors]
                    A = U.T @ U + lambda_reg * np.eye(self.factors)
                    b = U.T @ np.ones(len(users))
                    self.item_factors[i] = np.linalg.solve(A, b)
        
        print("[OK] ALS entrenado")
    
    def get_student_embedding(self, student_id: str) -> np.ndarray:
        """Obtiene embedding CF del estudiante"""
        if student_id not in self.preprocessor.student_to_idx:
            return np.zeros(self.factors, dtype=np.float32)
        
        idx = self.preprocessor.student_to_idx[student_id]
        return self.user_factors[idx]
    
    def get_course_embedding(self, course_code: str) -> np.ndarray:
        """Obtiene embedding CF del curso"""
        if course_code not in self.preprocessor.course_to_idx:
            return np.zeros(self.factors, dtype=np.float32)
        
        idx = self.preprocessor.course_to_idx[course_code]
        return self.item_factors[idx]
    
    def predict_score(self, student_id: str, course_code: str) -> float:
        """Predice score CF para un estudiante y curso"""
        user_emb = self.get_student_embedding(student_id)
        item_emb = self.get_course_embedding(course_code)
        
        score = np.dot(user_emb, item_emb)
        return float(score)
    
    def recommend(self, student_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Recomienda cursos usando CF"""
        if student_id not in self.preprocessor.student_to_idx:
            return []
        
        student_idx = self.preprocessor.student_to_idx[student_id]
        user_emb = self.user_factors[student_idx]
        
        # Calcular scores para todos los items
        scores = self.item_factors @ user_emb
        
        # Top-k items no vistos
        taken_items = self.interaction_matrix[student_idx].nonzero()[1]
        
        # Scores sin los items ya vistos
        scores_copy = scores.copy()
        scores_copy[taken_items] = -np.inf
        
        # Top-k
        top_indices = np.argsort(-scores_copy)[:top_k]
        
        recommendations = []
        for idx in top_indices:
            course_code = self.preprocessor.idx_to_course[idx]
            score = float(scores[idx])
            recommendations.append((course_code, score))
        
        return recommendations
    
    def save_model(self, path: str):
        """Guarda el modelo CF"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'interaction_matrix': self.interaction_matrix,
            'factors': self.factors
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Modelo CF guardado en {path}")
    
    def load_model(self, path: str):
        """Carga el modelo CF"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.user_factors = data['user_factors']
        self.item_factors = data['item_factors']
        self.interaction_matrix = data['interaction_matrix']
        self.factors = data['factors']
        
        print(f"✓ Modelo CF cargado desde {path}")