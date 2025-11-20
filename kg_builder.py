import networkx as nx
import numpy as np
import random
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

class KnowledgeGraphBuilder:
    """Construye y procesa el Knowledge Graph académico"""
    
    def __init__(self, data_loader, preprocessor):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.graph = nx.MultiDiGraph()
        self.embeddings = {}
        self.embedding_dim = 64
        
    def build_graph(self, pass_threshold: float = 11.0):
        """Construye grafo heterogéneo completo"""
        courses = self.data_loader.courses
        
        # 1. Agregar nodos de cursos
        for _, course in courses.iterrows():
            self.graph.add_node(
                f"Course:{course['course_code']}",
                type='course'
            )
        
        # 2. Agregar nodos de líneas de carrera
        all_lineas = set()
        for lineas in courses['lineas_carrera']:
            all_lineas.update(lineas)
        
        for linea in all_lineas:
            self.graph.add_node(f"Linea:{linea}", type='linea')
        
        # 3. Agregar nodos de estudiantes
        students = self.data_loader.get_all_students()
        for student in students:
            self.graph.add_node(f"Student:{student}", type='student')
        
        # 4. Relaciones BELONGS_TO (Curso -> Linea)
        for _, course in courses.iterrows():
            course_node = f"Course:{course['course_code']}"
            for linea in course['lineas_carrera']:
                self.graph.add_edge(
                    course_node, 
                    f"Linea:{linea}", 
                    type='BELONGS_TO'
                )
        
        # 5. Relaciones HAS_PREREQ
        for _, course in courses.iterrows():
            course_node = f"Course:{course['course_code']}"
            for prereq in course['prereq_codes']:
                prereq_node = f"Course:{prereq}"
                if self.graph.has_node(prereq_node):
                    self.graph.add_edge(
                        course_node,
                        prereq_node,
                        type='HAS_PREREQ'
                    )
        
        # 6. Relaciones TOOK (Estudiante -> Curso)
        for _, row in self.data_loader.courses_taken.iterrows():
            student_node = f"Student:{row['alumno']}"
            course_node = f"Course:{row['course_code']}"
            passed = row['grade'] >= pass_threshold
            
            self.graph.add_edge(
                student_node,
                course_node,
                type='TOOK',
                grade=row['grade'],
                passed=passed
            )
        
        print(f"Grafo construido: {self.graph.number_of_nodes()} nodos, {self.graph.number_of_edges()} aristas")
    
    def train_node2vec(self, dimensions: int = 64, walk_length: int = 30, 
                       num_walks: int = 200):
        """Entrena Node2Vec para embeddings usando random walk simplificado"""
        self.embedding_dim = dimensions
        
        # Inicializar embeddings aleatorios
        for node in self.graph.nodes():
            self.embeddings[node] = np.random.randn(dimensions).astype(np.float32)
        
        # Entrenar con random walks
        print("Entrenando embeddings con random walks...")
        
        # Compilar vecinos para cada nodo
        neighbors = {}
        for node in self.graph.nodes():
            neighbors[node] = list(self.graph.neighbors(node))
        
        # Random walks
        for walk_id in range(num_walks):
            if (walk_id + 1) % 50 == 0:
                print(f"  Walk {walk_id + 1}/{num_walks}")
            
            for start_node in list(self.graph.nodes()):
                walk = self._random_walk(start_node, walk_length, neighbors)
                self._update_embeddings(walk, dimensions)
        
        print(f"[OK] {len(self.embeddings)} embeddings generados")
    
    def _random_walk(self, start_node: str, walk_length: int, neighbors: Dict) -> List[str]:
        """Genera un random walk partiendo de un nodo"""
        walk = [start_node]
        current = start_node
        
        for _ in range(walk_length - 1):
            if current in neighbors and neighbors[current]:
                current = random.choice(neighbors[current])
                walk.append(current)
            else:
                break
        
        return walk
    
    def _update_embeddings(self, walk: List[str], dim: int, learning_rate: float = 0.01):
        """Actualiza embeddings basado en un walk (simplificado)"""
        window_size = 5
        
        for i, node in enumerate(walk):
            context_start = max(0, i - window_size)
            context_end = min(len(walk), i + window_size + 1)
            
            for j in range(context_start, context_end):
                if i != j and j < len(walk):
                    context_node = walk[j]
                    # Actualización simplificada
                    similarity = np.dot(self.embeddings[node], 
                                       self.embeddings[context_node])
                    gradient = (similarity - 1.0) * learning_rate
                    
                    self.embeddings[node] -= gradient * self.embeddings[context_node]
                    self.embeddings[context_node] -= gradient * self.embeddings[node]
            
            # Normalizar
            norm = np.linalg.norm(self.embeddings[node]) + 1e-10
            self.embeddings[node] /= norm
    
    def get_student_embedding(self, student_id: str) -> np.ndarray:
        """Obtiene embedding del estudiante desde el grafo"""
        node = f"Student:{student_id}"
        if node in self.embeddings:
            return self.embeddings[node]
        else:
            return np.zeros(self.embedding_dim)
    
    def get_course_embedding(self, course_code: str) -> np.ndarray:
        """Obtiene embedding del curso desde el grafo"""
        node = f"Course:{course_code}"
        if node in self.embeddings:
            return self.embeddings[node]
        else:
            return np.zeros(self.embedding_dim)
    
    def get_course_neighbors(self, course_code: str, k: int = 3) -> List[Tuple[str, str]]:
        """Obtiene cursos vecinos por prerequisitos o líneas"""
        course_node = f"Course:{course_code}"
        neighbors = []
        
        # Cursos relacionados por línea de carrera
        for neighbor in self.graph.neighbors(course_node):
            if neighbor.startswith("Linea:"):
                for linea_neighbor in self.graph.predecessors(neighbor):
                    if linea_neighbor.startswith("Course:") and linea_neighbor != course_node:
                        neighbors.append((linea_neighbor.replace("Course:", ""), "linea"))
        
        # Cursos prerequisito
        for neighbor in self.graph.neighbors(course_node):
            if neighbor.startswith("Course:"):
                neighbors.append((neighbor.replace("Course:", ""), "prereq"))
        
        return neighbors[:k]
    
    def save_graph(self, path: str):
        """Guarda el grafo y embeddings"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'graph': self.graph,
            'embeddings': self.embeddings,
            'embedding_dim': self.embedding_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Grafo guardado en {path}")
    
    def load_graph(self, path: str):
        """Carga el grafo y embeddings"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.graph = data['graph']
        self.embeddings = data['embeddings']
        self.embedding_dim = data['embedding_dim']
        
        print(f"✓ Grafo cargado desde {path}")