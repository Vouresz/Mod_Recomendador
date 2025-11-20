import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple

class HybridFusionMLP(nn.Module):
    """Red neuronal para fusión de embeddings"""
    
    def __init__(self, kg_dim, cf_dim, content_dim):
        super().__init__()
        
        input_dim = (kg_dim + cf_dim + content_dim) * 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, student_emb, course_emb):
        x = torch.cat([student_emb, course_emb], dim=1)
        return self.network(x).squeeze()

class HybridRecommenderModel:
    """Sistema híbrido KG + CF + Content"""
    
    def __init__(self, kg_builder, cf_model, 
                 content_model, preprocessor):
        self.kg_builder = kg_builder
        self.cf_model = cf_model
        self.content_model = content_model
        self.preprocessor = preprocessor
        
        kg_dim = 64
        cf_dim = cf_model.factors
        content_dim = len(preprocessor.mlb_lineas.classes_)
        
        self.model = HybridFusionMLP(kg_dim, cf_dim, content_dim)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None
    
    def get_student_embedding(self, student_id: str):
        """Embedding híbrido del estudiante"""
        kg_emb = self.kg_builder.get_student_embedding(student_id)
        cf_emb = self.cf_model.get_student_embedding(student_id)
        content_emb = self.content_model.get_student_profile(student_id)
        
        return np.concatenate([kg_emb, cf_emb, content_emb]).astype(np.float32)
    
    def get_course_embedding(self, course_code: str):
        """Embedding híbrido del curso"""
        kg_emb = self.kg_builder.get_course_embedding(course_code)
        cf_emb = self.cf_model.get_course_embedding(course_code)
        content_emb = self.content_model.get_course_embedding(course_code)
        
        return np.concatenate([kg_emb, cf_emb, content_emb]).astype(np.float32)
    
    def predict_score(self, student_id: str, 
                     course_code: str) -> float:
        """Predice score híbrido"""
        student_emb = torch.FloatTensor(
            self.get_student_embedding(student_id)
        ).unsqueeze(0)
        course_emb = torch.FloatTensor(
            self.get_course_embedding(course_code)
        ).unsqueeze(0)
        # Run in eval mode to avoid BatchNorm errors with batch size 1
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            score = self.model(student_emb, course_emb)
        if was_training:
            self.model.train()

        return float(score.item())
    
    def prepare_training_data(self, pass_threshold: float = 11.0) -> List[Tuple]:
        """Prepara datos de entrenamiento con pares positivos y negativos"""
        training_data = []
        
        students = self.preprocessor.data_loader.get_all_students()
        all_courses = self.preprocessor.data_loader.get_all_courses()
        all_courses_set = set(all_courses)
        
        print("Preparando datos de entrenamiento...")
        
        for student_id in students:
            history = self.preprocessor.data_loader.get_student_history(
                student_id, pass_threshold
            )
            passed = set(history['passed_courses'])
            not_passed = all_courses_set - passed
            
            # Pares positivos (cursos aprobados)
            for course in passed:
                training_data.append((student_id, course, 1))
            
            # Pares negativos (cursos no aprobados - muestreo)
            sample_size = min(len(passed), len(not_passed))
            if sample_size > 0:
                negative_courses = np.random.choice(
                    list(not_passed), size=sample_size, replace=False
                )
                for course in negative_courses:
                    training_data.append((student_id, course, 0))
        
        print(f"[OK] {len(training_data)} pares generados")
        return training_data
    
    def train(self, train_data: List[Tuple], epochs: int = 10, 
              batch_size: int = 32, learning_rate: float = 0.001):
        """Entrena el modelo híbrido"""
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=learning_rate)
        
        print(f"Entrenando modelo híbrido ({epochs} épocas)...")
        # Precompute embeddings for all students and courses in the training set
        unique_students = sorted({s for s, _, _ in train_data})
        unique_courses = sorted({c for _, c, _ in train_data})

        print(f"  Precomputando embeddings para {len(unique_students)} estudiantes y {len(unique_courses)} cursos...")
        student_cache = {sid: self.get_student_embedding(sid) for sid in unique_students}
        course_cache = {cc: self.get_course_embedding(cc) for cc in unique_courses}

        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle training data
            indices = np.random.permutation(len(train_data))
            
            # Mini-batches
            for batch_start in range(0, len(train_data), batch_size):
                batch_end = min(batch_start + batch_size, len(train_data))
                batch_indices = indices[batch_start:batch_end]
                
                # Preparar batch
                student_embs = []
                course_embs = []
                labels = []
                
                for idx in batch_indices:
                    student_id, course_code, label = train_data[idx]
                    # Use precomputed embeddings to avoid repeated expensive calls
                    student_embs.append(student_cache.get(student_id, 
                                                          np.zeros_like(next(iter(student_cache.values())))))
                    course_embs.append(course_cache.get(course_code, 
                                                        np.zeros_like(next(iter(course_cache.values())))))
                    labels.append(float(label))
                
                # Convertir a tensores
                student_embs = torch.FloatTensor(np.array(student_embs))
                course_embs = torch.FloatTensor(np.array(course_embs))
                labels = torch.FloatTensor(labels)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(student_embs, course_embs)
                loss = self.criterion(predictions, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(train_data) // batch_size + 1)
            print(f"  Época {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    def save_model(self, path: str):
        """Guarda el modelo híbrido"""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Modelo híbrido guardado en {path}")
    
    def load_model(self, path: str):
        """Carga el modelo híbrido"""
        self.model.load_state_dict(torch.load(path))
        print(f"✓ Modelo híbrido cargado desde {path}")