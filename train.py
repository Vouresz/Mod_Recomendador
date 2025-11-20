import yaml
import pickle
from pathlib import Path
import torch
import numpy as np
from data_loader import DataLoader
from preprocess import DataPreprocessor
from kg_builder import KnowledgeGraphBuilder
from cf_model import CollaborativeFilteringModel
from content_model import ContentBasedModel
from hybrid_model import HybridRecommenderModel

class TrainingPipeline:
    """Pipeline completo de entrenamiento del sistema"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = None
        self.preprocessor = None
        self.kg_builder = None
        self.cf_model = None
        self.content_model = None
        self.hybrid_model = None
    
    def run_full_pipeline(self):
        """Ejecuta el pipeline completo"""
        print("="*60)
        print("INICIANDO PIPELINE DE ENTRENAMIENTO")
        print("="*60)
        
        # 1. Cargar datos
        self.load_data()
        
        # 2. Preprocesar
        self.preprocess_data()
        
        # 3. Construir Knowledge Graph
        self.build_knowledge_graph()
        
        # 4. Entrenar CF
        self.train_collaborative_filtering()
        
        # 5. Preparar Content-Based
        self.prepare_content_based()
        
        # 6. Entrenar modelo híbrido
        self.train_hybrid_model()
        
        # 7. Guardar modelos
        self.save_models()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
    
    def load_data(self):
        """Paso 1: Carga de datos"""
        print("\n[1/7] Cargando datos...")
        self.data_loader = DataLoader(
            self.config['data']['data_dir']
        )
        
        courses = self.data_loader.load_courses()
        courses_taken = self.data_loader.load_courses_taken()
        
        print(f"  ✓ Cursos cargados: {len(courses)}")
        print(f"  ✓ Registros de estudiantes: {len(courses_taken)}")
        print(f"  ✓ Estudiantes únicos: {courses_taken['alumno'].nunique()}")
    
    def preprocess_data(self):
        """Paso 2: Preprocesamiento"""
        print("\n[2/7] Preprocesando datos...")
        self.preprocessor = DataPreprocessor(self.data_loader)
        self.preprocessor.build_indices()
        
        # Codificar líneas de carrera
        lineas_encoded = self.preprocessor.encode_lineas_carrera()
        
        print(f"  ✓ Estudiantes indexados: {len(self.preprocessor.student_to_idx)}")
        print(f"  ✓ Cursos indexados: {len(self.preprocessor.course_to_idx)}")
        print(f"  ✓ Líneas de carrera: {len(self.preprocessor.mlb_lineas.classes_)}")
    
    def build_knowledge_graph(self):
        """Paso 3: Construcción del Knowledge Graph"""
        print("\n[3/7] Construyendo Knowledge Graph...")
        self.kg_builder = KnowledgeGraphBuilder(
            self.data_loader, 
            self.preprocessor
        )
        
        self.kg_builder.build_graph(
            pass_threshold=self.config['training']['pass_threshold']
        )
        
        print("  Entrenando Node2Vec...")
        self.kg_builder.train_node2vec(
            dimensions=self.config['kg']['embedding_dim'],
            walk_length=self.config['kg']['walk_length'],
            num_walks=self.config['kg']['num_walks']
        )
        
        print(f"  ✓ Embeddings generados: {len(self.kg_builder.embeddings)}")
    
    def train_collaborative_filtering(self):
        """Paso 4: Entrenamiento de CF"""
        print("\n[4/7] Entrenando Collaborative Filtering...")
        self.cf_model = CollaborativeFilteringModel(
            self.preprocessor,
            factors=self.config['cf']['factors']
        )
        
        self.cf_model.train(
            pass_threshold=self.config['training']['pass_threshold']
        )
        
        print(f"  ✓ Modelo CF entrenado")
        print(f"  ✓ Factores latentes: {self.cf_model.factors}")
    
    def prepare_content_based(self):
        """Paso 5: Preparar Content-Based"""
        print("\n[5/7] Preparando Content-Based Model...")
        self.content_model = ContentBasedModel(self.preprocessor)
        self.content_model.build_course_vectors()
        
        print(f"  ✓ Vectores de contenido construidos")
    
    def train_hybrid_model(self):
        """Paso 6: Entrenamiento del modelo híbrido"""
        print("\n[6/7] Entrenando Modelo Híbrido...")
        self.hybrid_model = HybridRecommenderModel(
            self.kg_builder,
            self.cf_model,
            self.content_model,
            self.preprocessor
        )
        
        # Preparar datos de entrenamiento
        print("  Preparando datos de entrenamiento...")
        train_data = self.hybrid_model.prepare_training_data(
            pass_threshold=self.config['training']['pass_threshold']
        )
        
        # Entrenar
        print("  Entrenando MLP...")
        self.hybrid_model.train(
            train_data,
            epochs=self.config['hybrid']['epochs'],
            batch_size=self.config['hybrid']['batch_size'],
            learning_rate=self.config['hybrid']['learning_rate']
        )
        
        print(f"  ✓ Modelo híbrido entrenado")
    
    def save_models(self):
        """Paso 7: Guardar modelos"""
        print("\n[7/7] Guardando modelos...")
        models_dir = Path(self.config['output']['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar cada componente
        self.kg_builder.save_graph(models_dir / "kg_model.pkl")
        self.cf_model.save_model(models_dir / "cf_model.pkl")
        
        # Guardar preprocessor
        with open(models_dir / "preprocessor.pkl", 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Guardar content model
        with open(models_dir / "content_model.pkl", 'wb') as f:
            pickle.dump(self.content_model, f)
        
        # Guardar modelo híbrido
        torch.save(
            self.hybrid_model.model.state_dict(),
            models_dir / "hybrid_model.pt"
        )
        
        print(f"  ✓ Modelos guardados en {models_dir}")

def main():
    """Función principal"""
    pipeline = TrainingPipeline("configs/config.yaml")
    pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()