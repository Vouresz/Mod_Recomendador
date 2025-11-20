

import yaml
from data_loader import DataLoader
from preprocess import DataPreprocessor
from kg_builder import KnowledgeGraphBuilder
from cf_model import CollaborativeFilteringModel
from content_model import ContentBasedModel
from hybrid_model import HybridRecommenderModel
from recommend import CourseRecommender

def main():
    # Cargar configuración
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    pass_threshold = config['training']['pass_threshold']
    
    print("="*70)
    print("SISTEMA DE RECOMENDACIÓN HÍBRIDO - DEMO")
    print("="*70)
    
    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    data_loader = DataLoader("data/")
    try:
        data_loader.load_courses()
        data_loader.load_courses_taken()
        print(f"[OK] {len(data_loader.courses)} cursos cargados")
        print(f"[OK] {len(data_loader.courses_taken)} registros de estudiantes")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("  Asegúrate de que existen los archivos:")
        print("    - data/courses.csv")
        print("    - data/courses_taken.csv")
        return
    
    # 2. Preprocesamiento
    print("\n[2] Preprocesando datos...")
    preprocessor = DataPreprocessor(data_loader)
    preprocessor.build_indices()
    preprocessor.encode_lineas_carrera()
    print(f"[OK] Estudiantes indexados: {len(preprocessor.student_to_idx)}")
    print(f"[OK] Cursos indexados: {len(preprocessor.course_to_idx)}")
    print(f"[OK] Líneas de carrera: {len(preprocessor.mlb_lineas.classes_)}")
    
    # 3. Knowledge Graph
    print("\n[3] Construyendo Knowledge Graph...")
    kg_builder = KnowledgeGraphBuilder(data_loader, preprocessor)
    kg_builder.build_graph(pass_threshold=pass_threshold)
    print(f"[OK] Grafo: {kg_builder.graph.number_of_nodes()} nodos")
    print("  Entrenando embeddings con random walks...")
    kg_builder.train_node2vec(dimensions=64, walk_length=30, num_walks=200)
    
    # 4. Collaborative Filtering
    print("\n[4] Entrenando Collaborative Filtering...")
    cf_model = CollaborativeFilteringModel(preprocessor, factors=64)
    cf_model.train(pass_threshold=pass_threshold)
    print(f"[OK] Modelo CF entrenado con {cf_model.factors} factores")
    
    # 5. Content-Based
    print("\n[5] Preparando Content-Based Model...")
    content_model = ContentBasedModel(preprocessor)
    content_model.build_course_vectors()
    
    # 6. Modelo Híbrido
    print("\n[6] Inicializando Modelo Híbrido...")
    hybrid_model = HybridRecommenderModel(
        kg_builder, cf_model, content_model, preprocessor
    )
    
    print("  Preparando datos de entrenamiento...")
    train_data = hybrid_model.prepare_training_data(pass_threshold=pass_threshold)
    
    print("  Entrenando MLP...")
    hybrid_model.train(train_data, epochs=5, batch_size=32, learning_rate=0.001)
    
    # 7. Sistema de Recomendación
    print("\n[7] Sistema de Recomendación Completo")
    recommender = CourseRecommender(
        data_loader, preprocessor, kg_builder,
        cf_model, content_model, hybrid_model
    )
    
    # Ejemplo de recomendación - Seleccionar estudiante válido
    students = data_loader.get_all_students()
    if students and len(students) > 100:
        # Usar estudiante en índice 100
        student_id = students[100]
        print(f"\n  Generando recomendaciones para: {student_id}")
        
        recs = recommender.recomendar_cursos(student_id, top_k=5)
        
        if recs:
            print(f"\n  Top 5 Cursos Recomendados:")
            print(f"  {'='*60}")
            for i, rec in enumerate(recs, 1):
                lineas_str = ', '.join(rec.get('lineas_carrera', [])) if rec.get('lineas_carrera') else 'N/A'
                
                print(f"\n  {i}. {rec['course_code']}")
                print(f"     Carrera(s): {lineas_str}")
                print(f"     Score: {rec['score']:.4f}")
                reasons = rec['reasons']
                print(f"     Similitud de contenido: {reasons['content_similarity']:.3f}")
                print(f"     Score colaborativo: {reasons['collaborative_score']:.3f}")
                if reasons.get('kg_neighbors'):
                    print(f"     Cursos relacionados: {reasons['kg_neighbors']}")
        else:
            print("  No hay recomendaciones disponibles (estudiante puede haber aprobado todos los cursos)")
    else:
        print("  No hay estudiantes en la base de datos")
    
    print("\n" + "="*70)
    print("[OK] DEMO COMPLETADA EXITOSAMENTE")
    print("="*70)

if __name__ == "__main__":
    main()