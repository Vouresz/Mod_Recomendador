#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
import sys
import torch

from data_loader import DataLoader
from kg_builder import KnowledgeGraphBuilder
from cf_model import CollaborativeFilteringModel
from content_model import ContentBasedModel
from hybrid_model import HybridRecommenderModel
from recommend import CourseRecommender


def load_preprocessor(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_kg(models_dir: Path, data_loader, preprocessor):
    kg_path = models_dir / "kg_model.pkl"
    if not kg_path.exists():
        raise FileNotFoundError(f"KG model not found: {kg_path}")

    with open(kg_path, 'rb') as f:
        kg_data = pickle.load(f)

    # If the pickle already contains a kg_builder object
    if isinstance(kg_data, dict) and 'kg_builder' in kg_data:
        return kg_data['kg_builder']

    # Otherwise rebuild a minimal KnowledgeGraphBuilder and attach graph/embeddings
    kg_builder = KnowledgeGraphBuilder(data_loader, preprocessor)
    if isinstance(kg_data, dict):
        if 'graph' in kg_data:
            kg_builder.graph = kg_data['graph']
        if 'embeddings' in kg_data:
            kg_builder.embeddings = kg_data['embeddings']
        if 'embedding_dim' in kg_data:
            kg_builder.embedding_dim = kg_data['embedding_dim']
    else:
        # Unexpected format
        raise RuntimeError("Unrecognized kg_model.pkl format")

    return kg_builder


def load_cf(models_dir: Path, preprocessor):
    cf_path = models_dir / "cf_model.pkl"
    if not cf_path.exists():
        raise FileNotFoundError(f"CF model not found: {cf_path}")

    with open(cf_path, 'rb') as f:
        maybe_cf = pickle.load(f)

    if isinstance(maybe_cf, CollaborativeFilteringModel):
        return maybe_cf

    # Otherwise reconstruct via load_model
    cf_model = CollaborativeFilteringModel(preprocessor)
    cf_model.load_model(cf_path)
    return cf_model


def load_content(models_dir: Path):
    content_path = models_dir / "content_model.pkl"
    if not content_path.exists():
        raise FileNotFoundError(f"Content model not found: {content_path}")
    with open(content_path, 'rb') as f:
        return pickle.load(f)


def load_hybrid(models_dir: Path, kg_builder, cf_model, content_model, preprocessor):
    hybrid_pt = models_dir / "hybrid_model.pt"
    if not hybrid_pt.exists():
        raise FileNotFoundError(f"Hybrid model weights not found: {hybrid_pt}")

    hybrid_model = HybridRecommenderModel(kg_builder, cf_model, content_model, preprocessor)
    hybrid_model.model.load_state_dict(torch.load(hybrid_pt))
    return hybrid_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('student_id', nargs='?', default=None, help='ID del estudiante (ej. EST001)')
    parser.add_argument('--top_k', type=int, default=5, help='Número de recomendaciones a retornar')
    parser.add_argument('--models_dir', type=str, default='models', help='Directorio donde están los modelos')
    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    try:
        data_loader = DataLoader('data/')
        data_loader.load_courses()
        data_loader.load_courses_taken()
    except Exception as e:
        print(f"Error cargando datos: {e}")
        sys.exit(1)

    # Determine student_id
    student_id = "ALUMNO_REAL"

    """
    student_id = args.student_id
    if student_id is None:
        students = data_loader.get_all_students()
        if not students:
            print("No hay estudiantes en los datos.")
            sys.exit(1)
        student_id = students[0]                 
        print(f"Usando primer estudiante disponible: {student_id}")
    """ 

    try:
        preprocessor = load_preprocessor(models_dir / 'preprocessor.pkl')
    except Exception as e:
        print(f"Error cargando preprocessor: {e}")
        sys.exit(1)

    try:
        kg_builder = load_kg(models_dir, data_loader, preprocessor)
    except Exception as e:
        print(f"Error cargando KG: {e}")
        sys.exit(1)

    try:
        cf_model = load_cf(models_dir, preprocessor)
    except Exception as e:
        print(f"Error cargando CF: {e}")
        sys.exit(1)

    try:
        content_model = load_content(models_dir)
    except Exception as e:
        print(f"Error cargando Content model: {e}")
        sys.exit(1)

    try:
        hybrid_model = load_hybrid(models_dir, kg_builder, cf_model, content_model, preprocessor)
    except Exception as e:
        print(f"Error cargando Hybrid model: {e}")
        sys.exit(1)

    recommender = CourseRecommender(data_loader, preprocessor, kg_builder, cf_model, content_model, hybrid_model)

    try:
        recs = recommender.recomendar_cursos(student_id, top_k=args.top_k)
    except Exception as e:
        print(f"Error generando recomendaciones: {e}")
        sys.exit(1)

    # Print nicely
    print('\nTop {0} Cursos Recomendados para {1}:\n'.format(args.top_k, student_id))
    print('=============================================================')
    for i, rec in enumerate(recs, 1):
        lineas_str = ', '.join(rec['lineas_carrera']) if rec.get('lineas_carrera') else 'N/A'
        tipo_curso = "OBLIGATORIO" if rec.get('is_obligatory', False) else "Electivo"
        print(f"{i}. {rec['course_code']}")
        print(f"   Carrera(s): {lineas_str}")
        print(f"   Score: {rec['score']:.4f}")
        reasons = rec['reasons']
        print(f"   Similitud de contenido: {reasons['content_similarity']:.3f}")
        print(f"   Score colaborativo: {reasons['collaborative_score']:.3f}")
        if reasons.get('kg_neighbors'):
            print(f"   Cursos relacionados: {reasons['kg_neighbors']}")
        print('')


if __name__ == '__main__':
    main()
