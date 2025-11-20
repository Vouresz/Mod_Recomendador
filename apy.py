"""
API REST para el Sistema de Recomendación Híbrido

Endpoints disponibles:
- GET  /api/health                          - Health check
- GET  /api/students                        - Listar estudiantes
- GET  /api/students/{id}                   - Info de estudiante
- GET  /api/students/{id}/history           - Historial académico
- GET  /api/students/{id}/recommendations   - Recomendaciones
- GET  /api/courses                         - Listar cursos
- GET  /api/courses/{code}                  - Info de curso
- GET  /api/courses/{code}/students         - Estudiantes del curso
- POST /api/recommend                       - Recomendación personalizada
- GET  /api/stats                           - Estadísticas del sistema
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import torch
from pathlib import Path
from typing import Dict, List
import traceback

from data_loader import DataLoader
from kg_builder import KnowledgeGraphBuilder
from cf_model import CollaborativeFilteringModel
from content_model import ContentBasedModel
from hybrid_model import HybridRecommenderModel
from recommend import CourseRecommender
from utils import (
    analyze_student_performance,
    get_curriculum_progress,
    validate_data
)

class RecommenderAPI:
    """API del Sistema de Recomendación"""
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_loaded = False
        
        # Componentes del sistema
        self.data_loader = None
        self.preprocessor = None
        self.kg_builder = None
        self.cf_model = None
        self.content_model = None
        self.hybrid_model = None
        self.recommender = None
        
    def load_models(self):
        """Carga todos los modelos entrenados"""
        try:
            print("Cargando modelos...")
            
            # Cargar datos
            self.data_loader = DataLoader(str(self.data_dir))
            self.data_loader.load_courses()
            self.data_loader.load_courses_taken()
            print("✓ Datos cargados")
            
            # Cargar preprocessor
            with open(self.models_dir / "preprocessor.pkl", 'rb') as f:
                self.preprocessor = pickle.load(f)
            print("✓ Preprocessor cargado")
            
            # Cargar Knowledge Graph
            self.kg_builder = KnowledgeGraphBuilder(self.data_loader, self.preprocessor)
            with open(self.models_dir / "kg_model.pkl", 'rb') as f:
                kg_data = pickle.load(f)
                self.kg_builder.graph = kg_data['graph']
                self.kg_builder.embeddings = kg_data['embeddings']
                self.kg_builder.embedding_dim = kg_data['embedding_dim']
            print("✓ Knowledge Graph cargado")
            
            # Cargar CF Model
            self.cf_model = CollaborativeFilteringModel(self.preprocessor)
            self.cf_model.load_model(self.models_dir / "cf_model.pkl")
            print("✓ Modelo CF cargado")
            
            # Cargar Content Model
            with open(self.models_dir / "content_model.pkl", 'rb') as f:
                self.content_model = pickle.load(f)
            print("✓ Modelo Content-Based cargado")
            
            # Cargar Hybrid Model
            self.hybrid_model = HybridRecommenderModel(
                self.kg_builder, self.cf_model, 
                self.content_model, self.preprocessor
            )
            self.hybrid_model.model.load_state_dict(
                torch.load(self.models_dir / "hybrid_model.pt", map_location='cpu')
            )
            self.hybrid_model.model.eval()
            print("✓ Modelo Híbrido cargado")
            
            # Crear recommender
            self.recommender = CourseRecommender(
                self.data_loader, self.preprocessor,
                self.kg_builder, self.cf_model,
                self.content_model, self.hybrid_model
            )
            print("✓ Sistema de recomendación inicializado")
            
            self.models_loaded = True
            print("\n✅ Todos los modelos cargados exitosamente\n")
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            traceback.print_exc()
            raise


# Inicializar Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para requests desde frontend

# Inicializar API
api = RecommenderAPI()


# ==================== MIDDLEWARE ====================

@app.before_request
def check_models():
    """Verifica que los modelos estén cargados"""
    if request.endpoint not in ['health', 'static']:
        if not api.models_loaded:
            return jsonify({
                'error': 'Modelos no cargados',
                'message': 'Por favor espera a que los modelos se carguen'
            }), 503


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500


# ==================== ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check del API"""
    return jsonify({
        'status': 'online',
        'models_loaded': api.models_loaded,
        'version': '1.0.0'
    })


@app.route('/api/students', methods=['GET'])
def get_students():
    """Lista todos los estudiantes"""
    try:
        students = api.data_loader.get_all_students()
        
        # Paginación opcional
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        start = (page - 1) * per_page
        end = start + per_page
        
        return jsonify({
            'students': students[start:end],
            'total': len(students),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(students) + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/students/<student_id>', methods=['GET'])
def get_student(student_id):
    """Obtiene información detallada de un estudiante"""
    try:
        students = api.data_loader.get_all_students()
        
        if student_id not in students:
            return jsonify({'error': 'Estudiante no encontrado'}), 404
        
        # Historial
        history = api.data_loader.get_student_history(student_id)
        
        # Análisis de desempeño
        analysis = analyze_student_performance(api.data_loader, student_id)
        
        # Progreso curricular
        progress = get_curriculum_progress(
            api.data_loader, student_id,
            CourseRecommender.OBLIGATORY_COURSES
        )
        
        return jsonify({
            'student_id': student_id,
            'history': {
                'total_courses': len(history['all_courses']),
                'passed_courses': len(history['passed_courses']),
                'courses': history['all_courses']
            },
            'performance': {
                'pass_rate': analysis['pass_rate'],
                'avg_grade': analysis['avg_grade_passed'],
                'best_linea': analysis['best_linea'],
                'worst_linea': analysis['worst_linea'],
                'lineas_performance': analysis['lineas_performance']
            },
            'curriculum_progress': {
                'progress_percentage': progress['progress_percentage'],
                'obligatory_passed': progress['obligatory_passed'],
                'obligatory_failed': progress['obligatory_failed'],
                'obligatory_pending': progress['obligatory_pending'],
                'failed_list': progress['failed_list']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/students/<student_id>/history', methods=['GET'])
def get_student_history(student_id):
    """Obtiene historial académico completo"""
    try:
        students = api.data_loader.get_all_students()
        
        if student_id not in students:
            return jsonify({'error': 'Estudiante no encontrado'}), 404
        
        history = api.data_loader.get_student_history(student_id)
        
        # Organizar por ciclos
        courses_by_cycle = {}
        for cycle, courses_list in history['by_cycle'].items():
            courses_by_cycle[str(cycle)] = courses_list
        
        return jsonify({
            'student_id': student_id,
            'all_courses': history['all_courses'],
            'passed_courses': history['passed_courses'],
            'grades': history['grades'],
            'by_cycle': courses_by_cycle
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/students/<student_id>/recommendations', methods=['GET'])
def get_recommendations(student_id):
    """Genera recomendaciones para un estudiante"""
    try:
        students = api.data_loader.get_all_students()
        
        if student_id not in students:
            return jsonify({'error': 'Estudiante no encontrado'}), 404
        
        # Parámetros
        top_k = request.args.get('top_k', 10, type=int)
        
        # Generar recomendaciones
        recs = api.recommender.recomendar_cursos(student_id, top_k=top_k)
        
        # Formatear respuesta
        recommendations = []
        for rec in recs:
            recommendations.append({
                'course_code': rec['course_code'],
                'course_name': rec.get('course_name', ''),
                'score': round(rec['score'], 4),
                'lineas_carrera': rec['lineas_carrera'],
                'is_failed': rec.get('is_failed', False),
                'is_obligatory': rec.get('is_obligatory', False),
                'priority': rec.get('priority', 3),
                'reasons': {
                    'content_similarity': round(rec['reasons']['content_similarity'], 3),
                    'collaborative_score': round(rec['reasons']['collaborative_score'], 3),
                    'lineas_performance': round(rec['reasons']['lineas_performance'], 3),
                    'kg_neighbors': rec['reasons']['kg_neighbors'],
                    'prerequisites': rec['reasons'].get('prerequisites', []),
                    'prerequisites_met': rec['reasons'].get('prerequisites_met', True)
                }
            })
        
        return jsonify({
            'student_id': student_id,
            'top_k': top_k,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/courses', methods=['GET'])
def get_courses():
    """Lista todos los cursos"""
    try:
        courses = api.data_loader.courses
        
        # Filtros opcionales
        linea = request.args.get('linea', None)
        
        if linea:
            # Filtrar por línea de carrera
            filtered = courses[
                courses['lineas_carrera'].apply(lambda x: linea in x)
            ]
        else:
            filtered = courses
        
        # Paginación
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        start = (page - 1) * per_page
        end = start + per_page
        
        courses_list = []
        for _, course in filtered.iloc[start:end].iterrows():
            courses_list.append({
                'course_code': course['course_code'],
                'course_name': course['course_name'],
                'prereq_codes': course['prereq_codes'],
                'lineas_carrera': course['lineas_carrera']
            })
        
        return jsonify({
            'courses': courses_list,
            'total': len(filtered),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(filtered) + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/courses/<course_code>', methods=['GET'])
def get_course(course_code):
    """Obtiene información detallada de un curso"""
    try:
        course_info = api.data_loader.get_course_info(course_code)
        
        if not course_info:
            return jsonify({'error': 'Curso no encontrado'}), 404
        
        # Estadísticas del curso
        stats = api.data_loader.get_course_statistics(course_code)
        
        # Cursos relacionados
        neighbors = api.kg_builder.get_course_neighbors(course_code, k=5)
        
        return jsonify({
            'course_code': course_code,
            'course_name': course_info['course_name'],
            'prereq_codes': course_info['prereq_codes'],
            'lineas_carrera': course_info['lineas_carrera'],
            'statistics': {
                'num_students': stats['num_students'],
                'avg_grade': round(stats['avg_grade'], 2),
                'pass_rate': round(stats['pass_rate'], 1),
                'difficulty': stats['difficulty']
            },
            'related_courses': [
                {'course_code': code, 'relation': rel}
                for code, rel in neighbors
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/courses/<course_code>/students', methods=['GET'])
def get_course_students(course_code):
    """Obtiene estudiantes que tomaron un curso"""
    try:
        students = api.data_loader.get_students_who_took_course(course_code)
        
        return jsonify({
            'course_code': course_code,
            'students': students,
            'total': len(students)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Recomendación personalizada con parámetros"""
    try:
        data = request.json
        
        if not data or 'student_id' not in data:
            return jsonify({'error': 'student_id es requerido'}), 400
        
        student_id = data['student_id']
        top_k = data.get('top_k', 10)
        
        students = api.data_loader.get_all_students()
        if student_id not in students:
            return jsonify({'error': 'Estudiante no encontrado'}), 404
        
        # Generar recomendaciones
        recs = api.recommender.recomendar_cursos(student_id, top_k=top_k)
        
        # Formatear
        recommendations = []
        for rec in recs:
            recommendations.append({
                'course_code': rec['course_code'],
                'course_name': rec.get('course_name', ''),
                'score': round(rec['score'], 4),
                'lineas_carrera': rec['lineas_carrera'],
                'is_failed': rec.get('is_failed', False),
                'is_obligatory': rec.get('is_obligatory', False),
                'priority': rec.get('priority', 3),
                'reasons': {
                    'content_similarity': round(rec['reasons']['content_similarity'], 3),
                    'collaborative_score': round(rec['reasons']['collaborative_score'], 3),
                    'lineas_performance': round(rec['reasons']['lineas_performance'], 3)
                }
            })
        
        return jsonify({
            'student_id': student_id,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Estadísticas generales del sistema"""
    try:
        students = api.data_loader.get_all_students()
        courses = api.data_loader.get_all_courses()
        
        # Validación de datos
        validation = validate_data(
            api.data_loader.courses,
            api.data_loader.courses_taken
        )
        
        return jsonify({
            'system': {
                'total_students': len(students),
                'total_courses': len(courses),
                'total_records': len(api.data_loader.courses_taken),
                'total_lineas': len(api.preprocessor.mlb_lineas.classes_),
                'lineas': list(api.preprocessor.mlb_lineas.classes_)
            },
            'models': {
                'kg_embeddings': len(api.kg_builder.embeddings),
                'kg_nodes': api.kg_builder.graph.number_of_nodes(),
                'kg_edges': api.kg_builder.graph.number_of_edges(),
                'cf_factors': api.cf_model.factors,
                'embedding_dim': api.kg_builder.embedding_dim
            },
            'data_quality': {
                'valid': validation['valid'],
                'issues': validation['issues'],
                'warnings': validation['warnings']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/lineas', methods=['GET'])
def get_lineas():
    """Lista todas las líneas de carrera"""
    try:
        lineas = list(api.preprocessor.mlb_lineas.classes_)
        
        # Contar cursos por línea
        lineas_stats = {}
        for linea in lineas:
            count = api.data_loader.courses[
                api.data_loader.courses['lineas_carrera'].apply(lambda x: linea in x)
            ].shape[0]
            lineas_stats[linea] = count
        
        return jsonify({
            'lineas': lineas,
            'total': len(lineas),
            'courses_per_linea': lineas_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== MAIN ====================

def main():
    """Inicia el servidor API"""
    import argparse
    
    parser = argparse.ArgumentParser(description='API REST del Sistema de Recomendación')
    parser.add_argument('--host', default='0.0.0.0', help='Host del servidor')
    parser.add_argument('--port', type=int, default=5000, help='Puerto del servidor')
    parser.add_argument('--models-dir', default='models', help='Directorio de modelos')
    parser.add_argument('--data-dir', default='data', help='Directorio de datos')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    
    print("="*70)
    print("API REST - Sistema de Recomendación Híbrido")
    print("="*70)
    print(f"\nHost: {args.host}")
    print(f"Puerto: {args.port}")
    print(f"Modelos: {args.models_dir}")
    print(f"Datos: {args.data_dir}\n")
    
    # Actualizar directorios
    api.models_dir = Path(args.models_dir)
    api.data_dir = Path(args.data_dir)
    
    # Cargar modelos
    try:
        api.load_models()
    except Exception as e:
        print(f"\n❌ Error al cargar modelos: {e}")
        print("Asegúrate de haber entrenado los modelos primero con: python train.py")
        return
    
    print("="*70)
    print("🚀 Servidor API iniciado")
    print("="*70)
    print(f"\nDocumentación: http://{args.host}:{args.port}/api/health")
    print("\nEndpoints disponibles:")
    print("  • GET  /api/health")
    print("  • GET  /api/students")
    print("  • GET  /api/students/{id}")
    print("  • GET  /api/students/{id}/recommendations")
    print("  • GET  /api/courses")
    print("  • GET  /api/courses/{code}")
    print("  • POST /api/recommend")
    print("  • GET  /api/stats")
    print("\n" + "="*70 + "\n")
    
    # Iniciar servidor
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()