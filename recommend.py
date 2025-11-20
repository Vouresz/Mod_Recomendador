class CourseRecommender:
    """Sistema completo de recomendación con reglas de matrícula mejoradas"""
    
    # Cursos obligatorios de ciclos básicos
    OBLIGATORY_COURSES = {
    'BAE01', 'BFI01', 'BIC01', 'BMA01', 'BMA03', 'BRN01', 'CBS01',
    'BFI05', 'BMA02', 'BMA09', 'BQU01', 'BRC01', 'CBS02',
    'BEG01', 'BFI03', 'BMA05', 'BMA10', 'BMA15', 'EE306',
    'BEF01', 'CBN01', 'BMA07', 'BMA18', 'EE320', 'EE410', 'BIE01',
    'BMA22', 'TLR01', 'TLN01', 'EE428', 'EE522', 'CBS05',
    'EE430', 'TLR02', 'EE458', 'EE588', 'EE604', 'TLN02',
    'TLR03', 'EE530', 'EE590',
    'BEG06', 'EE498', 'EE592',
    'TLR04', 'CIB45', 'TLR05',
    'EE712', 'CIB46',
}

    
    def __init__(self, data_loader, preprocessor, 
                 kg_builder, cf_model, content_model, 
                 hybrid_model):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.kg_builder = kg_builder
        self.cf_model = cf_model
        self.content_model = content_model
        self.hybrid_model = hybrid_model
    
    def cumple_prereqs(self, student_id: str, course_code: str) -> bool:
        """Verifica si cumple prerequisitos"""
        course_info = self.data_loader.get_course_info(course_code)
        if not course_info:
            return True  # Si no hay info, asumir sin prerequisitos
        
        prereqs = course_info['prereq_codes']
        if not prereqs or len(prereqs) == 0:
            return True
        
        # Obtener cursos aprobados (usando el mismo umbral: 10.0)
        history = self.data_loader.get_student_history(student_id, pass_threshold=10.0)
        passed = set(history['passed_courses'])
        
        # Verificar que todos los prerequisitos estén aprobados
        return all(p in passed for p in prereqs)
    
    def calcular_peso_lineas(self, student_id: str, course_code: str) -> float:
        """
        Calcula peso adicional basado en desempeño en cursos de la misma línea
        """
        course_info = self.data_loader.get_course_info(course_code)
        if not course_info or not course_info['lineas_carrera']:
            return 0.0
        
        course_lineas = set(course_info['lineas_carrera'])
        history = self.data_loader.get_student_history(student_id, pass_threshold=10.0)
        
        # Encontrar cursos aprobados de las mismas líneas
        related_grades = []
        for passed_course in history['passed_courses']:
            passed_info = self.data_loader.get_course_info(passed_course)
            if passed_info and passed_info['lineas_carrera']:
                passed_lineas = set(passed_info['lineas_carrera'])
                # Si comparten al menos una línea
                if course_lineas & passed_lineas:
                    grade = history['grades'].get(passed_course, 0)
                    related_grades.append(grade)
        
        # Si tiene buen desempeño en líneas relacionadas, dar boost
        if related_grades:
            avg_grade = sum(related_grades) / len(related_grades)
            # Normalizar: notas 10-20 -> peso 0.0-1.0
            peso = (avg_grade - 10.0) / 10.0
            return max(0.0, min(1.0, peso))  # Limitar entre 0 y 1
        
        return 0.0
    
    def recomendar_cursos(self, student_id: str, top_k: int = 10):
        """Genera recomendaciones finales con priorización mejorada"""
        
        # 1. Obtener historial del estudiante
        history = self.data_loader.get_student_history(student_id, pass_threshold=10.0)
        all_courses = set(self.data_loader.get_all_courses())
        passed = set(history['passed_courses'])
        all_taken = set(history['all_courses'])
        
        # 2. Identificar cursos según prioridad
        failed_courses = (all_taken - passed) & self.OBLIGATORY_COURSES  # Obligatorios reprobados
        not_taken_obligatory = self.OBLIGATORY_COURSES - all_taken  # Obligatorios no llevados
        other_courses = all_courses - passed - self.OBLIGATORY_COURSES  # Electivos y otros
        
        # 3. Crear lista priorizada de candidatos
        # PRIORIDAD 1: Obligatorios reprobados
        # PRIORIDAD 2: Obligatorios no llevados
        # PRIORIDAD 3: Otros cursos
        prioritized_candidates = (
            list(failed_courses) + 
            list(not_taken_obligatory) + 
            list(other_courses)
        )
        
        if not prioritized_candidates:
            return []
        
        # 4. Calcular scores con todas las fuentes de información
        scores = []
        for course in prioritized_candidates:
            # Verificar prerequisitos
            if not self.cumple_prereqs(student_id, course):
                continue
            
            # Score híbrido base (KG + CF + Content)
            hybrid_score = self.hybrid_model.predict_score(student_id, course)
            
            # Peso adicional por desempeño en líneas relacionadas
            lineas_weight = self.calcular_peso_lineas(student_id, course)
            
            # Score final con boosts
            final_score = hybrid_score + lineas_weight * 0.5
            
            # BOOST CRÍTICO: Priorizar obligatorios reprobados
            if course in failed_courses:
                final_score += 2.0  # Boost fuerte para reprobados
                priority = 1
            elif course in not_taken_obligatory:
                final_score += 1.0  # Boost moderado para obligatorios no llevados
                priority = 2
            else:
                priority = 3
            
            scores.append({
                'course_code': course,
                'score': final_score,
                'priority': priority,
                'is_failed': course in failed_courses,
                'is_obligatory': course in self.OBLIGATORY_COURSES,
                'lineas_weight': lineas_weight
            })
        
        # 5. Ordenar: primero por prioridad, luego por score
        scores.sort(key=lambda x: (x['priority'], -x['score']))
        
        # 6. Agregar explicaciones detalladas
        recommendations = []
        for item in scores[:top_k]:
            course = item['course_code']
            explanation = self.explain_recommendation(student_id, course)
            course_info = self.data_loader.get_course_info(course)
            
            recommendations.append({
                'course_code': course,
                'course_name': course_info['course_name'] if course_info else '',
                'score': item['score'],
                'lineas_carrera': course_info['lineas_carrera'] if course_info else [],
                'is_failed': item['is_failed'],
                'is_obligatory': item['is_obligatory'],
                'priority': item['priority'],
                'reasons': explanation
            })
        
        return recommendations
    
    def explain_recommendation(self, student_id: str, course_code: str):
        """Explica por qué se recomienda un curso con múltiples métricas"""
        
        # 1. Similitud de contenido (líneas de carrera)
        content_sim = self.content_model.compute_similarity(student_id, course_code)
        
        # 2. Vecinos en Knowledge Graph
        kg_neighbors = self.kg_builder.get_course_neighbors(course_code, k=3)
        
        # 3. Score de filtro colaborativo
        cf_score = self.cf_model.predict_score(student_id, course_code)
        
        # 4. Desempeño en líneas relacionadas
        lineas_weight = self.calcular_peso_lineas(student_id, course_code)
        
        # 5. Información de prerequisitos
        course_info = self.data_loader.get_course_info(course_code)
        prereqs = course_info['prereq_codes'] if course_info else []
        
        return {
            'content_similarity': float(content_sim),
            'kg_neighbors': kg_neighbors,
            'collaborative_score': float(cf_score),
            'lineas_performance': float(lineas_weight),
            'prerequisites': prereqs,
            'prerequisites_met': self.cumple_prereqs(student_id, course_code)
        }