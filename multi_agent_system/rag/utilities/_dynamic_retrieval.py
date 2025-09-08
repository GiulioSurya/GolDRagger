"""
Modulo: DynamicRetrieval
Descrizione: Sistema dynamic top-k basato su query complexity con context window awareness
"""

from typing import Dict, Any, Tuple
import re


def estimate_token_count(text: str) -> int:
    """
    Stima approssimativa del numero di token.
    Usa euristica: 1 token ≈ 4 caratteri per testo italiano/inglese.

    Args:
        text: Testo da analizzare

    Returns:
        Numero stimato di token
    """
    if not text or not text.strip():
        return 0

    # Rimuovi spazi multipli per conteggio più accurato
    normalized_text = re.sub(r'\s+', ' ', text.strip())
    return max(1, len(normalized_text) // 4)


def analyze_query_structure(query: str) -> Dict[str, Any]:
    """
    Analizza la struttura sintattica della query.

    Args:
        query: Query dell'utente

    Returns:
        Dizionario con metriche strutturali
    """
    if not query or not query.strip():
        return {
            'word_count': 0,
            'question_words': 0,
            'conjunctions': 0,
            'has_multiple_clauses': False
        }

    words = query.lower().split()

    # Parole interrogative italiane
    question_words = ['chi', 'cosa', 'come', 'quando', 'dove', 'perché', 'perche', 'quanto', 'quale', 'che']
    question_word_count = sum(1 for word in words if word in question_words)

    # Connettori logici e congiunzioni
    conjunctions = ['e', 'o', 'ma', 'però', 'mentre', 'dopo', 'prima', 'quando', 'dove', 'se']
    conjunction_count = sum(1 for word in words if word in conjunctions)

    # Indica presenza di clausole multiple
    has_multiple_clauses = conjunction_count > 0 or ',' in query

    return {
        'word_count': len(words),
        'question_words': question_word_count,
        'conjunctions': conjunction_count,
        'has_multiple_clauses': has_multiple_clauses
    }


def analyze_query_semantics(query: str) -> Dict[str, Any]:
    """
    Analizza la complessità semantica della query.

    Args:
        query: Query dell'utente

    Returns:
        Dizionario con metriche semantiche
    """
    if not query or not query.strip():
        return {
            'entities_count': 0,
            'comparative_indicators': 0,
            'temporal_indicators': 0
        }

    query_lower = query.lower()

    # Stima entità (parole con maiuscola iniziale che non sono inizio frase)
    words = query.split()
    entities = sum(1 for i, word in enumerate(words)
                   if i > 0 and word[0].isupper() and len(word) > 2)

    # Indicatori comparativi
    comparative_words = ['confronta', 'paragona', 'differenza', 'versus', 'vs', 'rispetto', 'meglio', 'peggio']
    comparative_count = sum(1 for word in comparative_words if word in query_lower)

    # Indicatori temporali
    temporal_words = ['prima', 'dopo', 'durante', 'ieri', 'oggi', 'domani', 'anno', 'mese', 'giorno']
    temporal_count = sum(1 for word in temporal_words if word in query_lower)

    return {
        'entities_count': entities,
        'comparative_indicators': comparative_count,
        'temporal_indicators': temporal_count
    }


def calculate_complexity_score(query: str) -> Tuple[float, Dict[str, Any]]:
    """
    Calcola score di complessità basato su analisi strutturale e semantica.

    Args:
        query: Query dell'utente

    Returns:
        Tuple con (complexity_score, debug_metrics)
    """
    if not query or not query.strip():
        return 0.0, {'error': 'Query vuota'}

    # Analisi componenti
    structure = analyze_query_structure(query)
    semantics = analyze_query_semantics(query)

    # Calcolo score con pesi
    score = 0.0

    # Peso lunghezza (max contributo: 3.0)
    score += min(structure['word_count'] / 4.0, 3.0)

    # Peso parole interrogative (max contributo: 2.0)
    score += min(structure['question_words'] * 0.5, 2.0)

    # Peso congiunzioni (max contributo: 2.0)
    score += min(structure['conjunctions'] * 0.8, 2.0)

    # Peso clausole multiple (+1.0)
    if structure['has_multiple_clauses']:
        score += 1.0

    # Peso entità (max contributo: 1.5)
    score += min(semantics['entities_count'] * 0.3, 1.5)

    # Peso indicatori comparativi/temporali (max contributo: 1.0)
    score += min((semantics['comparative_indicators'] + semantics['temporal_indicators']) * 0.5, 1.0)

    # Metriche debug
    debug_metrics = {
        'complexity_score': round(score, 2),
        'structure_analysis': structure,
        'semantic_analysis': semantics,
        'score_breakdown': {
            'length_contribution': min(structure['word_count'] / 4.0, 3.0),
            'question_words_contribution': min(structure['question_words'] * 0.5, 2.0),
            'conjunctions_contribution': min(structure['conjunctions'] * 0.8, 2.0),
            'clauses_contribution': 1.0 if structure['has_multiple_clauses'] else 0.0,
            'entities_contribution': min(semantics['entities_count'] * 0.3, 1.5),
            'indicators_contribution': min(
                (semantics['comparative_indicators'] + semantics['temporal_indicators']) * 0.5, 1.0)
        }
    }

    return score, debug_metrics


def classify_query_complexity(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Classifica la complessità della query in simple/medium/complex.

    Args:
        query: Query dell'utente

    Returns:
        Tuple con (complexity_level, debug_metrics)
    """
    score, debug_metrics = calculate_complexity_score(query)

    # Classificazione basata su score
    if score <= 2.0:
        complexity = 'simple'
    elif score <= 4.0:
        complexity = 'medium'
    else:
        complexity = 'complex'

    debug_metrics['complexity_classification'] = complexity
    debug_metrics['classification_thresholds'] = {
        'simple': '≤ 2.0',
        'medium': '2.1 - 4.0',
        'complex': '> 4.0'
    }

    return complexity, debug_metrics


def get_percentage_for_complexity(complexity: str) -> float:
    """
    Mappa complexity level a percentuale di documenti da recuperare.

    Args:
        complexity: Livello di complessità ('simple'|'medium'|'complex')

    Returns:
        Percentuale da applicare al totale documenti
    """
    percentages = {
        'simple': 0.05,  # 5%
        'medium': 0.10,  # 10%
        'complex': 0.15  # 15%
    }

    return percentages.get(complexity, 0.10)  # Default: medium


def estimate_max_documents_for_tokens(available_tokens: int, avg_doc_tokens: int = 100) -> int:
    """
    Stima il numero massimo di documenti che possono essere inclusi nei token disponibili.

    Args:
        available_tokens: Token disponibili per document context
        avg_doc_tokens: Stima media token per documento (default: 100)

    Returns:
        Numero massimo di documenti
    """
    if available_tokens <= 0 or avg_doc_tokens <= 0:
        return 3  # Minimo garantito

    max_docs = available_tokens // avg_doc_tokens
    return max(3, max_docs)  # Minimo 3 documenti


def calculate_available_tokens_for_documents(
        base_template: str,
        current_query: str,
        conversation_context: str,
        model_context_window: int = 2048
) -> int:
    """
    Calcola token disponibili per document context dopo aver riservato spazio per componenti fissi.

    Args:
        base_template: Template base del SystemMessage
        current_query: Query corrente dell'utente
        conversation_context: Contesto conversazione
        model_context_window: Context window totale del modello

    Returns:
        Token disponibili per document context
    """
    # Calcola token utilizzati da componenti fissi
    template_tokens = estimate_token_count(base_template)
    query_tokens = estimate_token_count(current_query)
    conversation_tokens = estimate_token_count(conversation_context)

    # Buffer di sicurezza (5% del context window)
    safety_buffer = max(50, int(model_context_window * 0.05))

    # Token utilizzati
    used_tokens = template_tokens + query_tokens + conversation_tokens + safety_buffer

    # Token disponibili per documenti
    available_tokens = model_context_window - used_tokens

    return max(0, available_tokens)


def calculate_dynamic_k_with_limits(
        query: str,
        total_documents: int,
        base_template: str = "",
        conversation_context: str = "",
        model_context_window: int = 2048
) -> Tuple[int, Dict[str, Any]]:
    """
    Orchestratore principale: calcola k dinamico considerando complexity e context window limits.

    Args:
        query: Query dell'utente
        total_documents: Numero totale documenti nel vector store
        base_template: Template base SystemMessage
        conversation_context: Contesto conversazione
        model_context_window: Context window del modello

    Returns:
        Tuple con (k_final, debug_info)
    """
    # 1. Classifica complessità query
    complexity, complexity_metrics = classify_query_complexity(query)

    # 2. Ottieni percentuale per complexity
    percentage = get_percentage_for_complexity(complexity)

    # 3. Calcola k ideale basato su percentuale
    k_ideal = max(3, int(total_documents * percentage))

    # 4. Calcola token disponibili per documenti (priorità: query > conversation > documents)
    available_tokens = calculate_available_tokens_for_documents(
        base_template, query, conversation_context, model_context_window
    )

    # 5. Calcola k massimo per context window
    k_max_by_context = estimate_max_documents_for_tokens(available_tokens)

    # 6. K finale: minimo tra ideale e limit context window
    k_final = min(k_ideal, k_max_by_context)

    # 7. Garantisci minimo assoluto
    k_final = max(3, k_final)

    # Debug info completo
    debug_info = {
        # Risultato finale
        'k_final': k_final,
        'complexity_level': complexity,
        'percentage_applied': percentage,

        # Calcoli intermedi
        'total_documents': total_documents,
        'k_ideal_by_percentage': k_ideal,
        'k_max_by_context_window': k_max_by_context,

        # Context window analysis
        'context_window_analysis': {
            'total_context_window': model_context_window,
            'base_template_tokens': estimate_token_count(base_template),
            'query_tokens': estimate_token_count(query),
            'conversation_tokens': estimate_token_count(conversation_context),
            'available_for_documents': available_tokens,
            'safety_buffer': max(50, int(model_context_window * 0.05))
        },

        # Complexity analysis dettagliato
        'complexity_analysis': complexity_metrics,

        # Limitazioni applicate
        'limitations': {
            'limited_by_context_window': k_final == k_max_by_context,
            'limited_by_minimum': k_final == 3,
            'using_ideal_k': k_final == k_ideal
        }
    }

    return k_final, debug_info


def get_debug_metrics_summary(debug_info: Dict[str, Any]) -> str:
    """
    Formatta metriche debug in formato leggibile per troubleshooting.

    Args:
        debug_info: Info debug da calculate_dynamic_k_with_limits

    Returns:
        Stringa formattata con summary delle metriche
    """
    if not debug_info:
        return "Nessuna metrica debug disponibile"

    summary_lines = [
        f"🎯 K finale: {debug_info['k_final']} (complexity: {debug_info['complexity_level']})",
        f"📊 Percentuale applicata: {debug_info['percentage_applied'] * 100:.1f}%",
        f"📄 Documenti totali: {debug_info['total_documents']}",
        f"🔍 K ideale: {debug_info['k_ideal_by_percentage']}, K max context: {debug_info['k_max_by_context_window']}"
    ]

    # Context window info
    ctx = debug_info['context_window_analysis']
    summary_lines.extend([
        f"🧠 Context window: {ctx['total_context_window']} token",
        f"📝 Token: template({ctx['base_template_tokens']}) + query({ctx['query_tokens']}) + conversation({ctx['conversation_tokens']})",
        f"📂 Token disponibili per documenti: {ctx['available_for_documents']}"
    ])

    # Limitazioni
    limitations = debug_info['limitations']
    if limitations['limited_by_context_window']:
        summary_lines.append("⚠️ Limitato da context window")
    if limitations['limited_by_minimum']:
        summary_lines.append("⚠️ Applicato minimo di 3 documenti")
    if limitations['using_ideal_k']:
        summary_lines.append("✅ Usando k ideale")

    return "\n".join(summary_lines)