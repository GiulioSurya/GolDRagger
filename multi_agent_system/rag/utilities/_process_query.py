"""
Modulo: QueryProcessor
Descrizione: Preprocessing semplice delle query utente
"""

from typing import Dict, Any
import re


class QueryProcessor:
    """
    Gestisce la pulizia e normalizzazione semplice delle query utente.

    Pulisce il testo e lo prepara per essere processato dal LLM
    senza sovraingegnerizzare con funzionalità complesse.
    """

    def __init__(self) -> None:
        """Inizializza il processore di query."""
        pass

    def process_query(self, query: str) -> str:
        """
        Processa completamente una query attraverso tutta la pipeline di pulizia.

        Args:
            query: Query grezza dell'utente

        Returns:
            Query completamente processata e pronta per l'embedding
        """
        return self._prepare_for_llm(query)

    def _clean_query(self, query: str) -> str:
        """
        Pulisce e normalizza la query.

        Args:
            query: Query grezza dell'utente

        Returns:
            Query pulita e normalizzata
        """
        # Pulizia completa
        cleaned = query.strip()
        cleaned = self._normalize_text(cleaned)
        cleaned = self._remove_excessive_punctuation(cleaned)
        return cleaned.strip()

    def _normalize_text(self, text: str) -> str:
        """
        Normalizza il testo (spazi, punteggiatura, maiuscole).

        Args:
            text: Testo da normalizzare

        Returns:
            Testo normalizzato
        """
        # Rimuove spazi multipli
        normalized = re.sub(r'\s+', ' ', text)

        # Rimuove spazi prima della punteggiatura
        normalized = re.sub(r'\s+([,.!?])', r'\1', normalized)

        # Rimuove caratteri speciali problematici mantenendo punteggiatura base
        normalized = re.sub(r'[^\w\s,.!?àèéìíîòóùú-]', '', normalized)

        return normalized.strip()

    def _remove_excessive_punctuation(self, text: str) -> str:
        """
        Rimuove punteggiatura eccessiva (???, !!!, ecc.).

        Args:
            text: Testo da processare

        Returns:
            Testo con punteggiatura normalizzata
        """
        # Normalizza punteggiatura multipla
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r',+', ',', text)

        return text

    def _prepare_for_llm(self, query: str) -> str:
        """
        Prepara la query per essere inviata al LLM.

        Args:
            query: Query dell'utente

        Returns:
            Query pronta per il LLM
        """
        # Pulizia completa
        prepared = self._clean_query(query)

        # Assicura che finisca con un punto interrogativo se sembra una domanda
        if prepared and not prepared.endswith(('?', '.', '!')):
            # Parole interrogative comuni
            question_words = ['come', 'cosa', 'quando', 'dove', 'perché', 'perche', 'quanto', 'quale', 'chi']
            if any(prepared.lower().startswith(word) for word in question_words):
                prepared += '?'

        return prepared