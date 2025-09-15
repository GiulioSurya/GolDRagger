from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from enum import Enum
import json
import threading
from dataclasses import dataclass
from uuid import uuid4
from multi_agent_system.core.messages import SystemMessage, create_system_message


class ChangeType(Enum):
    """Tipi di cambiamenti sulla blackboard"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    CLEAR = "clear"


@dataclass
class BlackboardChange:
    """Rappresenta un cambiamento sulla blackboard"""
    change_type: ChangeType
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    changed_by: Optional[str] = None  # ID dell'agent che ha fatto il cambio


class BlackboardEntry(BaseModel):
    """Entry singola nella blackboard"""
    key: str = Field(..., description="Chiave dell'entry")
    value: Any = Field(..., description="Valore dell'entry")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = Field(default=None, description="Agent che ha creato l'entry")
    updated_by: Optional[str] = Field(default=None, description="Agent che ha aggiornato l'entry")
    access_count: int = Field(default=0, description="Numero di volte che è stata letta")
    tags: List[str] = Field(default_factory=list, description="Tag per categorizzazione")
    access_history: List[Dict[str, Any]] = Field(default_factory=list, description="Storia degli accessi")

    def mark_accessed(self, accessed_by: Optional[str] = None):
        """Marca l'entry come accessata"""
        self.access_count += 1
        if accessed_by:
            self.access_history.append({
                'agent': accessed_by,
                'timestamp': datetime.now(timezone.utc)
            })

            # Mantieni solo gli ultimi 10
            if len(self.access_history) > 10:
                self.access_history = self.access_history[-10:]


class BlackBoard():
    """
    Lavagna condivisa tra agent del sistema multi-agent.
    Fornisce storage thread-safe con sistema di notifiche.
    """

    def __init__(self):

        # Storage principale per le key
        self._storage: Dict[str, BlackboardEntry] = {}
        self._lock = threading.RLock()  # ReentrantLock per accesso thread-safe
        # Sistema di notifiche
        self._observers: Set[Callable[[BlackboardChange], None]] = set()
        # Storia dei cambiamenti (utile per debugging)
        self._change_history: List[BlackboardChange] = []
        self._max_history_size = 1000  # Limita la storia per memoria
        # Statistiche
        self._stats = {
            'total_reads': 0,
            'total_writes': 0,
            'total_deletes': 0,
            'observers_count': 0
        }

    def update(self, key: str, value: Any, updated_by: Optional[str] = None, tags: List[str] = None) -> bool:
        """
        Aggiorna o crea un valore sulla blackboard

        Args:
            key: Chiave per il valore
            value: Valore da salvare
            updated_by: ID dell'agent che fa l'update
            tags: Tag opzionali per categorizzazione

        Returns:
            bool: True se l'operazione è riuscita
        """
        with self._lock:
            try:
                now = datetime.now(timezone.utc)
                old_value = None
                change_type = ChangeType.CREATE

                # Controlla se la chiave esiste già
                if key in self._storage:
                    old_value = self._storage[key].value
                    change_type = ChangeType.UPDATE

                    # Aggiorna entry esistente
                    self._storage[key].value = value
                    self._storage[key].updated_at = now
                    self._storage[key].updated_by = updated_by
                    if tags:
                        self._storage[key].tags = tags
                else:
                    # Crea nuova entry
                    self._storage[key] = BlackboardEntry(
                        key=key,
                        value=value,
                        created_at=now,
                        updated_at=now,
                        created_by=updated_by,
                        updated_by=updated_by,
                        tags=tags or []
                    )

                # Crea record del cambiamento
                change = BlackboardChange(
                    change_type=change_type,
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    timestamp=now,
                    changed_by=updated_by
                )

                # Aggiunge alla storia
                self._add_to_history(change)

                # Aggiorna statistiche
                self._stats['total_writes'] += 1

                # Notifica observers
                self._notify_observers(change) ##questo viene chiamato più volte, controllare

                return True

            except Exception as e:
                print(f"Error updating blackboard key '{key}': {str(e)}")
                return False

    def get(self, key: str, accessed_by: Optional[str] = None, default: Any = None) -> Any:
        """
        Recupera un valore dalla blackboard

        Args:
            key: Chiave del valore da recuperare
            accessed_by: ID dell'agent che accede
            default: Valore di default se la chiave non esiste

        Returns:
            Any: Valore associato alla chiave o default
        """
        with self._lock:
            try:
                if key not in self._storage:
                    return default

                entry = self._storage[key]
                entry.mark_accessed(accessed_by)

                # Aggiorna statistiche
                self._stats['total_reads'] += 1

                return entry.value

            except Exception as e:
                print(f"Error getting blackboard key '{key}': {str(e)}")
                return default

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Recupera informazioni complete su un'entry (metadati inclusi)

        Args:
            key: Chiave dell'entry

        Returns:
            Dict: Informazioni complete dell'entry o None
        """
        with self._lock:
            if key not in self._storage:
                return None

            entry = self._storage[key]
            return {
                'key': entry.key,
                'value': entry.value,
                'created_at': entry.created_at.isoformat(),
                'updated_at': entry.updated_at.isoformat(),
                'created_by': entry.created_by,
                'updated_by': entry.updated_by,
                'access_count': entry.access_count,
                'tags': entry.tags,
                'access_history': entry.access_history
            }

    def delete(self, key: str, deleted_by: Optional[str] = None) -> bool:
        """
        Elimina una chiave dalla blackboard

        Args:
            key: Chiave da eliminare
            deleted_by: ID dell'agent che elimina

        Returns:
            bool: True se eliminata, False se non esisteva
        """
        with self._lock:
            if key not in self._storage:
                return False

            try:
                old_value = self._storage[key].value
                del self._storage[key]

                # Crea record del cambiamento
                change = BlackboardChange(
                    change_type=ChangeType.DELETE,
                    key=key,
                    old_value=old_value,
                    new_value=None,
                    timestamp=datetime.now(timezone.utc),
                    changed_by=deleted_by
                )

                self._add_to_history(change)
                self._stats['total_deletes'] += 1
                self._notify_observers(change)

                return True

            except Exception as e:
                print(f"Error deleting blackboard key '{key}': {str(e)}")
                return False

    def get_all(self, include_metadata: bool = False) -> Dict[str, Any]:
        """
        Recupera tutti i dati dalla blackboard

        Args:
            include_metadata: Se includere metadati delle entry

        Returns:
            Dict: Tutti i dati sulla blackboard
        """
        with self._lock:
            if include_metadata:
                return {
                    key: {
                        'value': entry.value,
                        'created_at': entry.created_at.isoformat(),
                        'updated_at': entry.updated_at.isoformat(),
                        'created_by': entry.created_by,
                        'updated_by': entry.updated_by,
                        'access_count': entry.access_count,
                        'tags': entry.tags,
                        'access_history': entry.access_history
                    }
                    for key, entry in self._storage.items()
                }
            else:
                return {key: entry.value for key, entry in self._storage.items()}

    def clear(self, cleared_by: Optional[str] = None):
        """
        Pulisce tutta la blackboard

        Args:
            cleared_by: ID dell'agent che pulisce
        """
        with self._lock:
            old_data = self.get_all()
            self._storage.clear()

            # Crea record del cambiamento
            change = BlackboardChange(
                change_type=ChangeType.CLEAR,
                key="*",
                old_value=old_data,
                new_value=None,
                timestamp=datetime.now(timezone.utc),
                changed_by=cleared_by
            )

            self._add_to_history(change)
            self._notify_observers(change)

    def keys(self) -> List[str]:
        """Lista di tutte le chiavi"""
        with self._lock:
            return list(self._storage.keys())

    def has_key(self, key: str) -> bool:
        """Controlla se una chiave esiste"""
        with self._lock:
            return key in self._storage

    def size(self) -> int:
        """Numero di entry nella blackboard"""
        with self._lock:
            return len(self._storage)

    def find_by_tags(self, tags: List[str], match_all: bool = False) -> Dict[str, Any]:
        """
        Trova entry per tag

        Args:
            tags: Tag da cercare
            match_all: Se True, deve matchare tutti i tag. Se False, almeno uno

        Returns:
            Dict: Entry che matchano i criteri
        """
        with self._lock:
            results = {}

            for key, entry in self._storage.items():
                entry_tags = set(entry.tags)
                search_tags = set(tags)

                if match_all:
                    # Deve avere tutti i tag
                    if search_tags.issubset(entry_tags):
                        results[key] = entry.value
                else:
                    # Deve avere almeno uno dei tag
                    if search_tags.intersection(entry_tags):
                        results[key] = entry.value

            return results

    def subscribe_to_changes(self, callback: Callable[[BlackboardChange], None]):
        """
        Registra un callback per essere notificato dei cambiamenti

        Args:
            callback: Funzione chiamata quando c'è un cambiamento
        """
        with self._lock:
            self._observers.add(callback)
            self._stats['observers_count'] = len(self._observers)

    def unsubscribe(self, callback: Callable[[BlackboardChange], None]):
        """
        Rimuove un callback dalle notifiche

        Args:
            callback: Callback da rimuovere
        """
        with self._lock:
            self._observers.discard(callback)
            self._stats['observers_count'] = len(self._observers)

    def get_change_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Recupera la storia dei cambiamenti

        Args:
            limit: Numero massimo di cambiamenti da restituire

        Returns:
            List: Storia dei cambiamenti (più recenti primi)
        """
        with self._lock:
            history = self._change_history.copy()
            history.reverse()  # Più recenti primi

            if limit:
                history = history[:limit]

            # Converte in formato serializable
            return [
                {
                    'change_type': change.change_type.value,
                    'key': change.key,
                    'old_value': change.old_value,
                    'new_value': change.new_value,
                    'timestamp': change.timestamp.isoformat(),
                    'changed_by': change.changed_by
                }
                for change in history
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Statistiche di utilizzo della blackboard"""
        with self._lock:
            return {
                **self._stats.copy(),
                'total_entries': len(self._storage),
                'history_size': len(self._change_history)
            }

    # ===== METODI PER TASK MANAGEMENT =====

    def create_task(self, assigned_to: str, task_type: str, task_data: Dict,
                    created_by: str) -> str:
        """
        Crea un nuovo task con ID univoco per un agent specifico

        Args:
            assigned_to: ID dell'agent a cui assegnare il task
            task_type: Tipo di task (es. "document_retrieval", "sentiment_analysis")
            task_data: Dati necessari per il task
            created_by: ID di chi crea il task

        Returns:
            str: Task ID univoco
        """
        task_id = str(uuid4())
        task = {
            "task_id": task_id,
            "assigned_to": assigned_to,
            "task_type": task_type,
            "task_data": task_data,
            "status": "pending",
            "created_by": created_by,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None,
            "completed_at": None,
            "result": None,
            "execution_time": None
        }

        # Salva con chiave specifica per l'agent
        key = f"task_{assigned_to}_{task_id}"
        self.update(key, task, updated_by=created_by,
                    tags=["task", "pending", assigned_to, task_type])

        return task_id

    def update_task_result(self, task_id: str, agent_id: str, result: Dict,
                           status: str = "completed", execution_time: Optional[float] = None) -> bool:
        """
        Agent aggiorna risultato del task

        Args:
            task_id: ID del task
            agent_id: ID dell'agent che aggiorna
            result: Risultato dell'elaborazione
            status: Nuovo status ("completed", "failed", "in_progress")
            execution_time: Tempo di esecuzione in secondi

        Returns:
            bool: True se aggiornato con successo
        """
        key = f"task_{agent_id}_{task_id}"
        task = self.get(key, accessed_by=agent_id)

        if task:
            now = datetime.now(timezone.utc).isoformat()
            task["result"] = result
            task["status"] = status
            task["updated_at"] = now
            task["execution_time"] = execution_time

            if status == "completed":
                task["completed_at"] = now
                task["completed_by"] = agent_id

            # Aggiorna tag
            new_tags = ["task", status, agent_id, task["task_type"]]

            self.update(key, task, updated_by=agent_id, tags=new_tags)
            return True
        return False

    def get_task_status(self, task_id: str, agent_id: str) -> Optional[Dict]:
        """
        Manager controlla status di un task

        Args:
            task_id: ID del task
            agent_id: ID dell'agent assegnato

        Returns:
            Dict: Status completo del task o None se non trovato
        """
        key = f"task_{agent_id}_{task_id}"
        return self.get(key, accessed_by="manager")

    def get_tasks_by_agent(self, agent_id: str, status: Optional[str] = None) -> List[Dict]:
        """
        Recupera tutti i task di un agent

        Args:
            agent_id: ID dell'agent
            status: Filtra per status specifico (opzionale)

        Returns:
            List[Dict]: Lista dei task
        """
        tasks = []
        prefix = f"task_{agent_id}_"

        for key in self.keys():
            if key.startswith(prefix):
                task = self.get(key, accessed_by="system")
                if status is None or task.get("status") == status:
                    tasks.append(task)

        return tasks

    def get_tasks_by_status(self, status: str) -> Dict[str, List[Dict]]:
        """
        Recupera tutti i task per status

        Args:
            status: Status da cercare ("pending", "completed", "failed")

        Returns:
            Dict: Mapping agent_id -> List[task]
        """
        result = {}

        for key in self.keys():
            if key.startswith("task_"):
                task = self.get(key, accessed_by="system")
                if task and task.get("status") == status:
                    agent_id = task["assigned_to"]
                    if agent_id not in result:
                        result[agent_id] = []
                    result[agent_id].append(task)

        return result

    def get_task_stats(self) -> Dict[str, Any]:
        """Statistiche sui task"""
        all_tasks = []

        for key in self.keys():
            if key.startswith("task_"):
                task = self.get(key, accessed_by="system")
                if task:
                    all_tasks.append(task)

        # Conta per status
        status_counts = {}
        agent_counts = {}
        type_counts = {}

        for task in all_tasks:
            status = task.get("status", "unknown")
            agent = task.get("assigned_to", "unknown")
            task_type = task.get("task_type", "unknown")

            status_counts[status] = status_counts.get(status, 0) + 1
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            type_counts[task_type] = type_counts.get(task_type, 0) + 1

        return {
            "total_tasks": len(all_tasks),
            "by_status": status_counts,
            "by_agent": agent_counts,
            "by_type": type_counts
        }

    # ===== METODI PER SYSTEM INSTRUCTIONS =====

    def add_system_instruction(self, agent_id: str, instruction_text: str,
                               instruction_type: str = "instruction",
                               expires_in_minutes: Optional[int] = None,
                               priority_level: int = 1,
                               added_by: Optional[str] = None) -> str:
        """
        Aggiunge istruzione di sistema per un agent specifico

        Args:
            agent_id: ID dell'agent destinatario
            instruction_text: Testo dell'istruzione
            instruction_type: Tipo di istruzione
            expires_in_minutes: Minuti dopo cui l'istruzione scade
            priority_level: Priorità dell'istruzione (1=alta, 10=bassa)
            added_by: Chi ha aggiunto l'istruzione

        Returns:
            str: ID del SystemMessage creato
        """
        # Crea SystemMessage
        sys_msg = create_system_message(
            target_agent=agent_id,
            instruction_text=instruction_text,
            instruction_type=instruction_type,
            expires_in_minutes=expires_in_minutes,
            priority_level=priority_level
        )

        # Chiave per le istruzioni di questo agent
        key = f"system_instructions_{agent_id}"

        # Recupera istruzioni esistenti
        current_instructions = self.get(key, accessed_by=added_by, default=[])

        # Aggiunge nuova istruzione
        instruction_entry = {
            'id': sys_msg.message_id,
            'message': sys_msg,
            'active': True,
            'created_at': sys_msg.timestamp.isoformat(),
            'added_by': added_by
        }

        current_instructions.append(instruction_entry)

        # Salva sulla blackboard
        self.update(key, current_instructions, updated_by=added_by,
                    tags=['system_instructions', 'agent_config'])

        return sys_msg.message_id

    def get_system_instructions_for_agent(self, agent_id: str,
                                          active_only: bool = True) -> List[SystemMessage]:
        """
        Recupera tutte le istruzioni attive per un agent

        Args:
            agent_id: ID dell'agent
            active_only: Se True, solo quelle attive

        Returns:
            List[SystemMessage]: Lista delle istruzioni
        """
        key = f"system_instructions_{agent_id}"
        instructions = self.get(key, accessed_by=f"system_query_{agent_id}", default=[])

        result = []
        now = datetime.now(timezone.utc)

        for instr in instructions:
            # Controlla se attiva
            if active_only and not instr.get('active', True):
                continue

            sys_msg = instr['message']

            # Controlla scadenza
            if sys_msg.expires_at and now > sys_msg.expires_at:
                # Istruzione scaduta, disattivala
                instr['active'] = False
                continue

            result.append(sys_msg)

        # Ordina per priorità (1=alta priorità viene prima)
        result.sort(key=lambda msg: msg.priority_level)

        return result

    def remove_system_instruction(self, agent_id: str, instruction_id: str,
                                  removed_by: Optional[str] = None) -> bool:
        """
        Rimuove/disattiva un'istruzione specifica

        Args:
            agent_id: ID dell'agent
            instruction_id: ID dell'istruzione da rimuovere
            removed_by: Chi rimuove l'istruzione

        Returns:
            bool: True se trovata e rimossa
        """
        key = f"system_instructions_{agent_id}"
        instructions = self.get(key, accessed_by=removed_by, default=[])

        for instr in instructions:
            if instr['id'] == instruction_id:
                instr['active'] = False
                instr['removed_by'] = removed_by
                instr['removed_at'] = datetime.now(timezone.utc).isoformat()

                # Aggiorna sulla blackboard
                self.update(key, instructions, updated_by=removed_by,
                            tags=['system_instructions', 'agent_config'])
                return True

        return False

    def clear_system_instructions(self, agent_id: str,
                                  cleared_by: Optional[str] = None):
        """
        Pulisce tutte le istruzioni per un agent

        Args:
            agent_id: ID dell'agent
            cleared_by: Chi pulisce le istruzioni
        """
        key = f"system_instructions_{agent_id}"
        instructions = self.get(key, accessed_by=cleared_by, default=[])

        # Disattiva tutte le istruzioni
        for instr in instructions:
            instr['active'] = False
            instr['cleared_by'] = cleared_by
            instr['cleared_at'] = datetime.now(timezone.utc).isoformat()

        # Aggiorna sulla blackboard
        self.update(key, instructions, updated_by=cleared_by,
                    tags=['system_instructions', 'agent_config'])

    def get_all_system_instructions(self) -> Dict[str, List[SystemMessage]]:
        """
        Tutte le istruzioni di sistema per tutti gli agent

        Returns:
            Dict: Mapping agent_id -> List[SystemMessage]
        """
        result = {}

        # Trova tutte le chiavi di system instructions
        for key in self.keys():
            if key.startswith('system_instructions_'):
                agent_id = key.replace('system_instructions_', '')
                instructions = self.get_system_instructions_for_agent(agent_id)
                if instructions:
                    result[agent_id] = instructions

        return result

    def get_system_instruction_stats(self) -> Dict[str, Any]:
        """Statistiche sulle system instructions"""
        all_instructions = self.get_all_system_instructions()

        total_instructions = sum(len(instr_list) for instr_list in all_instructions.values())
        agents_with_instructions = len(all_instructions)

        # Conteggio per tipo
        type_counts = {}
        for instr_list in all_instructions.values():
            for instr in instr_list:
                instr_type = instr.instruction_type
                type_counts[instr_type] = type_counts.get(instr_type, 0) + 1

        return {
            'total_instructions': total_instructions,
            'agents_with_instructions': agents_with_instructions,
            'instruction_types': type_counts,
            'agents': list(all_instructions.keys())
        }

    # ===== METODI PRIVATI =====

    def _notify_observers(self, change: BlackboardChange):
        """Notifica tutti gli observers di un cambiamento"""
        # Crea copia per evitare problemi di concorrenza
        observers_copy = self._observers.copy()

        for observer in observers_copy:
            try:
                observer(change)
            except Exception as e:
                print(f"Error notifying observer: {str(e)}")

    def _add_to_history(self, change: BlackboardChange):
        """Aggiunge un cambiamento alla storia"""
        self._change_history.append(change)

        # Mantiene la storia sotto il limite
        if len(self._change_history) > self._max_history_size:
            self._change_history = self._change_history[-self._max_history_size:]

    def export_to_json(self, file_path: str, include_metadata: bool = True) -> bool:
        """
        Esporta la blackboard in JSON

        Args:
            file_path: Percorso del file
            include_metadata: Se includere i metadati

        Returns:
            bool: True se l'export è riuscito
        """
        try:
            data = self.get_all(include_metadata=include_metadata)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"Error exporting blackboard to JSON: {str(e)}")
            return False

    def import_from_json(self, file_path: str, imported_by: Optional[str] = None) -> bool:
        """
        Importa dati da JSON nella blackboard

        Args:
            file_path: Percorso del file JSON
            imported_by: ID di chi fa l'import

        Returns:
            bool: True se l'import è riuscito
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Se il JSON ha metadati, estrae solo i valori
            for key, value in data.items():
                if isinstance(value, dict) and 'value' in value:
                    # JSON con metadati
                    actual_value = value['value']
                    tags = value.get('tags', [])
                else:
                    # JSON semplice
                    actual_value = value
                    tags = []

                self.update(key, actual_value, updated_by=imported_by, tags=tags)

            return True

        except Exception as e:
            print(f"Error importing blackboard from JSON: {str(e)}")
            return False


# ===== UTILITY FUNCTIONS =====

def create_blackboard() -> BlackBoard:
    """Factory per creare una blackboard"""
    return BlackBoard()


def create_observer_callback(agent_id: str, callback_func: Callable) -> Callable[[BlackboardChange], None]:
    """
    Crea un callback personalizzato per un agent

    Args:
        agent_id: ID dell'agent
        callback_func: Funzione da chiamare con (agent_id, change)

    Returns:
        Callable: Callback ready per subscribe
    """

    def observer(change: BlackboardChange):
        try:
            callback_func(agent_id, change)
        except Exception as e:
            print(f"Error in callback for agent {agent_id}: {str(e)}")

    return observer


if __name__ == "__main__":
    print("SISTEMA MULTI-AGENT CON TASK MANAGEMENT")
    print("=" * 60)

    bb = create_blackboard()

    # SCENARIO: Manager assegna task con ID a RAG e Analysis Agent

    # 1. Manager crea task per RAG Agent
    rag_task_id = bb.create_task(
        assigned_to="rag_agent",
        task_type="document_retrieval",
        task_data={
            "query": "Python machine learning tutorials",
            "top_k": 5,
            "user_id": "user123"
        },
        created_by="manager"
    )
    print(f"✓ Manager creato task RAG con ID: {rag_task_id}")

    # 2. RAG Agent legge il suo task (simulazione notifica)
    rag_task = bb.get_task_status(rag_task_id, "rag_agent")
    print(f"✓ RAG Agent ricevuto task: {rag_task['task_type']}")

    # 3. RAG Agent completa il task
    retrieved_docs = [
        {"id": "doc1", "title": "Python ML Basics", "score": 0.95},
        {"id": "doc2", "title": "Scikit-learn Guide", "score": 0.89},
        {"id": "doc3", "title": "TensorFlow Tutorial", "score": 0.84}
    ]

    bb.update_task_result(
        task_id=rag_task_id,
        agent_id="rag_agent",
        result={
            "documents": retrieved_docs,
            "total_found": len(retrieved_docs),
            "query_used": rag_task["task_data"]["query"]
        },
        status="completed",
        execution_time=2.3
    )
    print(f"✓ RAG Agent completato task in {2.3}s")

    # 4. Manager controlla status del task RAG
    rag_status = bb.get_task_status(rag_task_id, "rag_agent")
    if rag_status["status"] == "completed":
        print(f"✓ Manager verificato completamento task RAG")

        # 5. Manager crea task per Analysis Agent basandosi sui risultati RAG
        analysis_task_id = bb.create_task(
            assigned_to="analysis_agent",
            task_type="sentiment_analysis",
            task_data={
                "documents": rag_status["result"]["documents"],
                "analysis_type": "sentiment",
                "original_query": rag_status["task_data"]["query"]
            },
            created_by="manager"
        )
        print(f"✓ Manager creato task Analysis con ID: {analysis_task_id}")

    # 6. Analysis Agent completa il suo task
    analysis_task = bb.get_task_status(analysis_task_id, "analysis_agent")

    bb.update_task_result(
        task_id=analysis_task_id,
        agent_id="analysis_agent",
        result={
            "overall_sentiment": "positive",
            "confidence": 0.87,
            "doc_sentiments": [
                {"doc_id": "doc1", "sentiment": "positive", "score": 0.92},
                {"doc_id": "doc2", "sentiment": "positive", "score": 0.85},
                {"doc_id": "doc3", "sentiment": "neutral", "score": 0.68}
            ]
        },
        status="completed",
        execution_time=1.5
    )
    print(f"✓ Analysis Agent completato task in {1.5}s")

    print("\n" + "=" * 60)
    print("TASK TRACKING E STATISTICHE")
    print("=" * 60)

    # Statistiche task
    task_stats = bb.get_task_stats()
    print(f"📊 Task Statistics:")
    print(f"   • Totali: {task_stats['total_tasks']}")
    print(f"   • Per status: {task_stats['by_status']}")
    print(f"   • Per agent: {task_stats['by_agent']}")
    print(f"   • Per tipo: {task_stats['by_type']}")

    # Task completati
    completed_tasks = bb.get_tasks_by_status("completed")
    print(f"\n✅ Task Completati:")
    for agent_id, tasks in completed_tasks.items():
        print(f"   {agent_id}:")
        for task in tasks:
            exec_time = task.get('execution_time', 'N/A')
            print(f"      - {task['task_type']} (ID: {task['task_id'][:8]}..., tempo: {exec_time}s)")

    # Chain di task (tracciabilità)
    print(f"\n🔗 Chain di Task:")
    print(f"   1. RAG Task: {rag_task_id[:8]}... → Recuperati {len(retrieved_docs)} documenti")
    print(f"   2. Analysis Task: {analysis_task_id[:8]}... → Sentiment: positive (87% confidence)")

    print("\n" + "=" * 60)
    print("SYSTEM INSTRUCTIONS")
    print("=" * 60)

    # Aggiungi system instructions
    bb.add_system_instruction(
        agent_id="rag_agent",
        instruction_text="Focus on recent ML documents from 2024",
        instruction_type="constraint",
        priority_level=1,
        added_by="manager"
    )

    bb.add_system_instruction(
        agent_id="analysis_agent",
        instruction_text="Provide detailed confidence intervals",
        instruction_type="behavior",
        priority_level=2,
        added_by="user"
    )

    # Summary istruzioni
    instr_stats = bb.get_system_instruction_stats()
    print(f"🎯 System Instructions: {instr_stats['total_instructions']} totali")

    all_instructions = bb.get_all_system_instructions()
    for agent_id, instructions in all_instructions.items():
        print(f"   📝 {agent_id}: {len(instructions)} istruzioni")
        for instr in instructions:
            print(f"      - {instr.instruction_text} (priorità: {instr.priority_level})")

    print("\n" + "=" * 60)
    print("BLACKBOARD FINALE")
    print("=" * 60)

    # Overview finale
    final_stats = bb.get_stats()
    print(f"📈 BlackBoard Stats:")
    print(f"   • Entries: {final_stats['total_entries']}")
    print(f"   • Reads: {final_stats['total_reads']}")
    print(f"   • Writes: {final_stats['total_writes']}")

    # Task attivi vs completati
    all_task_keys = [k for k in bb.keys() if k.startswith("task_")]
    print(f"   • Task totali sulla board: {len(all_task_keys)}")

    # Ultime modifiche
    print(f"\n📜 Ultime modifiche:")
    recent_changes = bb.get_change_history(limit=5)
    for change in recent_changes:
        change_type = change['change_type']
        key = change['key'][:20] + "..." if len(change['key']) > 20 else change['key']
        by = change['changed_by']
        print(f"   • {change_type}: {key} by {by}")