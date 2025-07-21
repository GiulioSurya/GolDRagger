"""
Modulo: ChromaVectorStoreCreator
Descrizione: Crea vector store ChromaDB da PDF con embedding Ollama
"""

from typing import List
import os
import argparse
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from PyPDF2 import PdfReader


class ChromaVectorStoreCreator:
    """
    Crea e popola un vector store ChromaDB da documenti PDF.

    Utilizza OllamaEmbeddings per generare embedding e ChromaDB
    per memorizzare i vettori con ricerca semantica.
    """

    def __init__(self, collection_name: str = "rag_chunks_ita",
                 persist_directory: str = "./rag/chroma_db",
                 chunk_size: int = 450) -> None:
        """
        Inizializza il creatore del vector store.

        Args:
            collection_name: Nome della collection ChromaDB
            persist_directory: Directory per persistenza ChromaDB
            chunk_size: Dimensione dei chunk in caratteri
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.embeddings = None
        self.vector_store = None
        self._initialize_ollama_embeddings()
        self._initialize_chroma_client()

    def _initialize_ollama_embeddings(self) -> None:
        """Inizializza OllamaEmbeddings con modello nomic-embed-text."""
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

    def _initialize_chroma_client(self) -> None:
        """Inizializza client ChromaDB con persistenza."""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)

    def read_pdf(self, pdf_path: str) -> List[Document]:
        """
        Legge PDF e converte in documenti LangChain.

        Args:
            pdf_path: Percorso del file PDF

        Returns:
            Lista di documenti con metadata pagina

        Raises:
            FileNotFoundError: Se il file PDF non esiste
            Exception: Se errore nella lettura del PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File PDF non trovato: {pdf_path}")

        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Il file deve essere un PDF: {pdf_path}")

        reader = PdfReader(pdf_path)
        documents = []

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                doc = Document(
                    page_content=page_text.strip(),
                    metadata={"page": page_num + 1, "source": pdf_path}
                )
                documents.append(doc)

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide i documenti in chunk per embedding.

        Args:
            documents: Lista di documenti da dividere

        Returns:
            Lista di chunk con metadata
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=max(50, self.chunk_size // 9),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        return chunks

    def create_vector_store(self, chunks: List[Document]) -> Chroma:
        """
        Crea vector store ChromaDB dai chunk.

        Args:
            chunks: Lista di chunk da vettorizzare

        Returns:
            Vector store ChromaDB
        """
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name=self.collection_name
        )

        return self.vector_store

    def process_pdf_to_vectorstore(self, pdf_path: str) -> Chroma:
        """
        Pipeline completa: PDF -> documenti -> chunk -> vector store.

        Args:
            pdf_path: Percorso del file PDF

        Returns:
            Vector store ChromaDB popolato
        """
        documents = self.read_pdf(pdf_path)
        print(f"✓ Letti {len(documents)} pagine dal PDF")

        chunks = self.chunk_documents(documents)
        print(f"✓ Creati {len(chunks)} chunk (dimensione: {self.chunk_size} caratteri)")

        vector_store = self.create_vector_store(chunks)
        print(f"✓ Vector store creato con {len(chunks)} embedding")

        return vector_store

    def get_collection_info(self) -> dict:
        """
        Ottiene informazioni sulla collection ChromaDB.

        Returns:
            Informazioni sulla collection
        """
        if self.vector_store is None:
            return {"error": "Vector store non ancora creato"}

        collection = self.chroma_client.get_collection(self.collection_name)
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }


def parse_arguments() -> argparse.Namespace:
    """
    Parsing degli argomenti da command line.

    Returns:
        Namespace con gli argomenti parsati
    """
    parser = argparse.ArgumentParser(
        description="Crea vector store ChromaDB da documenti PDF con embedding Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python chroma_embender.py document.pdf
  python chroma_embender.py /path/to/document.pdf --chunk-size 600
  python chroma_embender.py document.pdf -c 300 --collection "my_docs"
        """
    )

    parser.add_argument(
        "pdf_path",
        help="Percorso del file PDF da processare"
    )

    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=450,
        help="Dimensione dei chunk in caratteri (default: 450)"
    )

    parser.add_argument(
        "--collection",
        default="rag_chunks_ita",
        help="Nome della collection ChromaDB (default: rag_chunks_ita)"
    )

    parser.add_argument(
        "--persist-dir",
        default="./rag/chroma_db",
        help="Directory per persistenza ChromaDB (default: ./rag/chroma_db)"
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Valida gli argomenti ricevuti.

    Args:
        args: Argomenti parsati

    Raises:
        ValueError: Se gli argomenti non sono validi
        FileNotFoundError: Se il file PDF non esiste
    """
    if args.chunk_size < 100 or args.chunk_size > 2000:
        raise ValueError(f"chunk_size deve essere tra 100 e 2000, ricevuto: {args.chunk_size}")

    if not os.path.exists(args.pdf_path):
        raise FileNotFoundError(f"File PDF non trovato: {args.pdf_path}")

    if not args.pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"Il file deve essere un PDF: {args.pdf_path}")


def main():
    """Funzione principale."""
    args = parse_arguments()
    validate_arguments(args)

    print(f"🚀 Avvio processing PDF: {os.path.basename(args.pdf_path)}")
    print(f"📏 Chunk size: {args.chunk_size} caratteri")
    print(f"📁 Collection: {args.collection}")

    creator = ChromaVectorStoreCreator(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        chunk_size=args.chunk_size
    )

    vector_store = creator.process_pdf_to_vectorstore(args.pdf_path)

    info = creator.get_collection_info()
    print(f"📊 Collection info: {info}")
    print("✅ Processing completato con successo!")


if __name__ == "__main__":
    main()