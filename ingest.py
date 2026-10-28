"""Index Markdown knowledge documents in a local persistent ChromaDB collection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import chromadb
from sentence_transformers import SentenceTransformer


CHUNK_SIZE: Final = 500
CHUNK_OVERLAP: Final = 50
MODEL_NAME: Final = "all-MiniLM-L6-v2"
COLLECTION_NAME: Final = "manuales_ti"


def dividir_en_chunks(
    texto: str,
    tamano: int = CHUNK_SIZE,
    solapamiento: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into fixed-size chunks with the requested overlap."""
    if tamano <= 0:
        raise ValueError("tamano debe ser mayor que cero")
    if solapamiento < 0 or solapamiento >= tamano:
        raise ValueError("solapamiento debe estar entre 0 y tamano - 1")

    paso = tamano - solapamiento
    return [
        texto[inicio : inicio + tamano]
        for inicio in range(0, len(texto), paso)
        if texto[inicio : inicio + tamano]
    ]


def indexar_carpeta_documentos(
    directorio: Path,
    coleccion: chromadb.Collection,
    modelo: SentenceTransformer,
) -> int:
    """Read Markdown files and upsert their chunks into ChromaDB."""
    archivos = sorted(directorio.glob("*.md"))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos .md en {directorio.resolve()}"
        )

    total_chunks = 0
    for archivo in archivos:
        contenido = archivo.read_text(encoding="utf-8")
        chunks = dividir_en_chunks(contenido)
        if not chunks:
            print(f"Advertencia: se omite el archivo vacío {archivo.name}")
            continue

        embeddings = modelo.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()
        ids = [f"{archivo.name}_chunk_{indice}" for indice in range(len(chunks))]
        metadatas = [
            {"fuente": archivo.name, "chunk_index": indice}
            for indice in range(len(chunks))
        ]

        # upsert permite ejecutar el script nuevamente sin duplicar chunks.
        coleccion.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"{archivo.name}: {len(chunks)} chunks indexados")

    return total_chunks


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Indexa documentos Markdown en una colección local de ChromaDB."
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path(__file__).resolve().parent / "docs_conocimiento",
        help="Carpeta que contiene los archivos .md.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parent / "chroma_db",
        help="Directorio de persistencia local de ChromaDB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()
    args.db.mkdir(parents=True, exist_ok=True)

    cliente = chromadb.PersistentClient(path=str(args.db))
    coleccion = cliente.get_or_create_collection(name=COLLECTION_NAME)
    modelo = SentenceTransformer(MODEL_NAME)
    total_chunks = indexar_carpeta_documentos(args.docs, coleccion, modelo)

    print(
        f"Ingestión completada: {total_chunks} chunks procesados. "
        f"Total en la colección: {coleccion.count()}"
    )


if __name__ == "__main__":
    main()
