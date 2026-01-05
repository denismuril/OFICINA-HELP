"""
OFICINA-HELP - Script de Ingestão de PDFs
==========================================
Este script varre as subpastas em data/ (jeep, audi, porsche),
carrega os PDFs, divide o texto em chunks e cria índices FAISS separados.

Uso:
    python ingest.py

Autor: Sistema OFICINA-HELP
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Verifica se a API key está configurada
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERRO: GOOGLE_API_KEY não encontrada!")
    print("   Configure a variável no arquivo .env")
    sys.exit(1)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# Configurações
DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/embedding-001"

# Marcas suportadas
MARCAS = ["jeep", "audi", "porsche"]


def criar_diretorio_se_nao_existe(diretorio: Path) -> None:
    """Cria o diretório se não existir."""
    diretorio.mkdir(parents=True, exist_ok=True)


def carregar_pdfs_da_pasta(pasta: Path) -> list:
    """
    Carrega todos os PDFs de uma pasta.
    
    Args:
        pasta: Caminho da pasta contendo os PDFs
        
    Returns:
        Lista de documentos carregados
    """
    documentos = []
    arquivos_pdf = list(pasta.glob("*.pdf"))
    
    if not arquivos_pdf:
        print(f"   ⚠️  Nenhum PDF encontrado em {pasta}")
        return documentos
    
    print(f"   📄 Encontrados {len(arquivos_pdf)} PDFs")
    
    for arquivo_pdf in arquivos_pdf:
        try:
            print(f"      └── Processando: {arquivo_pdf.name}")
            loader = PyPDFLoader(str(arquivo_pdf))
            docs = loader.load()
            
            # Adiciona metadados úteis
            for doc in docs:
                doc.metadata["source_file"] = arquivo_pdf.name
                doc.metadata["marca"] = pasta.name
            
            documentos.extend(docs)
            print(f"          ✓ {len(docs)} páginas extraídas")
            
        except Exception as e:
            print(f"          ❌ Erro ao processar {arquivo_pdf.name}: {e}")
    
    return documentos


def dividir_documentos(documentos: list) -> list:
    """
    Divide os documentos em chunks menores.
    
    Args:
        documentos: Lista de documentos a serem divididos
        
    Returns:
        Lista de chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documentos)
    return chunks


def criar_indice_faiss(chunks: list, nome_indice: str) -> None:
    """
    Cria e salva um índice FAISS a partir dos chunks.
    
    Args:
        chunks: Lista de chunks de documentos
        nome_indice: Nome do índice (ex: jeep_index)
    """
    if not chunks:
        print(f"   ⚠️  Nenhum chunk para criar índice {nome_indice}")
        return
    
    print(f"   🔄 Criando embeddings para {len(chunks)} chunks...")
    
    # Inicializa o modelo de embeddings do Google
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Cria o índice FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Salva o índice
    caminho_indice = VECTORSTORE_DIR / nome_indice
    criar_diretorio_se_nao_existe(caminho_indice.parent)
    vectorstore.save_local(str(caminho_indice))
    
    print(f"   ✅ Índice salvo em: {caminho_indice}")


def processar_marca(marca: str) -> None:
    """
    Processa todos os PDFs de uma marca específica.
    
    Args:
        marca: Nome da marca (jeep, audi, porsche)
    """
    pasta_marca = DATA_DIR / marca
    
    print(f"\n{'='*50}")
    print(f"🔧 Processando: {marca.upper()}")
    print(f"{'='*50}")
    
    if not pasta_marca.exists():
        print(f"   📁 Criando pasta: {pasta_marca}")
        criar_diretorio_se_nao_existe(pasta_marca)
        print(f"   ⚠️  Pasta vazia. Adicione PDFs e execute novamente.")
        return
    
    # Carrega os PDFs
    print(f"   📂 Pasta: {pasta_marca}")
    documentos = carregar_pdfs_da_pasta(pasta_marca)
    
    if not documentos:
        print(f"   ⚠️  Nenhum documento carregado para {marca}")
        return
    
    # Divide em chunks
    print(f"   ✂️  Dividindo documentos em chunks...")
    chunks = dividir_documentos(documentos)
    print(f"   📊 Total de chunks: {len(chunks)}")
    
    # Cria o índice FAISS
    nome_indice = f"{marca}_index"
    criar_indice_faiss(chunks, nome_indice)


def main():
    """Função principal de execução."""
    print("\n" + "="*60)
    print("🚗 OFICINA-HELP - Sistema de Ingestão de Manuais")
    print("="*60)
    print(f"\n📁 Diretório de dados: {DATA_DIR.absolute()}")
    print(f"💾 Diretório de índices: {VECTORSTORE_DIR.absolute()}")
    print(f"📏 Tamanho do chunk: {CHUNK_SIZE} caracteres")
    print(f"🔗 Sobreposição: {CHUNK_OVERLAP} caracteres")
    
    # Cria diretórios base se não existirem
    criar_diretorio_se_nao_existe(DATA_DIR)
    criar_diretorio_se_nao_existe(VECTORSTORE_DIR)
    
    # Processa cada marca
    for marca in MARCAS:
        processar_marca(marca)
    
    print("\n" + "="*60)
    print("✅ Processo de ingestão concluído!")
    print("="*60)
    print("\n📌 Próximos passos:")
    print("   1. Adicione os PDFs nas pastas correspondentes em data/")
    print("   2. Execute este script novamente: python ingest.py")
    print("   3. Inicie a aplicação: streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
