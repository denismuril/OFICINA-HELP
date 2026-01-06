"""
OFICINA-HELP - Aplicação Principal
====================================
Sistema RAG para consulta de manuais técnicos de veículos.
Interface web com Streamlit, LLM Google Gemini e FAISS para busca vetorial.

Uso:
    streamlit run app.py

Autor: Sistema OFICINA-HELP
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

VECTORSTORE_DIR = Path("vectorstore")
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"
LLM_TEMPERATURE = 0  # Máxima precisão

# Mapeamento de marcas para índices
MARCAS_CONFIG = {
    "Jeep": "jeep_index",
    "Audi": "audi_index",
    "Porsche": "porsche_index"
}

# Prompt do Sistema
SYSTEM_PROMPT_TEMPLATE = """Você é um assistente técnico especializado em manuais de veículos da marca {marca}.
Sua função é auxiliar mecânicos respondendo perguntas técnicas com base EXCLUSIVAMENTE no conteúdo dos manuais fornecidos.

REGRAS IMPORTANTES:
1. Responda APENAS com base no contexto fornecido abaixo.
2. Se a informação não estiver no contexto, responda: "Não consta no manual."
3. Seja preciso e técnico nas respostas.
4. Ao final de cada resposta, SEMPRE cite a fonte no formato: [Fonte: nome_do_arquivo, página X]
5. Se houver múltiplas fontes, liste todas.

CONTEXTO DOS MANUAIS:
{context}

PERGUNTA DO MECÂNICO:
{question}

RESPOSTA TÉCNICA:"""


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def verificar_api_key() -> bool:
    """Verifica se a API key está configurada."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ **GOOGLE_API_KEY não configurada!**")
        st.info("""
        **Como configurar:**
        1. Crie um arquivo `.env` na raiz do projeto
        2. Adicione: `GOOGLE_API_KEY=sua_chave_aqui`
        3. Reinicie a aplicação
        
        Obtenha sua chave em: [Google AI Studio](https://makersuite.google.com/app/apikey)
        """)
        return False
    return True


def verificar_indice_existe(marca: str) -> bool:
    """Verifica se o índice FAISS da marca existe."""
    nome_indice = MARCAS_CONFIG.get(marca)
    if not nome_indice:
        return False
    
    caminho_indice = VECTORSTORE_DIR / nome_indice
    return caminho_indice.exists() and (caminho_indice / "index.faiss").exists()


@st.cache_resource
def carregar_embeddings():
    """Carrega o modelo de embeddings (cache para performance)."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


@st.cache_resource
def carregar_llm():
    """Carrega o modelo LLM (cache para performance)."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def carregar_vectorstore(marca: str):
    """Carrega o índice FAISS da marca selecionada."""
    nome_indice = MARCAS_CONFIG.get(marca)
    caminho_indice = VECTORSTORE_DIR / nome_indice
    
    embeddings = carregar_embeddings()
    vectorstore = FAISS.load_local(
        str(caminho_indice),
        embeddings,
        allow_dangerous_deserialization=True  # Necessário para carregar índices salvos
    )
    return vectorstore


def criar_chain_qa(vectorstore, marca: str):
    """Cria a chain de Question-Answering."""
    llm = carregar_llm()
    
    # Configura o prompt
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
        partial_variables={"marca": marca}
    )
    
    # Cria a chain de QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def formatar_fonte(doc) -> str:
    """Formata a fonte do documento para exibição."""
    metadata = doc.metadata
    arquivo = metadata.get("source_file", metadata.get("source", "Desconhecido"))
    pagina = metadata.get("page", "N/A")
    
    # Se o source contém o caminho completo, pega só o nome do arquivo
    if "/" in str(arquivo) or "\\" in str(arquivo):
        arquivo = Path(arquivo).name
    
    return f"📄 **{arquivo}** | Página: {pagina + 1 if isinstance(pagina, int) else pagina}"


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    """Função principal da aplicação."""
    
    # Configuração da página
    st.set_page_config(
        page_title="OFICINA-HELP",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS customizado
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #1e3a5f, #2d5a87);
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: white;
            margin: 0;
        }
        .main-header p {
            color: #b0c4de;
            margin: 0.5rem 0 0 0;
        }
        .stExpander {
            background-color: #f0f2f6;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔧 OFICINA-HELP</h1>
        <p>Sistema de Consulta a Manuais Técnicos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificação da API Key
    if not verificar_api_key():
        return
    
    # ========================================================================
    # SIDEBAR - Seleção de Marca
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        st.markdown("---")
        
        # Seleção de marca (obrigatória)
        st.subheader("🚗 Selecione a Marca")
        
        marca_selecionada = st.selectbox(
            "Escolha a montadora:",
            options=["-- Selecione --"] + list(MARCAS_CONFIG.keys()),
            index=0,
            help="Selecione a marca do veículo para consultar o manual correspondente."
        )
        
        # Verifica se a marca foi selecionada
        marca_valida = marca_selecionada != "-- Selecione --"
        
        if marca_valida:
            # Verifica se o índice existe
            if verificar_indice_existe(marca_selecionada):
                st.success(f"✅ Índice **{marca_selecionada}** carregado!")
            else:
                st.error(f"❌ Índice **{marca_selecionada}** não encontrado!")
                st.warning("""
                **Para criar o índice:**
                1. Adicione os PDFs em `data/{marca}/`
                2. Execute: `python ingest.py`
                """.format(marca=marca_selecionada.lower()))
                marca_valida = False
        else:
            st.warning("⚠️ Selecione uma marca para continuar")
        
        st.markdown("---")
        
        # Informações
        st.subheader("ℹ️ Sobre")
        st.markdown("""
        Este sistema utiliza **IA Generativa** para responder perguntas
        técnicas com base nos manuais oficiais dos veículos.
        
        **Tecnologias:**
        - 🤖 Google Gemini 1.5 Flash
        - 🔍 FAISS Vector Search
        - 🔗 LangChain Framework
        """)
    
    # ========================================================================
    # ÁREA PRINCIPAL - Chat
    # ========================================================================
    
    # Área de pergunta
    st.subheader("💬 Faça sua pergunta técnica")
    
    if not marca_valida:
        st.info("👈 **Por favor, selecione uma marca na barra lateral para começar.**")
        return
    
    # Campo de pergunta
    pergunta = st.text_area(
        f"Digite sua dúvida sobre veículos {marca_selecionada}:",
        height=100,
        placeholder=f"Ex: Qual é o torque de aperto das rodas do {marca_selecionada}?",
        key="pergunta_input"
    )
    
    # Botão de envio
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        btn_enviar = st.button("🔍 Buscar Resposta", type="primary", use_container_width=True)
    with col2:
        btn_limpar = st.button("🗑️ Limpar", use_container_width=True)
    
    if btn_limpar:
        st.rerun()
    
    # Processa a pergunta
    if btn_enviar and pergunta.strip():
        with st.spinner("🔄 Consultando manuais..."):
            try:
                # Carrega o vectorstore
                vectorstore = carregar_vectorstore(marca_selecionada)
                
                # Cria a chain de QA
                qa_chain = criar_chain_qa(vectorstore, marca_selecionada)
                
                # Executa a consulta
                resultado = qa_chain.invoke({"query": pergunta})
                
                # Exibe a resposta
                st.markdown("---")
                st.subheader("📝 Resposta")
                st.markdown(resultado["result"])
                
                # Exibe as fontes em um expansor
                if resultado.get("source_documents"):
                    with st.expander("📚 Ver Fontes", expanded=False):
                        for i, doc in enumerate(resultado["source_documents"], 1):
                            st.markdown(f"**Trecho {i}:**")
                            st.markdown(formatar_fonte(doc))
                            st.text_area(
                                label="",
                                value=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                                height=100,
                                disabled=True,
                                key=f"fonte_{i}"
                            )
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Erro ao processar a pergunta: {str(e)}")
                st.info("Verifique se o índice FAISS foi criado corretamente.")
    
    elif btn_enviar and not pergunta.strip():
        st.warning("⚠️ Por favor, digite uma pergunta.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            🔧 OFICINA-HELP v1.0 | Sistema de Consulta a Manuais Técnicos | 
            Powered by Google Gemini & LangChain
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
