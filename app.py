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
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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
SYSTEM_PROMPT = """Você é um assistente técnico especializado em manuais de veículos da marca {marca}.
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
        allow_dangerous_deserialization=True
    )
    return vectorstore


def format_docs(docs):
    """Formata os documentos para o contexto."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", doc.metadata.get("source", "Desconhecido"))
        page = doc.metadata.get("page", "N/A")
        if "/" in str(source) or "\\" in str(source):
            source = Path(source).name
        formatted.append(f"[{source}, Página {page + 1 if isinstance(page, int) else page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def formatar_fonte(doc) -> str:
    """Formata a fonte do documento para exibição."""
    metadata = doc.metadata
    arquivo = metadata.get("source_file", metadata.get("source", "Desconhecido"))
    pagina = metadata.get("page", "N/A")
    
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
            help="Selecione a marca do veículo para consultar o manual."
        )
        
        # Verifica se a marca foi selecionada
        marca_valida = marca_selecionada != "-- Selecione --"
        
        if marca_valida:
            # Verifica se o índice existe
            if verificar_indice_existe(marca_selecionada):
                st.success(f"✅ Índice **{marca_selecionada}** carregado!")
            else:
                st.error(f"❌ Índice **{marca_selecionada}** não encontrado!")
                st.warning(f"""
                **Para criar o índice:**
                1. Adicione os PDFs em `data/{marca_selecionada.lower()}/`
                2. Execute: `python ingest.py`
                """)
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
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                # Busca documentos relevantes
                docs = retriever.invoke(pergunta)
                
                # Cria o contexto
                context = format_docs(docs)
                
                # Cria o prompt
                prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
                
                # Cria a chain
                llm = carregar_llm()
                chain = prompt | llm | StrOutputParser()
                
                # Executa
                resposta = chain.invoke({
                    "marca": marca_selecionada,
                    "context": context,
                    "question": pergunta
                })
                
                # Exibe a resposta
                st.markdown("---")
                st.subheader("📝 Resposta")
                st.markdown(resposta)
                
                # Exibe as fontes em um expansor
                if docs:
                    with st.expander("📚 Ver Fontes", expanded=False):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Trecho {i}:**")
                            st.markdown(formatar_fonte(doc))
                            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            st.text_area(
                                label="",
                                value=content,
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
