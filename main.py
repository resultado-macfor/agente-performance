# app_completo.py - App com TODAS as colunas do BigQuery + Análise de Performance
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import io

# TENTAR importar Gemini apenas se a chave existir
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False
    st.warning("Biblioteca google.generativeai não disponível")

# Configuração da página
st.set_page_config(
    layout="wide",
    page_title="Analytics Platform",
    page_icon="📊"
)

# CSS personalizado
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton button {
        background-color: #4f46e5 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 12px;
        margin: 5px;
        text-align: center;
    }
    .okr-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid #4f46e5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #4f46e5 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #4f46e5 !important;
    }
    .column-info {
        background: #f0f9ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #0ea5e9;
    }
    .data-card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .campaign-analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 6px 10px rgba(0,0,0,0.1);
    }
    .performance-metric {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 3px 5px rgba(0,0,0,0.05);
    }
    .insight-card {
        background: #f0f9ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #0ea5e9;
    }
    .recommendation-card {
        background: #f0f9ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.title("📊 Analytics Platform - TODAS as Colunas")

# =============================================================================
# CONFIGURAÇÃO GEMINI PARA ANÁLISE DE CAMPANHA
# =============================================================================

# Inicializar modelo Gemini como None
modelo_texto = None

# Primeiro tentar obter a chave GEM_API_KEY
gemini_api_key = os.getenv("GEM_API_KEY")

# Se não encontrar, tentar outras chaves possíveis
if not gemini_api_key:
    # Tentar outras chaves comuns
    possible_keys = ["GEN_API_KEY", "GEN_API_KEY2", "GEMINI_API_KEY", "GOOGLE_API_KEY"]
    for key_name in possible_keys:
        key_value = os.getenv(key_name)
        if key_value:
            gemini_api_key = key_value
            st.sidebar.info(f"Usando chave de {key_name} para Gemini")
            break

# Se ainda não encontrou, verificar secrets do Streamlit
if not gemini_api_key and hasattr(st, 'secrets'):
    # Tentar várias chaves possíveis nos secrets
    secrets_keys = ["GEM_API_KEY", "GEN_API_KEY", "GEN_API_KEY2", "GEMINI_API_KEY", "GOOGLE_API_KEY"]
    for key_name in secrets_keys:
        if key_name in st.secrets:
            gemini_api_key = st.secrets[key_name]
            st.sidebar.info(f"Usando {key_name} dos secrets para Gemini")
            break

# Configurar Gemini se tiver chave e biblioteca disponível
if gemini_api_key and GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=gemini_api_key)
        modelo_texto = genai.GenerativeModel("gemini-1.5-flash")
        st.sidebar.success("✅ Gemini configurado com sucesso!")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro ao configurar Gemini: {str(e)[:50]}...")
        modelo_texto = None
elif gemini_api_key and not GEMINI_AVAILABLE:
    st.sidebar.warning("⚠️ Chave Gemini encontrada mas biblioteca não disponível")
else:
    st.sidebar.info("ℹ️ Gemini não configurado. A funcionalidade de análise com IA estará limitada.")

# =============================================================================
# CONEXÃO E CARREGAMENTO - TODAS AS COLUNAS (COM VARIÁVEIS DE AMBIENTE)
# =============================================================================

@st.cache_resource
def get_bigquery_client():
    """Cria cliente BigQuery usando variáveis de ambiente"""
    try:
        # OPÇÃO 1: Streamlit Secrets
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            service_account_info = dict(st.secrets["gcp_service_account"])
            if isinstance(service_account_info.get("private_key"), str):
                service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
        
        # OPÇÃO 2: Variáveis de ambiente individuais
        elif all(key in os.environ for key in ['type', 'project_id', 'private_key', 'client_email', 'token_uri']):
            service_account_info = {
                "type": os.environ['type'],
                "project_id": os.environ['project_id'],
                "private_key_id": os.environ.get('private_key_id', ''),
                "private_key": os.environ['private_key'].replace('\\n', '\n'),
                "client_email": os.environ['client_email'],
                "client_id": os.environ.get('client_id', ''),
                "auth_uri": os.environ.get('auth_uri', 'https://accounts.google.com/o/oauth2/auth'),
                "token_uri": os.environ['token_uri'],
                "auth_provider_x509_cert_url": os.environ.get('auth_provider_x509_cert_url', 'https://www.googleapis.com/oauth2/v1/certs'),
                "client_x509_cert_url": os.environ.get('client_x509_cert_url', ''),
                "universe_domain": os.environ.get('universe_domain', 'googleapis.com')
            }
        
        # OPÇÃO 3: JSON string completo em variável de ambiente
        elif 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
            credentials_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
            service_account_info = json.loads(credentials_json)
        
        else:
            st.error("""
            ❌ Credenciais não encontradas!
            
            Configure uma das seguintes opções:
            
            1. **Streamlit Secrets** (no formato TOML):
               ```toml
               [gcp_service_account]
               type = "service_account"
               project_id = "seu-project"
               private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----"
               client_email = "email@project.iam.gserviceaccount.com"
               token_uri = "https://oauth2.googleapis.com/token"
               ```
            
            2. **Variáveis de ambiente individuais**:
               - `type`
               - `project_id`
               - `private_key`
               - `client_email`
               - `token_uri`
            
            3. **JSON completo em variável de ambiente**:
               - `GOOGLE_APPLICATION_CREDENTIALS_JSON`
            """)
            return None
        
        # Criar credenciais
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        client = bigquery.Client(
            credentials=credentials,
            project=service_account_info["project_id"]
        )
        
        return client
    
    except Exception as e:
        st.error(f"❌ Erro na conexão com BigQuery: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_all_columns_data(_client, data_inicio=None, data_fim=None, data_sources=None, limit=50000):
    """Carrega TODAS as colunas disponíveis na tabela"""
    try:
        st.info("🔍 Carregando dados... Isso pode levar alguns instantes")
        
        # Construir query básica
        query = """
        SELECT *
        FROM `macfor-media-flow.ads.app_view_campaigns`
        """
        
        conditions = []
        
        if data_inicio:
            conditions.append(f"DATE(date) >= DATE('{data_inicio}')")
        if data_fim:
            conditions.append(f"DATE(date) <= DATE('{data_fim}')")
        if data_sources and len(data_sources) > 0:
            ds_str = ", ".join([f"'{ds}'" for ds in data_sources])
            conditions.append(f"datasource IN ({ds_str})")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY date DESC LIMIT {limit}"
        
        df = _client.query(query).to_dataframe()
        
        if df.empty:
            st.warning("Nenhum dado encontrado")
            return pd.DataFrame()
        
        st.success(f"✅ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        
        return df
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def safe_metric(label, value, delta=None, delta_color="normal"):
    """Função segura para criar métricas"""
    try:
        # Se for NaN, mostrar como 0
        if pd.isna(value):
            value = 0
        
        # Se for número, formatar
        if isinstance(value, (int, np.integer)):
            display_value = f"{int(value):,}"
        elif isinstance(value, (float, np.floating)):
            # Para valores decimais, formatar baseado no tamanho
            if abs(value) < 0.01:
                display_value = f"{value:.4f}"
            elif abs(value) < 1:
                display_value = f"{value:.3f}"
            elif abs(value) < 1000:
                display_value = f"{value:.2f}"
            else:
                display_value = f"{value:,.0f}"
        else:
            # Tentar converter para número
            try:
                num_val = float(value)
                display_value = f"{num_val:,.2f}"
            except:
                display_value = str(value)
        
        # Criar métrica
        if delta is not None:
            if pd.isna(delta):
                delta = None
            elif isinstance(delta, (int, float, np.integer, np.floating)):
                delta = f"{delta:+.2f}"
            else:
                try:
                    delta_val = float(delta)
                    delta = f"{delta_val:+.2f}"
                except:
                    delta = str(delta)
        
        return st.metric(label, display_value, delta=delta, delta_color=delta_color)
    
    except Exception as e:
        # Fallback seguro
        return st.metric(label, f"Erro: {str(e)[:20]}")

def identificar_colunas_numericas(df):
    """Identifica automaticamente colunas numéricas"""
    colunas_numericas = []
    
    for col in df.columns:
        try:
            # Verificar se é numérico
            if pd.api.types.is_numeric_dtype(df[col]):
                colunas_numericas.append(col)
            # Tentar converter para ver se é numérico
            else:
                # Testar com amostra
                amostra = df[col].dropna().head(10)
                if len(amostra) > 0:
                    # Tentar converter para numérico
                    try:
                        pd.to_numeric(amostra)
                        colunas_numericas.append(col)
                    except:
                        pass
        except:
            continue
    
    return colunas_numericas

def analisar_coluna(df, coluna):
    """Analisa uma coluna específica - VERSÃO CORRIGIDA"""
    if coluna not in df.columns:
        return None
    
    try:
        dados_coluna = df[coluna]
        
        # GARANTIR VALORES NUMÉRICOS
        total = int(len(dados_coluna))
        nao_nulos = int(dados_coluna.notna().sum())
        nulos = int(dados_coluna.isna().sum())
        valores_unicos = int(dados_coluna.nunique())
        
        # Calcular percentual com tratamento de divisão por zero
        if total > 0:
            percentual_nulos = float((nulos / total) * 100)
        else:
            percentual_nulos = 0.0
        
        analise = {
            'nome': coluna,
            'tipo': str(dados_coluna.dtype),
            'total': total,
            'nao_nulos': nao_nulos,
            'nulos': nulos,
            'percentual_nulos': percentual_nulos,
            'valores_unicos': valores_unicos
        }
        
        # Se for numérica
        if pd.api.types.is_numeric_dtype(dados_coluna):
            dados_validos = dados_coluna.dropna()
            if len(dados_validos) > 0:
                analise.update({
                    'tipo_detalhado': 'Numérica',
                    'min': float(dados_validos.min()),
                    'max': float(dados_validos.max()),
                    'media': float(dados_validos.mean()),
                    'mediana': float(dados_validos.median()),
                    'desvio_padrao': float(dados_validos.std()),
                    'q1': float(dados_validos.quantile(0.25)),
                    'q3': float(dados_validos.quantile(0.75))
                })
            else:
                analise.update({
                    'tipo_detalhado': 'Numérica (vazia)',
                    'min': 0.0,
                    'max': 0.0,
                    'media': 0.0,
                    'mediana': 0.0,
                    'desvio_padrao': 0.0,
                    'q1': 0.0,
                    'q3': 0.0
                })
        # Se for categórica/texto
        elif dados_coluna.dtype == 'object':
            value_counts = dados_coluna.value_counts()
            analise.update({
                'tipo_detalhado': 'Texto/Categórica',
                'valores_mais_comuns': value_counts.head(10).to_dict(),
                'valor_mais_frequente': value_counts.index[0] if len(value_counts) > 0 else None,
                'frequencia_valor_mais_comum': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            })
        # Se for data
        elif pd.api.types.is_datetime64_any_dtype(dados_coluna):
            dados_validos = dados_coluna.dropna()
            if len(dados_validos) > 0:
                analise.update({
                    'tipo_detalhado': 'Data',
                    'data_minima': dados_validos.min(),
                    'data_maxima': dados_validos.max(),
                    'intervalo_dias': int((dados_validos.max() - dados_validos.min()).days)
                })
            else:
                analise.update({
                    'tipo_detalhado': 'Data (vazia)',
                    'data_minima': None,
                    'data_maxima': None,
                    'intervalo_dias': 0
                })
        else:
            analise.update({'tipo_detalhado': 'Outro'})
            
        return analise
        
    except Exception as e:
        # Retorna valores numéricos seguros mesmo em caso de erro
        return {
            'nome': coluna,
            'tipo': 'Erro',
            'tipo_detalhado': f'Erro na análise',
            'total': 0,
            'nao_nulos': 0,
            'nulos': 0,
            'percentual_nulos': 0.0,
            'valores_unicos': 0
        }

def criar_visualizacao_coluna(df, coluna):
    """Cria visualização adequada para o tipo de coluna"""
    if coluna not in df.columns:
        return None
    
    try:
        dados = df[coluna].dropna()
        
        if len(dados) == 0:
            return None
        
        # Coluna numérica
        if pd.api.types.is_numeric_dtype(df[coluna]):
            fig = px.histogram(
                df, 
                x=coluna,
                nbins=min(50, len(dados)),
                title=f"Distribuição de {coluna}",
                marginal="box"
            )
            return fig
        
        # Coluna categórica/texto (até 50 categorias)
        elif df[coluna].nunique() <= 50:
            contagem = df[coluna].value_counts().head(20)
            fig = px.bar(
                x=contagem.index,
                y=contagem.values,
                title=f"Top 20 Valores em {coluna}",
                labels={'x': coluna, 'y': 'Contagem'}
            )
            fig.update_xaxes(tickangle=45)
            return fig
        
        # Coluna data
        elif pd.api.types.is_datetime64_any_dtype(df[coluna]):
            try:
                contagem_diaria = df.groupby(df[coluna].dt.date).size().reset_index()
                contagem_diaria.columns = ['data', 'contagem']
                
                fig = px.line(
                    contagem_diaria,
                    x='data',
                    y='contagem',
                    title=f"Frequência por Data - {coluna}"
                )
                return fig
            except:
                return None
        
        return None
    except Exception as e:
        st.error(f"Erro ao criar visualização para {coluna}: {str(e)[:100]}")
        return None

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================

# Inicializar estado
if 'df_completo' not in st.session_state:
    st.session_state.df_completo = pd.DataFrame()
if 'colunas_numericas' not in st.session_state:
    st.session_state.colunas_numericas = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Testar conexão
    st.subheader("🔗 Conexão")
    if st.button("Testar Conexão BigQuery"):
        with st.spinner("Conectando..."):
            client = get_bigquery_client()
            if client:
                st.success("✅ Conexão bem-sucedida!")
            else:
                st.error("❌ Falha na conexão. Verifique as credenciais.")
    
    # Data sources
    data_sources_opcoes = ["facebook", "google ads", "tiktok"]
    selected_sources = st.multiselect(
        "Data Sources",
        options=data_sources_opcoes,
        default=data_sources_opcoes
    )
    
    # Período
    st.subheader("📅 Período")
    periodo = st.radio(
        "Selecione",
        ["Últimos 30 dias", "Últimos 90 dias", "Últimos 180 dias", "Todo período", "Personalizado"],
        index=1
    )
    
    data_fim = datetime.now().date()
    
    if periodo == "Últimos 30 dias":
        data_inicio = data_fim - timedelta(days=30)
    elif periodo == "Últimos 90 dias":
        data_inicio = data_fim - timedelta(days=90)
    elif periodo == "Últimos 180 dias":
        data_inicio = data_fim - timedelta(days=180)
    elif periodo == "Todo período":
        data_inicio = None
        data_fim = None
    else:
        col1, col2 = st.columns(2)
        with col1:
            data_inicio = st.date_input("Início", value=data_fim - timedelta(days=90))
        with col2:
            data_fim = st.date_input("Fim", value=data_fim)
    
    # Limite
    limite = st.slider("Limite de registros", 1000, 100000, 20000, 1000)
    
    # Botão carregar
    if st.button("📊 Carregar Dados Gerais", use_container_width=True, type="primary"):
        with st.spinner("Carregando TODAS as colunas... Isso pode demorar"):
            client = get_bigquery_client()
            if client:
                df = load_all_columns_data(
                    client,
                    data_inicio=data_inicio,
                    data_fim=data_fim,
                    data_sources=selected_sources,
                    limit=limite
                )
                
                if not df.empty:
                    st.session_state.df_completo = df
                    # Identificar colunas numéricas
                    st.session_state.colunas_numericas = identificar_colunas_numericas(df)
                    st.success(f"✅ {len(df):,} registros carregados")
                    st.success(f"📊 {len(st.session_state.colunas_numericas)} colunas numéricas identificadas")
                else:
                    st.error("Nenhum dado encontrado")
            else:
                st.error("❌ Não foi possível conectar ao BigQuery.")

# Verificar se há dados carregados
df = st.session_state.df_completo
colunas_numericas = st.session_state.colunas_numericas

if df.empty:
    st.warning("📭 Nenhum dado carregado. Use o botão na sidebar para carregar dados.")
    st.stop()

# Abas principais
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Visão Geral", 
    "📈 Análise Numérica", 
    "🔍 Explorar Colunas", 
    "📊 Visualizar Dados",
    "📐 Análise Estatística",
    "🎯 Performance de Campanhas"
])

# =============================================================================
# TAB 1: VISÃO GERAL DAS COLUNAS
# =============================================================================

with tab1:
    st.header("📋 Visão Geral de TODAS as Colunas")
    
    # Estatísticas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        safe_metric("Total de Colunas", len(df.columns))
    
    with col2:
        safe_metric("Colunas Numéricas", len(colunas_numericas))
    
    with col3:
        safe_metric("Total de Registros", len(df))
    
    with col4:
        memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
        safe_metric("Uso de Memória", memoria_mb, delta=None)
    
    # Lista de todas as colunas com informações
    st.subheader("📊 Detalhes de Cada Coluna")
    
    # Filtrar colunas
    col_filtro1, col_filtro2 = st.columns(2)
    
    with col_filtro1:
        tipo_filtro = st.selectbox(
            "Filtrar por tipo",
            ["Todas", "Numéricas", "Texto", "Datas"],
            key="filtro_tipo_tab1"
        )
    
    with col_filtro2:
        pesquisa_coluna = st.text_input("🔍 Pesquisar coluna", "", key="pesquisa_coluna_tab1")
    
    # Preparar lista de colunas filtradas
    colunas_para_mostrar = []
    
    for col in df.columns:
        incluir = True
        
        # Filtrar por tipo
        if tipo_filtro == "Numéricas":
            incluir = col in colunas_numericas
        elif tipo_filtro == "Texto":
            incluir = df[col].dtype == 'object' and col not in colunas_numericas
        elif tipo_filtro == "Datas":
            incluir = pd.api.types.is_datetime64_any_dtype(df[col])
        
        # Filtrar por pesquisa
        if pesquisa_coluna and pesquisa_coluna.lower() not in col.lower():
            incluir = False
        
        if incluir:
            colunas_para_mostrar.append(col)
    
    # Mostrar informações de cada coluna
    for col in sorted(colunas_para_mostrar)[:50]:
        analise = analisar_coluna(df, col)
        
        with st.expander(f"**{col}** ({analise['tipo_detalhado'] if 'tipo_detalhado' in analise else analise['tipo']})"):
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                safe_metric("Tipo", analise['tipo'])
                safe_metric("Não nulos", analise['nao_nulos'])
                safe_metric("Valores únicos", analise['valores_unicos'])
            
            with col_info2:
                safe_metric("Nulos", analise['nulos'])
                safe_metric("% Nulos", analise['percentual_nulos'])
            
            if analise['tipo_detalhado'] == 'Numérica' and analise['nao_nulos'] > 0:
                st.subheader("📈 Estatísticas Numéricas")
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    safe_metric("Média", analise.get('media', 0))
                    safe_metric("Min", analise.get('min', 0))
                
                with col_stats2:
                    safe_metric("Mediana", analise.get('mediana', 0))
                    safe_metric("Max", analise.get('max', 0))
                
                with col_stats3:
                    safe_metric("Q1 (25%)", analise.get('q1', 0))
                    safe_metric("Desvio Padrão", analise.get('desvio_padrao', 0))
                
                with col_stats4:
                    safe_metric("Q3 (75%)", analise.get('q3', 0))
            
            # Botão para visualizar
            if st.button(f"📊 Visualizar {col}", key=f"viz_{col}_tab1"):
                st.session_state.coluna_selecionada = col
                st.rerun()

# =============================================================================
# TAB 2: ANÁLISE NUMÉRICA
# =============================================================================

with tab2:
    st.header("📈 Análise de Colunas Numéricas")
    
    if not colunas_numericas:
        st.warning("Nenhuma coluna numérica identificada")
    else:
        st.success(f"✅ {len(colunas_numericas)} colunas numéricas disponíveis para análise")
        
        # Selecionar colunas para análise
        colunas_selecionadas = st.multiselect(
            "Selecione colunas numéricas para análise",
            options=colunas_numericas,
            default=colunas_numericas[:min(5, len(colunas_numericas))],
            key="colunas_selecionadas_tab2"
        )
        
        if colunas_selecionadas:
            # Estatísticas descritivas
            st.subheader("📊 Estatísticas Descritivas")
            
            stats_df = df[colunas_selecionadas].describe().T
            stats_df['missing'] = df[colunas_selecionadas].isna().sum()
            stats_df['missing_pct'] = (df[colunas_selecionadas].isna().sum() / len(df) * 100)
            
            # Formatar DataFrame para exibição
            def formatar_numero(x):
                if isinstance(x, (int, np.integer)):
                    return f"{x:,}"
                elif isinstance(x, (float, np.floating)):
                    if pd.isna(x):
                        return "N/A"
                    elif abs(x) < 0.01:
                        return f"{x:.4f}"
                    elif abs(x) < 1:
                        return f"{x:.3f}"
                    elif abs(x) < 1000:
                        return f"{x:.2f}"
                    else:
                        return f"{x:,.0f}"
                return str(x)
            
            st.dataframe(
                stats_df.style.format(formatar_numero),
                use_container_width=True
            )
            
            # Histogramas
            if len(colunas_selecionadas) > 0:
                st.subheader("📈 Distribuições")
                
                num_cols = min(3, len(colunas_selecionadas))
                cols_vis = st.columns(num_cols)
                
                for idx, col in enumerate(colunas_selecionadas[:num_cols*3]):
                    with cols_vis[idx % num_cols]:
                        fig = criar_visualizacao_coluna(df, col)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_tab2")
            
            # Correlações
            if len(colunas_selecionadas) >= 2:
                st.subheader("🔥 Matriz de Correlação")
                
                try:
                    # Calcular correlações apenas para colunas numéricas
                    df_numeric = df[colunas_selecionadas].apply(pd.to_numeric, errors='coerce')
                    correlacao = df_numeric.corr()
                    
                    fig_corr = px.imshow(
                        correlacao,
                        text_auto='.2f',
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlações entre Variáveis Numéricas"
                    )
                    fig_corr.update_layout(height=600)
                    st.plotly_chart(fig_corr, use_container_width=True, key="corr_matrix_tab2")
                    
                    # Top correlações
                    st.subheader("🔗 Principais Correlações")
                    
                    correlacoes_fortes = []
                    for i in range(len(correlacao.columns)):
                        for j in range(i+1, len(correlacao.columns)):
                            corr = correlacao.iloc[i, j]
                            if not pd.isna(corr) and abs(corr) > 0.3:
                                correlacoes_fortes.append({
                                    'Variável 1': correlacao.columns[i],
                                    'Variável 2': correlacao.columns[j],
                                    'Correlação': corr,
                                    'Força': 'Forte' if abs(corr) > 0.7 else 'Moderada'
                                })
                    
                    if correlacoes_fortes:
                        correlacoes_fortes.sort(key=lambda x: abs(x['Correlação']), reverse=True)
                        df_corr = pd.DataFrame(correlacoes_fortes[:20])
                        st.dataframe(df_corr, use_container_width=True)
                    else:
                        st.info("Não foram encontradas correlações fortes (> 0.3)")
                        
                except Exception as e:
                    st.error(f"Erro ao calcular correlações: {str(e)[:100]}")

# =============================================================================
# TAB 3: EXPLORAR COLUNAS
# =============================================================================

with tab3:
    st.header("🔍 Explorar Colunas Individualmente")
    
    coluna_selecionada = st.selectbox(
        "Selecione uma coluna para explorar",
        options=sorted(df.columns),
        index=0,
        key="coluna_selecionada_tab3"
    )
    
    if coluna_selecionada:
        analise = analisar_coluna(df, coluna_selecionada)
        
        if analise is not None:
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                safe_metric("Total de Valores", analise['total'])
                safe_metric("Valores Não Nulos", analise['nao_nulos'])
                safe_metric("Valores Únicos", analise['valores_unicos'])
            
            with col_info2:
                safe_metric("Valores Nulos", analise['nulos'])
                safe_metric("% Nulos", analise['percentual_nulos'])
            
            # Visualização
            st.subheader("📊 Visualização")
            fig = criar_visualizacao_coluna(df, coluna_selecionada)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"viz_{coluna_selecionada}_tab3")
            
            # Valores
            st.subheader("📋 Amostra de Valores")
            
            col_amostra1, col_amostra2 = st.columns(2)
            
            with col_amostra1:
                st.write("**Primeiros 10 valores:**")
                st.write(df[coluna_selecionada].head(10).tolist())
            
            with col_amostra2:
                st.write("**Últimos 10 valores:**")
                st.write(df[coluna_selecionada].tail(10).tolist())
            
            # Se for categórica, mostrar distribuição
            if analise['tipo_detalhado'] == 'Texto/Categórica' and analise['valores_unicos'] <= 100:
                st.subheader("📊 Distribuição de Valores")
                
                contagem = df[coluna_selecionada].value_counts()
                df_contagem = pd.DataFrame({
                    'Valor': contagem.index,
                    'Contagem': contagem.values,
                    'Percentual': (contagem.values / len(df) * 100)
                })
                
                st.dataframe(
                    df_contagem.style.format({'Contagem': '{:,}', 'Percentual': '{:.1f}%'}),
                    use_container_width=True
                )

# =============================================================================
# TAB 4: VISUALIZAR DADOS
# =============================================================================

with tab4:
    st.header("📊 Visualizar Dados Completos")
    
    # Selecionar colunas para visualizar
    colunas_vis = st.multiselect(
        "Selecione colunas para visualizar",
        options=sorted(df.columns),
        default=sorted(df.columns)[:min(10, len(df.columns))],
        key="colunas_vis_tab4"
    )
    
    if colunas_vis:
        # Filtros
        st.subheader("🔍 Filtros")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        df_filtrado = df.copy()
        
        with col_f1:
            # Filtro por datasource se existir
            if 'datasource' in df.columns:
                datasources = sorted(df['datasource'].dropna().unique())
                ds_selecionados = st.multiselect(
                    "Data Sources",
                    options=datasources,
                    default=datasources[:min(3, len(datasources))],
                    key="ds_selecionados_tab4"
                )
                if ds_selecionados:
                    df_filtrado = df_filtrado[df_filtrado['datasource'].isin(ds_selecionados)]
        
        with col_f2:
            # Filtro por valor mínimo se coluna numérica selecionada
            colunas_num_vis = [c for c in colunas_vis if c in colunas_numericas]
            if colunas_num_vis:
                col_filtro = st.selectbox(
                    "Filtrar por coluna numérica",
                    options=['Nenhum'] + colunas_num_vis,
                    key="col_filtro_tab4"
                )
                if col_filtro != 'Nenhum':
                    # Verificar se há valores válidos
                    col_data = df_filtrado[col_filtro].dropna()
                    if len(col_data) > 0:
                        min_val = st.number_input(
                            f"Valor mínimo de {col_filtro}",
                            value=float(col_data.min()),
                            key=f"min_val_{col_filtro}_tab4"
                        )
                        df_filtrado = df_filtrado[df_filtrado[col_filtro] >= min_val]
                    else:
                        st.info(f"Coluna '{col_filtro}' não tem valores numéricos válidos")
        
        with col_f3:
            # Limite de linhas
            limite_linhas = st.slider("Linhas para mostrar", 10, 1000, 100, key="limite_linhas_tab4")
        
        # Mostrar dados
        st.subheader(f"📋 Dados ({len(df_filtrado):,} registros filtrados)")
        
        # Paginação - CORREÇÃO APLICADA AQUI
        if len(df_filtrado) > 0:
            total_pages = max(1, len(df_filtrado) // limite_linhas + 1)
            
            col_pg1, col_pg2, col_pg3 = st.columns([1, 2, 1])
            
            with col_pg1:
                # CORREÇÃO: Verificar se total_pages é maior que 0
                if total_pages > 0:
                    page_number = st.number_input(
                        "Página", 
                        min_value=1, 
                        max_value=total_pages, 
                        value=1, 
                        key="page_number_tab4"
                    )
                else:
                    page_number = 1
                    st.write("Página: 1")
            
            with col_pg3:
                st.caption(f"Total: {len(df_filtrado):,} registros")
            
            # Calcular índice
            start_idx = (page_number - 1) * limite_linhas
            end_idx = min(start_idx + limite_linhas, len(df_filtrado))
            
            # Formatar DataFrame para exibição
            df_display = df_filtrado[colunas_vis].iloc[start_idx:end_idx].copy()
            
            # Formatar colunas numéricas
            for col in colunas_vis:
                if col in colunas_numericas:
                    # Formatar como número com separadores
                    def format_number(x):
                        if pd.isna(x):
                            return ""
                        elif isinstance(x, (int, np.integer)):
                            return f"{x:,}"
                        elif isinstance(x, (float, np.floating)):
                            return f"{x:,.2f}"
                        return str(x)
                    
                    df_display[col] = df_display[col].apply(format_number)
            
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400
            )
        else:
            st.info("Nenhum dado para mostrar após aplicar filtros")
        
        # Download
        st.subheader("📥 Exportar Dados")
        
        if len(df_filtrado) > 0:
            csv = df_filtrado[colunas_vis].to_csv(index=False)
            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name=f"dados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="download_csv_tab4"
            )
        else:
            st.warning("Nenhum dado para exportar")

# =============================================================================
# TAB 5: ANÁLISE ESTATÍSTICA AVANÇADA
# =============================================================================

with tab5:
    st.header("📐 Análise Estatística Avançada")
    
    if not colunas_numericas:
        st.warning("Nenhuma coluna numérica disponível para análise estatística")
    else:
        # Selecionar variáveis para análise
        st.subheader("🔧 Configurar Análise")
        
        col_analise1, col_analise2 = st.columns(2)
        
        with col_analise1:
            variavel_resposta = st.selectbox(
                "Variável Resposta (Y)",
                options=['Nenhuma'] + colunas_numericas,
                help="Variável que queremos explicar/prever",
                key="variavel_resposta_tab5"
            )
        
        with col_analise2:
            variaveis_explicativas = st.multiselect(
                "Variáveis Explicativas (X)",
                options=[c for c in colunas_numericas if c != variavel_resposta or variavel_resposta == 'Nenhuma'],
                help="Variáveis que podem explicar a variável resposta",
                key="variaveis_explicativas_tab5"
            )
        
        if variavel_resposta != 'Nenhuma' and variaveis_explicativas:
            # Preparar dados
            colunas_analise = [variavel_resposta] + variaveis_explicativas
            dados_analise = df[colunas_analise].apply(pd.to_numeric, errors='coerce').dropna()
            
            if len(dados_analise) < 10:
                st.warning("Dados insuficientes para análise (necessário pelo menos 10 observações)")
            else:
                st.success(f"✅ {len(dados_analise)} observações válidas para análise")
                
                # Análise de correlação
                st.subheader("🔥 Correlações com Variável Resposta")
                
                correlacoes = {}
                for var in variaveis_explicativas:
                    correlacao = dados_analise[variavel_resposta].corr(dados_analise[var])
                    if not pd.isna(correlacao):
                        correlacoes[var] = correlacao
                
                if correlacoes:
                    # Ordenar por força da correlação
                    df_correl = pd.DataFrame({
                        'Variável': list(correlacoes.keys()),
                        'Correlação': list(correlacoes.values()),
                        'Força': [abs(x) for x in correlacoes.values()]
                    }).sort_values('Força', ascending=False)
                    
                    # Adicionar cores
                    def color_correlation(val):
                        if pd.isna(val):
                            return ''
                        if abs(val) > 0.7:
                            return 'background-color: #10b981; color: white; font-weight: bold'
                        elif abs(val) > 0.5:
                            return 'background-color: #f59e0b; color: white; font-weight: bold'
                        elif abs(val) > 0.3:
                            return 'background-color: #ef4444; color: white; font-weight: bold'
                        return ''
                    
                    st.dataframe(
                        df_correl.style.map(
                            lambda x: color_correlation(x) if isinstance(x, (int, float)) else '',
                            subset=['Correlação']
                        ).format({'Correlação': '{:.3f}'}),
                        use_container_width=True
                    )
                    
                    # Gráfico de dispersão para as top 3 correlações
                    if len(correlacoes) > 0:
                        st.subheader("📈 Relações Principais")
                        
                        top_correl = df_correl.head(min(3, len(df_correl)))
                        
                        if not top_correl.empty:
                            cols_graf = st.columns(min(3, len(top_correl)))
                            
                            for idx, row in top_correl.iterrows():
                                var_x = row['Variável']
                                with cols_graf[idx % len(cols_graf)]:
                                    fig_scatter = px.scatter(
                                        dados_analise,
                                        x=var_x,
                                        y=variavel_resposta,
                                        trendline="ols",
                                        title=f"{variavel_resposta} vs {var_x}",
                                        labels={
                                            var_x: var_x,
                                            variavel_resposta: variavel_resposta
                                        }
                                    )
                                    fig_scatter.update_traces(marker=dict(size=5, opacity=0.6))
                                    st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{idx}_tab5")
                
                # Distribuição da variável resposta
                st.subheader(f"📊 Distribuição de {variavel_resposta}")
                
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    # Histograma
                    fig_hist = px.histogram(
                        dados_analise,
                        x=variavel_resposta,
                        nbins=min(50, len(dados_analise)),
                        title=f"Histograma de {variavel_resposta}"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{variavel_resposta}_tab5")
                
                with col_dist2:
                    # Box plot
                    fig_box = px.box(
                        dados_analise,
                        y=variavel_resposta,
                        title=f"Box Plot de {variavel_resposta}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True, key=f"box_{variavel_resposta}_tab5")
                
                # Análise de outliers
                st.subheader("📉 Detecção de Outliers")
                
                try:
                    # Método IQR
                    Q1 = dados_analise[variavel_resposta].quantile(0.25)
                    Q3 = dados_analise[variavel_resposta].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    limite_inferior = Q1 - 1.5 * IQR
                    limite_superior = Q3 + 1.5 * IQR
                    
                    outliers = dados_analise[
                        (dados_analise[variavel_resposta] < limite_inferior) | 
                        (dados_analise[variavel_resposta] > limite_superior)
                    ]
                    
                    col_out1, col_out2, col_out3, col_out4 = st.columns(4)
                    
                    with col_out1:
                        safe_metric("Q1 (25%)", Q1)
                    
                    with col_out2:
                        safe_metric("Q3 (75%)", Q3)
                    
                    with col_out3:
                        safe_metric("IQR", IQR)
                    
                    with col_out4:
                        outlier_count = len(outliers)
                        outlier_pct = (outlier_count / len(dados_analise) * 100) if len(dados_analise) > 0 else 0
                        safe_metric("Outliers", f"{outlier_count} ({outlier_pct:.1f}%)")
                    
                    if not outliers.empty and len(outliers) > 0:
                        with st.expander("📋 Ver Outliers"):
                            st.dataframe(outliers, use_container_width=True)
                except:
                    st.info("Não foi possível calcular outliers para esta variável")

# =============================================================================
# TAB 6: PERFORMANCE DE CAMPANHAS
# =============================================================================

with tab6:
    st.header("🎯 Performance de Campanhas")
    st.markdown("Análise detalhada de campanhas de marketing")
    
    # Verificar coluna de campanhas
    if 'campaign' not in df.columns:
        st.error("❌ Coluna 'campaign' não encontrada nos dados.")
        st.info("ℹ️ Esta análise requer uma coluna chamada 'campaign' para identificar as campanhas.")
        
        # Mostrar colunas disponíveis
        st.subheader("📋 Colunas disponíveis:")
        st.write(sorted(df.columns.tolist()))
        st.stop()
    
    # Listar campanhas disponíveis
    campaigns = sorted(df['campaign'].dropna().unique())
    
    if not campaigns:
        st.error("Nenhuma campanha encontrada na coluna 'campaign'")
        st.stop()
    
    # Sidebar para configuração da análise
    st.sidebar.subheader("🎯 Configuração da Análise de Campanhas")
    
    selected_campaign = st.sidebar.selectbox(
        "Selecione a campanha:",
        options=campaigns,
        index=0,
        help="Escolha uma campanha para análise detalhada"
    )
    
    # Identificar métricas disponíveis para esta campanha
    campaign_data = df[df['campaign'] == selected_campaign].copy()
    
    if campaign_data.empty:
        st.error(f"❌ Nenhum dado encontrado para a campanha '{selected_campaign}'")
        st.stop()
    
    # Encontrar colunas numéricas específicas desta campanha
    campaign_numeric_cols = []
    for col in colunas_numericas:
        if col in campaign_data.columns:
            # Verificar se a coluna tem dados
            if campaign_data[col].notna().any():
                campaign_numeric_cols.append(col)
    
    if not campaign_numeric_cols:
        st.warning("⚠️ Nenhuma métrica numérica disponível para esta campanha.")
        st.stop()
    
    # Selecionar métricas para análise
    selected_metrics = st.sidebar.multiselect(
        "Métricas para análise:",
        options=campaign_numeric_cols,
        default=campaign_numeric_cols[:min(5, len(campaign_numeric_cols))],
        help="Selecione as métricas que deseja analisar"
    )
    
    if not selected_metrics:
        st.warning("Selecione pelo menos uma métrica para análise")
        st.stop()
    
    # ====================
    # DASHBOARD DE MÉTRICAS
    # ====================
    
    st.subheader(f"📊 Dashboard - {selected_campaign}")
    
    # Mostrar período se tiver data
    if 'date' in campaign_data.columns:
        # Converter para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(campaign_data['date']):
            try:
                campaign_data['date'] = pd.to_datetime(campaign_data['date'])
            except:
                pass
        
        if pd.api.types.is_datetime64_any_dtype(campaign_data['date']):
            start_date = campaign_data['date'].min()
            end_date = campaign_data['date'].max()
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                safe_metric("📅 Início", start_date.strftime('%d/%m/%Y') if not pd.isna(start_date) else "N/A")
            with col_info2:
                safe_metric("📅 Término", end_date.strftime('%d/%m/%Y') if not pd.isna(end_date) else "N/A")
            with col_info3:
                try:
                    days_active = (end_date - start_date).days + 1
                    safe_metric("⏱️ Dias ativa", days_active)
                except:
                    safe_metric("⏱️ Dias ativa", "N/A")
    
    # Métricas principais em cards
    st.subheader("📈 Métricas Principais")
    
    # Mostrar até 6 métricas em linha
    num_cols = min(6, len(selected_metrics))
    if num_cols > 0:
        cols = st.columns(num_cols)
        
        for idx, metric in enumerate(selected_metrics[:num_cols]):
            with cols[idx]:
                if metric in campaign_data.columns:
                    try:
                        # Tratar valores NaN
                        metric_data = campaign_data[metric].fillna(0)
                        
                        total_value = float(metric_data.sum())
                        avg_value = float(metric_data.mean()) if len(metric_data) > 0 else 0
                        
                        # Formatar valores
                        if total_value >= 1000000:
                            display_total = f"{total_value/1000000:.1f}M"
                        elif total_value >= 1000:
                            display_total = f"{total_value/1000:.1f}K"
                        elif abs(total_value) < 0.01:
                            display_total = f"{total_value:.4f}"
                        elif abs(total_value) < 1:
                            display_total = f"{total_value:.3f}"
                        else:
                            display_total = f"{total_value:.0f}"
                        
                        safe_metric(
                            metric[:20],  # Limitar nome
                            display_total,
                            delta=f"Média: {avg_value:.2f}"
                        )
                    except Exception as e:
                        safe_metric(metric[:20], f"Erro", delta=str(e)[:20])
    
    # ====================
    # GRÁFICOS
    # ====================
    
    st.subheader("📊 Visualizações")
    
    tab_charts, tab_table = st.tabs(["Gráficos", "Dados"])
    
    with tab_charts:
        # Gráfico de linhas para métricas ao longo do tempo
        if 'date' in campaign_data.columns and selected_metrics:
            try:
                # Preparar dados para gráfico
                plot_data = campaign_data[['date'] + selected_metrics].copy()
                
                # Garantir que date seja datetime
                if not pd.api.types.is_datetime64_any_dtype(plot_data['date']):
                    plot_data['date'] = pd.to_datetime(plot_data['date'], errors='coerce')
                
                # Remover linhas com data inválida
                plot_data = plot_data.dropna(subset=['date'])
                
                if len(plot_data) > 0:
                    fig = go.Figure()
                    
                    colors = px.colors.qualitative.Set1
                    
                    for idx, metric in enumerate(selected_metrics[:5]):  # Limitar a 5 métricas
                        if metric in plot_data.columns:
                            # Normalizar para melhor visualização
                            metric_values = plot_data[metric].fillna(0)
                            
                            # Verificar se há valores válidos
                            if metric_values.max() > 0 and len(metric_values) > 0:
                                normalized = metric_values / metric_values.max()
                            else:
                                normalized = metric_values
                            
                            fig.add_trace(go.Scatter(
                                x=plot_data['date'],
                                y=normalized,
                                name=metric[:20],  # Limitar nome
                                mode='lines+markers',
                                line=dict(color=colors[idx % len(colors)], width=2)
                            ))
                    
                    if len(fig.data) > 0:
                        fig.update_layout(
                            title=f"Evolução das Métricas - {selected_campaign}",
                            xaxis_title="Data",
                            hovermode='x unified',
                            template='plotly_white',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Nenhum dado válido para plotar gráfico de evolução")
            except Exception as e:
                st.error(f"Erro ao criar gráfico de evolução: {str(e)[:100]}")
        
        # Gráfico de barras para métricas totais
        if selected_metrics:
            try:
                totals = []
                metric_names_display = []
                
                for metric in selected_metrics[:8]:  # Limitar a 8 métricas
                    if metric in campaign_data.columns:
                        try:
                            total_val = float(campaign_data[metric].fillna(0).sum())
                            totals.append(total_val)
                            metric_names_display.append(metric[:20])  # Limitar nome
                        except:
                            continue
                
                if totals:
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Bar(
                        x=metric_names_display,
                        y=totals,
                        marker_color='#4f46e5',
                        text=[f"{val:,.0f}" for val in totals],
                        textposition='auto'
                    ))
                    
                    fig2.update_layout(
                        title=f"Totais por Métrica - {selected_campaign}",
                        xaxis_title="Métrica",
                        yaxis_title="Valor Total",
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao criar gráfico de barras: {str(e)[:100]}")
    
    with tab_table:
        # Tabela com dados brutos
        st.subheader("📋 Dados da Campanha")
        
        # Selecionar colunas para mostrar
        display_cols = []
        if 'date' in campaign_data.columns:
            display_cols.append('date')
        
        # Adicionar métricas selecionadas
        display_cols.extend([m for m in selected_metrics if m in campaign_data.columns])
        
        if display_cols:
            # Ordenar por data se disponível
            sort_col = 'date' if 'date' in display_cols else display_cols[0]
            
            try:
                display_df = campaign_data[display_cols].copy()
                
                # Ordenar
                if sort_col in display_df.columns:
                    try:
                        display_df = display_df.sort_values(sort_col, ascending=False)
                    except:
                        pass
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # Estatísticas descritivas
                st.subheader("📊 Estatísticas Descritivas")
                
                try:
                    stats_df = campaign_data[selected_metrics].describe().T
                    st.dataframe(stats_df, use_container_width=True)
                except:
                    st.info("Não foi possível calcular estatísticas descritivas")
                    
            except Exception as e:
                st.error(f"Erro ao exibir dados: {str(e)[:100]}")
    
    # ====================
    # ANÁLISE COM IA (OPCIONAL)
    # ====================
    
    if modelo_texto and selected_metrics:
        st.subheader("🤖 Análise com IA")
        
        if st.button("🔍 Gerar Insights com IA", type="secondary"):
            with st.spinner("Analisando dados com IA..."):
                try:
                    # Preparar dados para a IA
                    summary_stats = {}
                    for metric in selected_metrics[:5]:  # Limitar a 5 métricas
                        if metric in campaign_data.columns:
                            metric_data = campaign_data[metric].fillna(0)
                            if len(metric_data) > 0:
                                summary_stats[metric] = {
                                    'total': float(metric_data.sum()),
                                    'media': float(metric_data.mean()),
                                    'min': float(metric_data.min()),
                                    'max': float(metric_data.max())
                                }
                    
                    # Informações de data
                    date_info = ""
                    if 'date' in campaign_data.columns:
                        try:
                            dates = campaign_data['date'].dropna()
                            if len(dates) > 0:
                                start = dates.min()
                                end = dates.max()
                                date_info = f"Período: {start.strftime('%d/%m/%Y')} a {end.strftime('%d/%m/%Y')}"
                        except:
                            pass
                    
                    prompt = f"""
                    Analise os dados desta campanha de marketing:
                    
                    Campanha: {selected_campaign}
                    {date_info}
                    Total de registros: {len(campaign_data)}
                    
                    Métricas analisadas:
                    {json.dumps(summary_stats, indent=2)}
                    
                    Forneça uma análise concisa em português com:
                    1. Resumo executivo (1 parágrafo)
                    2. 2-3 pontos fortes
                    3. 2-3 pontos de atenção
                    4. 3 recomendações práticas
                    
                    Seja direto e baseado apenas nos dados fornecidos.
                    """
                    
                    response = modelo_texto.generate_content(prompt)
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Erro na análise com IA: {str(e)[:100]}")
    
    # ====================
    # DOWNLOAD DE DADOS
    # ====================
    
    st.subheader("📥 Exportar Dados")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # CSV da campanha
        try:
            csv_data = campaign_data.to_csv(index=False)
            st.download_button(
                label="💾 Baixar Dados da Campanha (CSV)",
                data=csv_data,
                file_name=f"campanha_{selected_campaign}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except:
            st.error("Erro ao gerar arquivo CSV")
    
    with col_dl2:
        # Relatório simples
        try:
            report_text = f"""
            RELATÓRIO DE CAMPANHA - {selected_campaign}
            Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            
            PERÍODO: {date_info if 'date_info' in locals() else "Não disponível"}
            TOTAL DE REGISTROS: {len(campaign_data)}
            
            MÉTRICAS ANALISADAS:
            """
            
            for metric in selected_metrics[:10]:
                if metric in campaign_data.columns:
                    metric_data = campaign_data[metric].fillna(0)
                    total = metric_data.sum()
                    avg = metric_data.mean() if len(metric_data) > 0 else 0
                    report_text += f"\n- {metric}: Total={total:,.2f}, Média={avg:,.2f}"
            
            st.download_button(
                label="📄 Baixar Relatório (TXT)",
                data=report_text,
                file_name=f"relatorio_{selected_campaign}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        except:
            st.error("Erro ao gerar relatório")

# =============================================================================
# RODAPÉ
# =============================================================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if not df.empty:
        st.caption(f"📊 Dados Gerais: {len(df):,} registros • {len(df.columns)} colunas")

with footer_col2:
    if 'campaign' in df.columns:
        num_campaigns = len(df['campaign'].unique())
        st.caption(f"🎯 Campanhas: {num_campaigns:,} campanhas ativas")

with footer_col3:
    st.caption(f"⏰ Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Nota sobre IA
if modelo_texto:
    st.sidebar.success("✅ Gemini disponível para análises")
else:
    st.sidebar.info("ℹ️ Gemini não configurado - Análises básicas apenas")
