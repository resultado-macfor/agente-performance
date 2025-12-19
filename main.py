# app_completo.py - App Analytics Platform Completo CORRIGIDO
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

# Tentar importar Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

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
    .gemini-analysis {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .analysis-section {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .gemini-response {
        background: #f8fafc;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #10b981;
        white-space: pre-wrap;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 14px;
    }
    .filter-section {
        background: #f0f9ff;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e2e8f0;
    }
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .data-table {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="header-gradient"><h1>📊 Analytics Platform - Análise Completa de Dados</h1></div>', unsafe_allow_html=True)

# =============================================================================
# CONFIGURAÇÃO GEMINI
# =============================================================================

# Inicializar modelo Gemini
modelo_texto = None

# Procurar chave Gemini
gemini_api_key = os.getenv("GEM_API_KEY")

if not gemini_api_key:
    possible_keys = ["GEN_API_KEY", "GEN_API_KEY2", "GEMINI_API_KEY", "GOOGLE_API_KEY"]
    for key_name in possible_keys:
        key_value = os.getenv(key_name)
        if key_value:
            gemini_api_key = key_value
            break

if not gemini_api_key and hasattr(st, 'secrets'):
    secrets_keys = ["GEM_API_KEY", "GEN_API_KEY", "GEN_API_KEY2", "GEMINI_API_KEY", "GOOGLE_API_KEY"]
    for key_name in secrets_keys:
        if key_name in st.secrets:
            gemini_api_key = st.secrets[key_name]
            break

# Configurar Gemini
if gemini_api_key and GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=gemini_api_key)
        modelo_texto = genai.GenerativeModel("gemini-2.5-flash")
        st.sidebar.success("✅ Gemini configurado!")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro Gemini: {str(e)[:50]}")
        modelo_texto = None
elif gemini_api_key and not GEMINI_AVAILABLE:
    st.sidebar.warning("⚠️ Biblioteca Gemini não disponível")
else:
    st.sidebar.info("ℹ️ Gemini não configurado")

# =============================================================================
# FUNÇÕES GEMINI
# =============================================================================

def generate_gemini_analysis(df_filtered, analysis_type="overall", user_instructions=""):
    """Gera análise com Gemini"""
    
    if not modelo_texto:
        return "⚠️ Gemini não configurado."
    
    if df_filtered.empty:
        return "❌ Nenhum dado disponível."
    
    try:
        num_records = len(df_filtered)
        has_campaigns = 'campaign' in df_filtered.columns
        has_date = 'date' in df_filtered.columns
        
        # Formatar datas se disponível
        date_info = "N/A"
        if has_date and 'date' in df_filtered.columns and not df_filtered['date'].isna().all():
            # Garantir que a coluna date é datetime
            try:
                if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
                    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
                
                # Filtrar apenas datas válidas
                valid_dates = df_filtered['date'].dropna()
                if len(valid_dates) > 0:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    if isinstance(min_date, pd.Timestamp) and isinstance(max_date, pd.Timestamp):
                        date_info = f"{min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}"
            except:
                date_info = "N/A"
        
        # Informações gerais
        general_info = f"""
        ## 📊 CONTEXTO GERAL:
        - **Total de registros:** {num_records:,}
        - **Período:** {date_info}
        - **Colunas disponíveis:** {len(df_filtered.columns)}
        - **Campanhas:** {df_filtered['campaign'].nunique() if has_campaigns else 'N/A'}
        """
        
        # Análise de campanhas
        campaign_analysis = ""
        if has_campaigns and 'campaign' in df_filtered.columns:
            try:
                campaign_stats = df_filtered['campaign'].value_counts()
                campaign_analysis = f"""
                ## 🎯 ANÁLISE DE CAMPANHAS:
                - **Total de campanhas:** {len(campaign_stats)}
                - **Top 5 campanhas por volume:**
                """
                for i, (campaign, count) in enumerate(campaign_stats.head(5).items(), 1):
                    campaign_name = str(campaign)[:30] + "..." if len(str(campaign)) > 30 else str(campaign)
                    campaign_analysis += f"  {i}. **{campaign_name}**: {count:,} registros\n"
            except:
                campaign_analysis = "\n## 🎯 ANÁLISE DE CAMPANHAS: (Erro na análise)\n"
        
        # Análise de métricas
        numeric_cols = []
        for col in df_filtered.columns:
            try:
                if pd.api.types.is_numeric_dtype(df_filtered[col]):
                    numeric_cols.append(col)
                else:
                    # Tentar converter para numérico
                    sample = df_filtered[col].dropna().head(10)
                    if len(sample) > 0:
                        pd.to_numeric(sample, errors='raise')
                        numeric_cols.append(col)
            except:
                continue
        
        metric_analysis = ""
        if numeric_cols:
            important_metrics = []
            priority_metrics = ['spend', 'revenue', 'conversions', 'impressions', 'clicks', 'cpc', 'cpm', 'ctr', 'roas']
            
            for metric in priority_metrics:
                for col in numeric_cols:
                    if metric in col.lower():
                        important_metrics.append(col)
                        break
            
            if not important_metrics:
                important_metrics = numeric_cols[:5]
            
            metric_analysis = "## 📈 MÉTRICAS PRINCIPAIS:\n"
            for metric in important_metrics[:8]:
                if metric in df_filtered.columns:
                    try:
                        metric_data = pd.to_numeric(df_filtered[metric], errors='coerce').dropna()
                        if len(metric_data) > 0:
                            total = metric_data.sum()
                            avg = metric_data.mean()
                            metric_analysis += f"\n**{metric}**:\n"
                            metric_analysis += f"- **Total:** {total:,.2f}\n"
                            metric_analysis += f"- **Média:** {avg:,.2f}\n"
                    except:
                        continue
        
        # Dadosource analysis
        datasource_analysis = ""
        if 'datasource' in df_filtered.columns:
            try:
                ds_stats = df_filtered['datasource'].value_counts()
                datasource_analysis = "\n## 📱 DATA SOURCES:\n"
                for ds, count in ds_stats.head().items():
                    percentage = (count / num_records) * 100
                    datasource_analysis += f"- **{ds}**: {count:,} registros ({percentage:.1f}%)\n"
            except:
                datasource_analysis = "\n## 📱 DATA SOURCES: (Erro na análise)\n"
        
        # Sample data - limitar tamanho para evitar erros no prompt
        try:
            sample_df = df_filtered.head(20).copy()
            # Converter colunas para string para evitar problemas de formatação
            for col in sample_df.columns:
                sample_df[col] = sample_df[col].astype(str)
            sample_data = sample_df.to_string()
        except:
            sample_data = "Erro ao gerar amostra"
        
        # Build prompt
        if analysis_type == "overall":
            focus_text = "ANÁLISE GERAL DE PERFORMANCE"
        elif analysis_type == "trends":
            focus_text = "ANÁLISE DE TENDÊNCIAS"
        elif analysis_type == "efficiency":
            focus_text = "ANÁLISE DE EFICIÊNCIA"
        else:
            focus_text = "ANÁLISE COMPLETA"
        
        prompt = f"""
        # {focus_text} - RELATÓRIO EXECUTIVO
        
        {general_info}
        
        {campaign_analysis}
        
        {metric_analysis}
        
        {datasource_analysis}
        
        ## 🎯 FOCO DA ANÁLISE:
        {analysis_type.upper()}
        
        ## 📝 INSTRUÇÕES DO USUÁRIO:
        {user_instructions if user_instructions else "Forneça uma análise completa do desempenho geral."}
        
        ## 📋 DADOS DE AMOSTRA (20 primeiros registros):
        {sample_data}
        
        ## 📊 TAREFA:
        
        Analise os dados acima e crie um relatório executivo em português com:
        
        1. **📈 RESUMO EXECUTIVO** (1-2 parágrafos)
        2. **🎯 ANÁLISE DAS CAMPANHAS** (se disponível)
        3. **💰 ANÁLISE FINANCEIRA** (investimento, ROI, eficiência)
        4. **📊 ANÁLISE DE MÉTRICAS** (principais KPIs, tendências)
        5. **🔍 INSIGHTS ESTRATÉGICOS** (3-5 insights principais)
        6. **🚀 RECOMENDAÇÕES ACIONÁVEIS** (5-7 recomendações)
        7. **📅 PRÓXIMOS PASSOS** (plano de ação)
        
        Seja específico, baseado em dados e prático.
        """
        
        with st.spinner("🤖 Gemini está analisando..."):
            response = modelo_texto.generate_content(prompt)
            return response.text
    
    except Exception as e:
        return f"❌ Erro: {str(e)[:200]}"

# =============================================================================
# CONEXÃO BIGQUERY
# =============================================================================

@st.cache_resource
def get_bigquery_client():
    """Cria cliente BigQuery"""
    try:
        # Tentar várias opções
        service_account_info = None
        
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            service_account_info = dict(st.secrets["gcp_service_account"])
            if isinstance(service_account_info.get("private_key"), str):
                service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
        
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
        
        elif 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
            credentials_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
            service_account_info = json.loads(credentials_json)
        
        else:
            st.error("❌ Credenciais não encontradas!")
            return None
        
        if not service_account_info:
            st.error("❌ Não foi possível obter as credenciais")
            return None
        
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
        st.error(f"❌ Erro na conexão: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_all_columns_data(_client, data_inicio=None, data_fim=None, data_sources=None, limit=50000):
    """Carrega TODAS as colunas"""
    try:
        st.info("🔍 Carregando dados...")
        
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
        
        # Garantir que a coluna date é datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        st.success(f"✅ {len(df):,} registros, {len(df.columns)} colunas")
        
        return df
    
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def safe_metric(label, value, delta=None):
    """Função segura para métricas"""
    try:
        if pd.isna(value):
            value = 0
        
        if isinstance(value, (int, np.integer)):
            display_value = f"{int(value):,}"
        elif isinstance(value, (float, np.floating)):
            if abs(value) < 0.01:
                display_value = f"{value:.4f}"
            elif abs(value) < 1:
                display_value = f"{value:.3f}"
            elif abs(value) < 1000:
                display_value = f"{value:.2f}"
            else:
                display_value = f"{value:,.0f}"
        else:
            try:
                num_val = float(value)
                display_value = f"{num_val:,.2f}"
            except:
                display_value = str(value)
        
        if delta is not None:
            if pd.isna(delta):
                delta = None
            elif isinstance(delta, (int, float, np.integer, np.floating)):
                delta = f"{delta:+.2f}"
        
        return st.metric(label, display_value, delta=delta)
    
    except:
        return st.metric(label, "Erro")

def identificar_colunas_numericas(df):
    """Identifica colunas numéricas"""
    colunas_numericas = []
    
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                colunas_numericas.append(col)
            else:
                # Tentar converter para ver se é numérico
                amostra = df[col].dropna().head(10)
                if len(amostra) > 0:
                    try:
                        pd.to_numeric(amostra)
                        colunas_numericas.append(col)
                    except:
                        pass
        except:
            continue
    
    return colunas_numericas

def analisar_coluna(df, coluna):
    """Analisa uma coluna específica"""
    if coluna not in df.columns:
        return None
    
    try:
        dados_coluna = df[coluna]
        
        total = int(len(dados_coluna))
        nao_nulos = int(dados_coluna.notna().sum())
        nulos = int(dados_coluna.isna().sum())
        valores_unicos = int(dados_coluna.nunique())
        
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
                analise.update({'tipo_detalhado': 'Numérica (vazia)'})
        elif dados_coluna.dtype == 'object':
            value_counts = dados_coluna.value_counts()
            analise.update({
                'tipo_detalhado': 'Texto/Categórica',
                'valores_mais_comuns': value_counts.head(10).to_dict(),
                'valor_mais_frequente': value_counts.index[0] if len(value_counts) > 0 else None,
                'frequencia_valor_mais_comum': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            })
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
                analise.update({'tipo_detalhado': 'Data (vazia)'})
        else:
            analise.update({'tipo_detalhado': 'Outro'})
            
        return analise
        
    except Exception as e:
        return {
            'nome': coluna,
            'tipo': 'Erro',
            'tipo_detalhado': f'Erro na análise: {str(e)[:50]}',
            'total': 0,
            'nao_nulos': 0,
            'nulos': 0,
            'percentual_nulos': 0.0,
            'valores_unicos': 0
        }

def criar_visualizacao_coluna(df, coluna):
    """Cria visualização para coluna"""
    if coluna not in df.columns:
        return None
    
    try:
        dados = df[coluna].dropna()
        
        if len(dados) == 0:
            return None
        
        if pd.api.types.is_numeric_dtype(df[coluna]):
            # Converter para numérico se não for
            dados_numeric = pd.to_numeric(dados, errors='coerce').dropna()
            if len(dados_numeric) == 0:
                return None
            
            fig = px.histogram(
                x=dados_numeric,
                nbins=min(50, len(dados_numeric)),
                title=f"Distribuição de {coluna}",
                marginal="box"
            )
            return fig
        
        elif df[coluna].nunique() <= 50:
            contagem = df[coluna].value_counts().head(20)
            if len(contagem) == 0:
                return None
            
            fig = px.bar(
                x=contagem.index.astype(str),
                y=contagem.values,
                title=f"Top 20 Valores em {coluna}",
                labels={'x': coluna, 'y': 'Contagem'}
            )
            fig.update_xaxes(tickangle=45)
            return fig
        
        elif pd.api.types.is_datetime64_any_dtype(df[coluna]):
            try:
                # Garantir que é datetime
                dados_dt = pd.to_datetime(dados, errors='coerce').dropna()
                if len(dados_dt) == 0:
                    return None
                
                contagem_diaria = pd.Series(dados_dt.dt.date).value_counts().sort_index().reset_index()
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
        st.error(f"Erro ao criar visualização: {str(e)[:100]}")
        return None

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================

# Inicializar estado
if 'df_completo' not in st.session_state:
    st.session_state.df_completo = pd.DataFrame()
if 'colunas_numericas' not in st.session_state:
    st.session_state.colunas_numericas = []
if 'gemini_analysis' not in st.session_state:
    st.session_state.gemini_analysis = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Testar conexão
    if st.button("Testar Conexão BigQuery"):
        with st.spinner("Conectando..."):
            client = get_bigquery_client()
            if client:
                st.success("✅ Conexão OK!")
    
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
    if st.button("📊 Carregar Dados", use_container_width=True, type="primary"):
        with st.spinner("Carregando..."):
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
                    st.session_state.colunas_numericas = identificar_colunas_numericas(df)
                    st.success(f"✅ {len(df):,} registros carregados")
                    st.session_state.gemini_analysis = None
                else:
                    st.error("Nenhum dado encontrado")
            else:
                st.error("❌ Não foi possível conectar.")

# Verificar dados
df = st.session_state.df_completo
colunas_numericas = st.session_state.colunas_numericas

if df.empty:
    st.warning("📭 Nenhum dado carregado. Use o botão na sidebar.")
    st.stop()

# Abas principais
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Visão Geral", 
    "📈 Análise Numérica", 
    "🔍 Explorar Colunas", 
    "📊 Visualizar Dados",
    "🎯 Performance",
    "🤖 Análise com IA"
])

# =============================================================================
# TAB 1: VISÃO GERAL
# =============================================================================

with tab1:
    st.header("📋 Visão Geral das Colunas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        safe_metric("Total de Colunas", len(df.columns))
    
    with col2:
        safe_metric("Colunas Numéricas", len(colunas_numericas))
    
    with col3:
        safe_metric("Total de Registros", len(df))
    
    with col4:
        try:
            memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
            safe_metric("Uso de Memória", memoria_mb)
        except:
            safe_metric("Uso de Memória", "N/A")
    
    # Listar colunas
    st.subheader("📊 Detalhes de Cada Coluna")
    
    col_filtro1, col_filtro2 = st.columns(2)
    
    with col_filtro1:
        tipo_filtro = st.selectbox(
            "Filtrar por tipo",
            ["Todas", "Numéricas", "Texto", "Datas"],
            key="filtro_tipo_tab1"
        )
    
    with col_filtro2:
        pesquisa_coluna = st.text_input("🔍 Pesquisar coluna", "", key="pesquisa_coluna_tab1")
    
    # Preparar lista
    colunas_para_mostrar = []
    
    for col in df.columns:
        incluir = True
        
        if tipo_filtro == "Numéricas":
            incluir = col in colunas_numericas
        elif tipo_filtro == "Texto":
            incluir = df[col].dtype == 'object' and col not in colunas_numericas
        elif tipo_filtro == "Datas":
            incluir = pd.api.types.is_datetime64_any_dtype(df[col])
        
        if pesquisa_coluna and pesquisa_coluna.lower() not in col.lower():
            incluir = False
        
        if incluir:
            colunas_para_mostrar.append(col)
    
    # Mostrar informações
    for col in sorted(colunas_para_mostrar)[:50]:
        analise = analisar_coluna(df, col)
        
        if analise:
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
                    st.subheader("📈 Estatísticas")
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                    
                    with col_stats1:
                        safe_metric("Média", analise.get('media', 0))
                        safe_metric("Min", analise.get('min', 0))
                    
                    with col_stats2:
                        safe_metric("Mediana", analise.get('mediana', 0))
                        safe_metric("Max", analise.get('max', 0))
                    
                    with col_stats3:
                        safe_metric("Q1 (25%)", analise.get('q1', 0))
                    
                    with col_stats4:
                        safe_metric("Q3 (75%)", analise.get('q3', 0))

# =============================================================================
# TAB 2: ANÁLISE NUMÉRICA
# =============================================================================

with tab2:
    st.header("📈 Análise de Colunas Numéricas")
    
    if not colunas_numericas:
        st.warning("Nenhuma coluna numérica")
    else:
        st.success(f"✅ {len(colunas_numericas)} colunas numéricas")
        
        colunas_selecionadas = st.multiselect(
            "Selecione colunas para análise",
            options=colunas_numericas,
            default=colunas_numericas[:min(5, len(colunas_numericas))],
            key="colunas_selecionadas_tab2"
        )
        
        if colunas_selecionadas:
            # Estatísticas
            st.subheader("📊 Estatísticas Descritivas")
            
            # Converter para numérico
            df_numeric = df[colunas_selecionadas].apply(pd.to_numeric, errors='coerce')
            
            stats_df = df_numeric.describe().T
            stats_df['missing'] = df_numeric.isna().sum()
            stats_df['missing_pct'] = (df_numeric.isna().sum() / len(df) * 100)
            
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
            
            try:
                st.dataframe(
                    stats_df.style.format(formatar_numero),
                    use_container_width=True
                )
            except:
                st.dataframe(stats_df, use_container_width=True)
            
            # Histogramas
            if len(colunas_selecionadas) > 0:
                st.subheader("📈 Distribuições")
                
                num_cols = min(3, len(colunas_selecionadas))
                cols_vis = st.columns(num_cols)
                
                for idx, col in enumerate(colunas_selecionadas[:num_cols*3]):
                    with cols_vis[idx % num_cols]:
                        fig = criar_visualizacao_coluna(df, col)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"Não foi possível criar gráfico para {col}")
            
            # Correlações
            if len(colunas_selecionadas) >= 2:
                st.subheader("🔥 Correlações")
                
                try:
                    df_numeric_corr = df_numeric.copy()
                    correlacao = df_numeric_corr.corr()
                    
                    fig_corr = px.imshow(
                        correlacao,
                        text_auto='.2f',
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlações"
                    )
                    fig_corr.update_layout(height=600)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
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
                        st.info("Sem correlações fortes (> 0.3)")
                        
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
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Não foi possível criar visualização para esta coluna")
            
            # Valores
            st.subheader("📋 Amostra de Valores")
            
            col_amostra1, col_amostra2 = st.columns(2)
            
            with col_amostra1:
                st.write("**Primeiros 10:**")
                try:
                    primeiros = df[coluna_selecionada].head(10).tolist()
                    primeiros_str = [str(x) for x in primeiros]
                    st.write(primeiros_str)
                except:
                    st.write("Erro ao mostrar valores")
            
            with col_amostra2:
                st.write("**Últimos 10:**")
                try:
                    ultimos = df[coluna_selecionada].tail(10).tolist()
                    ultimos_str = [str(x) for x in ultimos]
                    st.write(ultimos_str)
                except:
                    st.write("Erro ao mostrar valores")
            
            # Distribuição
            if analise['tipo_detalhado'] == 'Texto/Categórica' and analise['valores_unicos'] <= 100:
                st.subheader("📊 Distribuição")
                
                try:
                    contagem = df[coluna_selecionada].value_counts()
                    df_contagem = pd.DataFrame({
                        'Valor': contagem.index.astype(str),
                        'Contagem': contagem.values,
                        'Percentual': (contagem.values / len(df) * 100)
                    })
                    
                    st.dataframe(
                        df_contagem.style.format({'Contagem': '{:,}', 'Percentual': '{:.1f}%'}),
                        use_container_width=True
                    )
                except:
                    st.error("Erro ao calcular distribuição")

# =============================================================================
# TAB 4: VISUALIZAR DADOS
# =============================================================================

with tab4:
    st.header("📊 Visualizar Dados Completos")
    
    # Selecionar colunas
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
            colunas_num_vis = [c for c in colunas_vis if c in colunas_numericas]
            if colunas_num_vis:
                col_filtro = st.selectbox(
                    "Filtrar por coluna numérica",
                    options=['Nenhum'] + colunas_num_vis,
                    key="col_filtro_tab4"
                )
                if col_filtro != 'Nenhum':
                    try:
                        col_data = pd.to_numeric(df_filtrado[col_filtro], errors='coerce').dropna()
                        if len(col_data) > 0:
                            min_val = st.number_input(
                                f"Valor mínimo de {col_filtro}",
                                value=float(col_data.min()),
                                key=f"min_val_{col_filtro}_tab4"
                            )
                            df_filtrado = df_filtrado[pd.to_numeric(df_filtrado[col_filtro], errors='coerce') >= min_val]
                    except:
                        st.warning(f"Não foi possível filtrar por {col_filtro}")
        
        with col_f3:
            limite_linhas = st.slider("Linhas para mostrar", 10, 1000, 100, key="limite_linhas_tab4")
        
        # Mostrar dados
        st.subheader(f"📋 Dados ({len(df_filtrado):,} registros)")
        
        if len(df_filtrado) > 0:
            total_pages = max(1, len(df_filtrado) // limite_linhas + 1)
            
            col_pg1, col_pg2, col_pg3 = st.columns([1, 2, 1])
            
            with col_pg1:
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
            
            # Formatar DataFrame
            df_display = df_filtrado[colunas_vis].iloc[start_idx:end_idx].copy()
            
            # Formatar números e datas
            for col in colunas_vis:
                if col in colunas_numericas:
                    try:
                        df_display[col] = df_display[col].apply(
                            lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else ""
                        )
                    except:
                        pass
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')
                    except:
                        pass
            
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400
            )
        else:
            st.info("Nenhum dado após filtros")
        
        # Download
        st.subheader("📥 Exportar")
        
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
# TAB 5: PERFORMANCE
# =============================================================================

with tab5:
    st.header("🎯 Análise de Performance")
    
    if 'campaign' not in df.columns:
        st.error("❌ Coluna 'campaign' não encontrada.")
    else:
        # Métricas gerais
        st.subheader("📊 Métricas Gerais")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                num_campaigns = df['campaign'].nunique()
                safe_metric("Campanhas", num_campaigns)
            except:
                safe_metric("Campanhas", "Erro")
        
        with col2:
            if 'date' in df.columns:
                try:
                    # Garantir que é datetime
                    df_date = df['date'].dropna()
                    if len(df_date) > 0:
                        if not pd.api.types.is_datetime64_any_dtype(df_date):
                            df_date = pd.to_datetime(df_date, errors='coerce')
                        days = (df_date.max() - df_date.min()).days + 1
                        safe_metric("Dias", days)
                    else:
                        safe_metric("Dias", 0)
                except:
                    safe_metric("Dias", "Erro")
        
        with col3:
            if 'datasource' in df.columns:
                try:
                    sources = df['datasource'].nunique()
                    safe_metric("Data Sources", sources)
                except:
                    safe_metric("Data Sources", "Erro")
        
        with col4:
            try:
                num_campaigns_val = df['campaign'].nunique()
                records_per_campaign = len(df) / num_campaigns_val if num_campaigns_val > 0 else 0
                safe_metric("Média Reg/Camp", f"{records_per_campaign:.1f}")
            except:
                safe_metric("Média Reg/Camp", "Erro")
        
        # Análise por campanha
        st.subheader("📈 Top Campanhas")
        
        if 'campaign' in df.columns:
            try:
                campaign_stats = df['campaign'].value_counts().head(10)
                
                fig = px.bar(
                    x=campaign_stats.index.astype(str),
                    y=campaign_stats.values,
                    title="Top 10 Campanhas por Volume",
                    labels={'x': 'Campanha', 'y': 'Registros'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao criar gráfico: {str(e)[:100]}")
        
        # Métricas financeiras
        st.subheader("💰 Métricas Financeiras")
        
        financial_metrics = []
        for metric in ['spend', 'revenue', 'conversions', 'roas', 'cpc']:
            for col in colunas_numericas:
                if metric in col.lower():
                    financial_metrics.append(col)
                    break
        
        if financial_metrics:
            cols = st.columns(min(4, len(financial_metrics)))
            for idx, metric in enumerate(financial_metrics[:4]):
                with cols[idx]:
                    if metric in df.columns:
                        try:
                            total = pd.to_numeric(df[metric], errors='coerce').sum()
                            safe_metric(metric, total)
                        except:
                            safe_metric(metric, "Erro")

# =============================================================================
# TAB 6: ANÁLISE COM IA - CORRIGIDO
# =============================================================================

with tab6:
    st.header("🤖 Análise com Gemini IA")
    
    if not modelo_texto:
        st.error("❌ Gemini não configurado!")
        st.info("Configure a chave do Gemini nas variáveis de ambiente ou secrets.")
        st.stop()
    
    if df.empty:
        st.warning("📭 Nenhum dado carregado.")
        st.stop()
    
    # Filtros para análise
    st.markdown("### 🔍 Filtros para Análise")
    
    with st.expander("⚙️ Configurar", expanded=True):
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            if 'datasource' in df.columns:
                datasources = sorted(df['datasource'].dropna().unique())
                selected_ds = st.multiselect(
                    "Data Sources:",
                    options=datasources,
                    default=datasources[:min(3, len(datasources))]
                )
            else:
                selected_ds = None
            
            if 'date' in df.columns:
                try:
                    # Garantir que a coluna date é datetime
                    date_series = df['date'].dropna()
                    if len(date_series) > 0:
                        if not pd.api.types.is_datetime64_any_dtype(date_series):
                            date_series = pd.to_datetime(date_series, errors='coerce')
                        
                        min_date = date_series.min().date()
                        max_date = date_series.max().date()
                        
                        date_range = st.date_input(
                            "Período:",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date
                        )
                    else:
                        st.info("Sem datas disponíveis")
                        date_range = None
                except Exception as e:
                    st.error(f"Erro com datas: {str(e)[:100]}")
                    date_range = None
            else:
                date_range = None
        
        with col_filter2:
            if 'campaign' in df.columns:
                campaigns = sorted(df['campaign'].dropna().unique())
                selected_campaigns = st.multiselect(
                    "Campanhas (opcional):",
                    options=campaigns
                )
            else:
                selected_campaigns = None
            
            max_records = st.slider(
                "Máximo de registros:",
                min_value=100,
                max_value=min(10000, len(df)),
                value=min(5000, len(df)),
                step=100
            )
    
    # Aplicar filtros - CORREÇÃO DO ERRO PRINCIPAL
    df_filtered = df.copy()
    
    # Filtro por datasource
    if selected_ds and 'datasource' in df_filtered.columns and len(selected_ds) > 0:
        df_filtered = df_filtered[df_filtered['datasource'].isin(selected_ds)]
    
    # Filtro por data - CORREÇÃO AQUI
    if date_range and len(date_range) == 2 and 'date' in df_filtered.columns:
        start_date, end_date = date_range
        
        # Garantir que a coluna date é datetime
        if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
            df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
        
        # Filtrar apenas datas válidas
        mask = df_filtered['date'].notna()
        
        # Converter start_date e end_date para datetime
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        # Aplicar filtro de data
        df_filtered = df_filtered[
            mask & 
            (df_filtered['date'] >= start_dt) & 
            (df_filtered['date'] <= end_dt)
        ]
    
    # Filtro por campanha
    if selected_campaigns and 'campaign' in df_filtered.columns and len(selected_campaigns) > 0:
        df_filtered = df_filtered[df_filtered['campaign'].isin(selected_campaigns)]
    
    # Limitar registros
    df_filtered = df_filtered.head(max_records)
    
    # Estatísticas
    st.markdown("### 📊 Dados Selecionados")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        safe_metric("Registros", len(df_filtered))
    
    with col_stat2:
        if 'campaign' in df_filtered.columns:
            try:
                num_campaigns = df_filtered['campaign'].nunique()
                safe_metric("Campanhas", num_campaigns)
            except:
                safe_metric("Campanhas", "Erro")
    
    with col_stat3:
        if 'datasource' in df_filtered.columns:
            try:
                num_sources = df_filtered['datasource'].nunique()
                safe_metric("Data Sources", num_sources)
            except:
                safe_metric("Data Sources", "Erro")
    
    with col_stat4:
        if 'date' in df_filtered.columns:
            try:
                # Garantir que é datetime
                date_series = df_filtered['date'].dropna()
                if len(date_series) > 0:
                    if not pd.api.types.is_datetime64_any_dtype(date_series):
                        date_series = pd.to_datetime(date_series, errors='coerce')
                    period_days = (date_series.max() - date_series.min()).days + 1
                    safe_metric("Dias", period_days)
                else:
                    safe_metric("Dias", 0)
            except:
                safe_metric("Dias", "Erro")
    
    # Configuração
    st.markdown("### 🎯 Configuração")
    
    analysis_focus = st.selectbox(
        "Foco da Análise:",
        options=["overall", "trends", "efficiency", "complete"],
        format_func=lambda x: {
            "overall": "📈 Performance Geral",
            "trends": "📊 Tendências", 
            "efficiency": "💰 Eficiência",
            "complete": "🏆 Análise Completa"
        }[x]
    )
    
    user_instructions = st.text_area(
        "📝 Instruções (opcional):",
        placeholder="Ex: Foque no ROI, identifique as melhores campanhas, analise tendências por data source...",
        height=100
    )
    
    # Gerar análise
    st.markdown("### 🚀 Gerar Análise")
    
    generate_button = st.button("🤖 Gerar Análise com Gemini", type="primary", use_container_width=True)
    
    if generate_button:
        if df_filtered.empty:
            st.error("❌ Nenhum dado após filtros.")
        else:
            with st.spinner(f"🤖 Analisando {len(df_filtered):,} registros..."):
                try:
                    analysis_result = generate_gemini_analysis(
                        df_filtered, 
                        analysis_focus, 
                        user_instructions
                    )
                    st.session_state.gemini_analysis = analysis_result
                    st.success("✅ Análise concluída!")
                except Exception as e:
                    st.error(f"❌ Erro ao gerar análise: {str(e)[:200]}")
    
    # Mostrar análise
    if st.session_state.gemini_analysis:
        st.markdown("---")
        st.markdown("### 📄 Relatório de Análise")
        
        # Ações
        col_actions1, col_actions2 = st.columns(2)
        
        with col_actions1:
            analysis_text = st.session_state.gemini_analysis
            st.download_button(
                label="💾 Baixar Relatório",
                data=analysis_text,
                file_name=f"analise_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_actions2:
            if st.button("🔄 Nova Análise", use_container_width=True):
                st.session_state.gemini_analysis = None
                st.rerun()
        
        # Mostrar análise
        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
        st.markdown(st.session_state.gemini_analysis)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Instruções
        st.info("""
        ## 📋 Como usar:
        
        1. **Carregue os dados** na sidebar
        2. **Ajuste os filtros** acima
        3. **Selecione o foco** da análise
        4. **Adicione instruções** se desejar
        5. **Clique em 'Gerar Análise'**
        
        ## 🎯 Você receberá:
        
        - 📊 Resumo executivo
        - 🎯 Análise de campanhas
        - 💰 Insights financeiros
        - 📈 Tendências identificadas
        - 🚀 Recomendações acionáveis
        """)

# =============================================================================
# RODAPÉ
# =============================================================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if not df.empty:
        st.caption(f"📊 Dados: {len(df):,} registros")

with footer_col2:
    if 'campaign' in df.columns:
        try:
            num_campaigns = df['campaign'].nunique()
            st.caption(f"🎯 Campanhas: {num_campaigns}")
        except:
            st.caption("🎯 Campanhas: Erro")

with footer_col3:
    st.caption(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Status Gemini
if modelo_texto:
    st.sidebar.success("✅ Gemini ativo")
else:
    st.sidebar.info("ℹ️ Gemini inativo")
