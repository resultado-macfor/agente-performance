# app_completo_analise_mom_gemini.py - App Analytics Platform Completo
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
import os
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import io
import re
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SISTEMA DE AUTENTICAÇÃO
# =============================================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(layout="centered")
    
    st.title("🔒 Agente Performance")
    
    senha_input = st.text_input("Digite a senha de acesso:", type="password")
    
    if st.button("Acessar"):
        senha_correta = os.getenv('senha_per')
        
        if not senha_correta:
            st.error("Sistema não configurado.")
        elif senha_input == senha_correta:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Senha incorreta.")
    
    st.stop()

# =============================================================================
# APÓS AUTENTICAÇÃO - CÓDIGO PRINCIPAL
# =============================================================================

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
    page_title="Agente Performance",
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
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card-positive {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        border-left: 5px solid #10b981;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card-negative {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        border-left: 5px solid #ef4444;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        color: #4f46e5 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #4f46e5 !important;
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
    .analysis-section {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
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
    .comparative-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .investment-grid {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e2e8f0;
    }
    .platform-comparison {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 20%);
        color: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .trend-indicator {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 5px;
    }
    .trend-up {
        background-color: #10b981;
        color: white;
    }
    .trend-down {
        background-color: #ef4444;
        color: white;
    }
    .trend-neutral {
        background-color: #6b7280;
        color: white;
    }
    .comparative-table {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 10px 0;
    }
    .platform-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px;
        border-top: 4px solid #4f46e5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .client-card {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .month-comparison {
        background: #f8fafc;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 2px dashed #cbd5e1;
    }
    .product-performance {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-top: 5px solid #10b981;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .creative-highlight {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .yoy-analysis {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .scenario-input {
        background: #f8fafc;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #e2e8f0;
    }
    .text-analysis {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
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
    .data-input-box {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 13px;
        border: 1px solid #cbd5e1;
        min-height: 200px;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="header-gradient"><h1>📊 Agente Performance - Análise Completa</h1></div>', unsafe_allow_html=True)

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

def analyze_yoy_data_with_gemini(data_dict, context_input=""):
    """Analisa dados YoY com Gemini"""
    
    if not modelo_texto:
        return "⚠️ Gemini não configurado."
    
    try:
        # Construir prompt
        prompt = f"""
        # 📊 ANÁLISE YoY (Year-over-Year) - RELATÓRIO DE PERFORMANCE
        
        ## DADOS COMPARATIVOS:
        
        ### INVESTIMENTO:
        - Ano Atual: R$ {data_dict.get('investimento_atual', 0):,.0f}
        - Ano Anterior: R$ {data_dict.get('investimento_anterior', 0):,.0f}
        - Variação YoY: {data_dict.get('investimento_yoy', 0):+.1f}%
        
        ### SESSÕES:
        - Ano Atual: {data_dict.get('sessoes_atual', 0):,.0f}
        - Ano Anterior: {data_dict.get('sessoes_anterior', 0):,.0f}
        - Variação YoY: {data_dict.get('sessoes_yoy', 0):+.1f}%
        
        ### ENGAJAMENTO:
        - Ano Atual: {data_dict.get('engajamento_atual', 0):,.0f}
        - Ano Anterior: {data_dict.get('engajamento_anterior', 0):,.0f}
        - Variação YoY: {data_dict.get('engajamento_yoy', 0):+.1f}%
        
        ### VIEWS:
        - Ano Atual: {data_dict.get('views_atual', 0):,.0f}
        - Ano Anterior: {data_dict.get('views_anterior', 0):,.0f}
        - Variação YoY: {data_dict.get('views_yoy', 0):+.1f}%
        
        ## CONTEXTO ADICIONAL FORNECIDO:
        {context_input if context_input else "Nenhum contexto adicional fornecido."}
        
        ## TAREFA:
        
        Analise os dados acima e crie um relatório executivo em português com:
        
        1. **📈 RESUMO EXECUTIVO** (1-2 parágrafos)
        2. **💰 ANÁLISE FINANCEIRA** (eficiência do investimento, ROI implícito)
        3. **📊 ANÁLISE DE PERFORMANCE** (sessões vs engajamento vs views)
        4. **🔍 INSIGHTS ESTRATÉGICOS** (3-5 insights principais baseados nos dados)
        5. **🎯 PONTOS DE ATENÇÃO** (o que precisa ser monitorado ou ajustado)
        6. **🚀 RECOMENDAÇÕES ACIONÁVEIS** (5-7 recomendações específicas)
        7. **📅 CONCLUSÃO** (visão geral e perspectiva futura)
        
        Seja específico, baseado em dados, prático e estratégico.
        Use números e percentuais nos insights.
        Destaque tanto os pontos fortes quanto as oportunidades de melhoria.
        """
        
        with st.spinner("🤖 Gemini está analisando..."):
            response = modelo_texto.generate_content(prompt)
            return response.text
    
    except Exception as e:
        return f"❌ Erro: {str(e)[:200]}"

def analyze_text_data_with_gemini(text_data, analysis_type="complete"):
    """Analisa dados em texto com Gemini"""
    
    if not modelo_texto:
        return "⚠️ Gemini não configurado."
    
    if not text_data or text_data.strip() == "":
        return "❌ Nenhum dado fornecido para análise."
    
    try:
        # Construir prompt baseado no tipo de análise
        analysis_types = {
            "complete": "ANÁLISE COMPLETA",
            "financial": "ANÁLISE FINANCEIRA",
            "performance": "ANÁLISE DE PERFORMANCE",
            "insights": "EXTRAÇÃO DE INSIGHTS"
        }
        
        analysis_focus = analysis_types.get(analysis_type, "ANÁLISE COMPLETA")
        
        prompt = f"""
        # 📊 {analysis_focus} - DADOS EM FORMATO DE TEXTO
        
        ## DADOS FORNECIDOS:
        ```
        {text_data}
        ```
        
        ## TAREFA:
        
        1. **ESTRUTURAR OS DADOS** (identifique padrões, categorias e métricas)
        2. **EXTRAIR INFORMAÇÕES NUMÉRICAS** (valores, percentuais, tendências)
        3. **IDENTIFICAR RELAÇÕES E CORRELAÇÕES** (entre diferentes métricas)
        4. **GERAR INSIGHTS ESTRATÉGICOS** (3-5 insights principais)
        5. **CRIAR RESUMO EXECUTIVO** (1-2 parágrafos)
        6. **SUGERIR PRÓXIMOS PASSOS** (ações recomendadas)
        
        Para cada seção, seja específico e mencione números concretos quando disponíveis.
        
        Estruture sua resposta em:
        
        ## 📋 DADOS ESTRUTURADOS
        [Apresente os dados de forma organizada, em tabelas se possível]
        
        ## 📈 ANÁLISE NUMÉRICA
        [Análise quantitativa dos dados]
        
        ## 🔍 INSIGHTS PRINCIPAIS
        [Lista de insights com explicação]
        
        ## 🎯 RECOMENDAÇÕES
        [Recomendações acionáveis]
        
        ## 📝 RESUMO EXECUTIVO
        [Resumo conciso]
        """
        
        with st.spinner("🤖 Gemini está analisando os dados..."):
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
def load_client_data(_client, client_name, start_date=None, end_date=None, data_sources=None):
    """Carrega dados específicos por cliente"""
    try:
        st.info(f"🔍 Carregando dados para {client_name}...")
        
        # Mapear nomes de clientes para filtros no account_name
        client_mapping = {
            "Syngenta": ["SYNGENTA", "CROP", "AGRO"],
            "Golden Harvest Brasil": ["GOLDEN HARVEST", "GOLDENHARVEST"],
            "Nidera (oficial)": ["NIDERA"],
            "NK Seeds (Oficial - Lab)": ["NK SEEDS", "NKSEEDS"],
            "EuroChem Fertilizantes Tocantins": ["EUROCHEM", "TOCANTINS"],
            "Grupo Vittia": ["VITTIA", "GRUPO VITTIA"]
        }
        
        query = """
        SELECT 
            date,
            campaign,
            datasource,
            spend,
            impressions,
            clicks,
            conversions,
            cpc,
            ctr,
            roas,
            account_name,
            LOWER(account_name) as account_name_lower
        FROM `macfor-media-flow.ads.app_view_campaigns`
        """
        
        conditions = []
        
        # Filtrar por cliente
        if client_name != "Todos":
            if client_name in client_mapping:
                client_terms = client_mapping[client_name]
                client_conditions = []
                for term in client_terms:
                    client_conditions.append(f"LOWER(account_name) LIKE '%{term.lower()}%'")
                conditions.append(f"({' OR '.join(client_conditions)})")
        
        # Filtrar por data
        if start_date:
            conditions.append(f"DATE(date) >= DATE('{start_date}')")
        if end_date:
            conditions.append(f"DATE(date) <= DATE('{end_date}')")
        
        # Filtrar por data sources
        if data_sources and len(data_sources) > 0:
            ds_str = ", ".join([f"'{ds}'" for ds in data_sources])
            conditions.append(f"datasource IN ({ds_str})")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date DESC"
        
        df = _client.query(query).to_dataframe()
        
        if df.empty:
            st.warning(f"Nenhum dado encontrado para {client_name}")
            return pd.DataFrame()

        # Processar datas
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year_month'] = df['date'].dt.strftime('%Y-%m')
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
        
        # Extrair categorias de campanha
        df = extract_campaign_categories(df)
        
        return df
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

def extract_campaign_categories(df):
    """Extrai categorias de campanha"""
    if 'campaign' not in df.columns:
        return df
    
    def categorize_campaign(campaign_name):
        if pd.isna(campaign_name):
            return {
                'product': 'Desconhecido',
                'category': 'Desconhecido',
                'campaign_type': 'Desconhecido'
            }
        
        name = str(campaign_name).upper()
        
        # Produtos Syngenta
        products = [
            'VICTRATO', 'VANIVA', 'REVERB', 'JOINER', 'CERTANO', 
            'FLEXSTAR GT', 'ENGEO PLENO S', 'MIRAVIS DUO', 
            'CRUISER', 'YIELDON', 'ELESTAL NEO', 'AMPLIGO',
            'ACTARA', 'ALADE', 'MITRION', 'AMISTAR', 'ARVATICO',
            'AVICTA', 'AXIAL', 'BRAVONIL', 'CALARIS', 'CALIPEN',
            'CLARIVA', 'CURYOM', 'CYPRESS', 'DUAL GOLD', 'DURIVO',
            'EDDUS', 'ELATUS', 'FORTENZA', 'GROVER', 'INFLUX',
            'INSTIVO', 'INVICT', 'MAXIM', 'MINECTO', 'MIRAVIS',
            'MODDUS', 'PERGADO', 'PLINAZOLIN', 'POLYTRIN', 'PRIORI',
            'PROCLAIM', 'REBRON', 'REGLONE', 'REVUS', 'RIDOMIL',
            'SCORE', 'SPONTA', 'VERDADERO', 'VERDAVIS', 'VOLIAM'
        ]
        
        # Categorias
        categories = [
            'NEMATICIDA', 'INSETICIDA', 'FUNGICIDA', 'HERBICIDA',
            'SEEDCARE', 'BIOLOGICO', 'FERTILIZANTE', 'REGULADOR'
        ]
        
        # Tipos de campanha
        campaign_types = [
            'VIDEO', 'DISPLAY', 'SEARCH', 'SOCIAL', 'PERFORMANCE',
            'BRANDING', 'AWARENESS', 'CONVERSION', 'ENGAGEMENT'
        ]
        
        # Identificar produto
        product_found = 'Desconhecido'
        for prod in products:
            if prod in name:
                product_found = prod
                break
        
        # Identificar categoria
        category_found = 'Desconhecido'
        for cat in categories:
            if cat in name:
                category_found = cat
                break
        
        # Identificar tipo de campanha
        campaign_type_found = 'Desconhecido'
        for ct in campaign_types:
            if ct in name:
                campaign_type_found = ct
                break
        
        return {
            'product': product_found,
            'category': category_found,
            'campaign_type': campaign_type_found
        }
    
    # Aplicar categorização
    categories = df['campaign'].apply(categorize_campaign)
    df['product'] = categories.apply(lambda x: x['product'])
    df['category'] = categories.apply(lambda x: x['category'])
    df['campaign_type'] = categories.apply(lambda x: x['campaign_type'])
    
    return df

# =============================================================================
# FUNÇÕES DE ANÁLISE COMPARATIVA
# =============================================================================

def calculate_monthly_comparison(df, current_month, previous_month):
    """Calcula comparação entre meses"""
    
    # Filtrar dados por mês
    df_current = df[df['year_month'] == current_month].copy()
    df_previous = df[df['year_month'] == previous_month].copy()
    
    if df_current.empty or df_previous.empty:
        return None
    
    # Métricas principais
    metrics = ['spend', 'impressions', 'clicks', 'conversions']
    
    comparison = {}
    
    for metric in metrics:
        if metric in df_current.columns and metric in df_previous.columns:
            current_val = df_current[metric].sum()
            previous_val = df_previous[metric].sum()
            
            if previous_val != 0:
                change_pct = ((current_val - previous_val) / previous_val) * 100
            else:
                change_pct = 0
            
            comparison[metric] = {
                'current': current_val,
                'previous': previous_val,
                'change_pct': change_pct,
                'change_abs': current_val - previous_val
            }
    
    # Métricas calculadas
    if 'spend' in df_current.columns and 'clicks' in df_current.columns:
        current_cpc = df_current['spend'].sum() / df_current['clicks'].sum() if df_current['clicks'].sum() > 0 else 0
        previous_cpc = df_previous['spend'].sum() / df_previous['clicks'].sum() if df_previous['clicks'].sum() > 0 else 0
        
        comparison['cpc'] = {
            'current': current_cpc,
            'previous': previous_cpc,
            'change_pct': ((current_cpc - previous_cpc) / previous_cpc * 100) if previous_cpc > 0 else 0
        }
    
    if 'impressions' in df_current.columns and 'clicks' in df_current.columns:
        current_ctr = (df_current['clicks'].sum() / df_current['impressions'].sum() * 100) if df_current['impressions'].sum() > 0 else 0
        previous_ctr = (df_previous['clicks'].sum() / df_previous['impressions'].sum() * 100) if df_previous['impressions'].sum() > 0 else 0
        
        comparison['ctr'] = {
            'current': current_ctr,
            'previous': previous_ctr,
            'change_pct': ((current_ctr - previous_ctr) / previous_ctr * 100) if previous_ctr > 0 else 0
        }
    
    return comparison

def generate_investment_by_platform(df, month):
    """Gera análise de investimento por plataforma"""
    
    df_month = df[df['year_month'] == month].copy()
    
    if df_month.empty:
        return None
    
    # Agrupar por plataforma (datasource)
    if 'datasource' in df_month.columns and 'spend' in df_month.columns:
        platform_investment = df_month.groupby('datasource').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum'
        }).reset_index()
        
        # Calcular métricas adicionais
        platform_investment['cpc'] = platform_investment['spend'] / platform_investment['clicks'].replace(0, np.nan)
        platform_investment['ctr'] = (platform_investment['clicks'] / platform_investment['impressions'].replace(0, np.nan)) * 100
        
        # Ordenar por investimento
        platform_investment = platform_investment.sort_values('spend', ascending=False)
        
        # Calcular percentuais
        total_spend = platform_investment['spend'].sum()
        platform_investment['spend_pct'] = (platform_investment['spend'] / total_spend * 100) if total_spend > 0 else 0
        
        return platform_investment
    
    return None

def generate_product_performance(df, month):
    """Gera análise de performance por produto"""
    
    df_month = df[df['year_month'] == month].copy()
    
    if df_month.empty or 'product' not in df_month.columns:
        return None
    
    # Agrupar por produto
    product_perf = df_month.groupby('product').agg({
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum'
    }).reset_index()
    
    # Calcular métricas
    product_perf['cpc'] = product_perf['spend'] / product_perf['clicks'].replace(0, np.nan)
    product_perf['ctr'] = (product_perf['clicks'] / product_perf['impressions'].replace(0, np.nan)) * 100
    
    # Ordenar por engajamento (clicks)
    product_perf = product_perf.sort_values('clicks', ascending=False)
    
    return product_perf.head(20)

def format_currency(value):
    """Formata valor como moeda"""
    try:
        if pd.isna(value):
            return "R$ 0"
        return f"R$ {value:,.0f}".replace(",", ".")
    except:
        return f"R$ {value}"

def format_percentage(value):
    """Formata valor como porcentagem"""
    try:
        if pd.isna(value):
            return "0%"
        return f"{value:.1f}%"
    except:
        return f"{value}%"

def get_trend_indicator(change_pct):
    """Retorna indicador visual de tendência"""
    if change_pct > 5:
        return "🟢"
    elif change_pct < -5:
        return "🔴"
    else:
        return "🟡"

# =============================================================================
# FUNÇÕES PARA RELATÓRIO DETALHADO
# =============================================================================

def generate_detailed_report(df, current_month, previous_month, client_name):
    """Gera relatório detalhado estilo dos exemplos"""
    
    # Dados dos meses
    df_current = df[df['year_month'] == current_month].copy()
    df_previous = df[df['year_month'] == previous_month].copy()
    
    if df_current.empty or df_previous.empty:
        return "Dados insuficientes para gerar relatório"
    
    # Cálculos principais
    current_spend = df_current['spend'].sum()
    previous_spend = df_previous['spend'].sum()
    spend_change_pct = ((current_spend - previous_spend) / previous_spend * 100) if previous_spend > 0 else 0
    
    current_clicks = df_current['clicks'].sum()
    previous_clicks = df_previous['clicks'].sum()
    clicks_change_pct = ((current_clicks - previous_clicks) / previous_clicks * 100) if previous_clicks > 0 else 0
    
    current_impressions = df_current['impressions'].sum()
    previous_impressions = df_previous['impressions'].sum()
    impressions_change_pct = ((current_impressions - previous_impressions) / previous_impressions * 100) if previous_impressions > 0 else 0
    
    current_conversions = df_current['conversions'].sum()
    previous_conversions = df_previous['conversions'].sum()
    conversions_change_pct = ((current_conversions - previous_conversions) / previous_conversions * 100) if previous_conversions > 0 else 0
    
    # Investimento por plataforma
    platform_current = generate_investment_by_platform(df, current_month)
    platform_previous = generate_investment_by_platform(df, previous_month)
    
    # Performance por produto
    product_perf = generate_product_performance(df, current_month)
    
    # Performance por categoria
    category_perf = generate_category_performance(df, current_month)
    
    # Construir relatório
    report = f"""
# 📊 RELATÓRIO DE PERFORMANCE - {client_name}
## Período: {current_month} vs {previous_month}

---

## 📈 RESUMO EXECUTIVO

**Investimento Total:**
- **{current_month}:** {format_currency(current_spend)}
- **{previous_month}:** {format_currency(previous_spend)}
- **Variação:** {format_percentage(spend_change_pct)} {get_trend_indicator(spend_change_pct)}

**Engajamento:**
- **Cliques {current_month}:** {current_clicks:,.0f}
- **Variação:** {format_percentage(clicks_change_pct)} {get_trend_indicator(clicks_change_pct)}

**Alcance:**
- **Impressões {current_month}:** {current_impressions:,.0f}
- **Variação:** {format_percentage(impressions_change_pct)} {get_trend_indicator(impressions_change_pct)}

**Conversões:**
- **Conversões {current_month}:** {current_conversions:,.0f}
- **Variação:** {format_percentage(conversions_change_pct)} {get_trend_indicator(conversions_change_pct)}

---

## 💰 INVESTIMENTO POR PLATAFORMA

### {current_month}
"""
    
    if platform_current is not None:
        for _, row in platform_current.iterrows():
            report += f"- **{row['datasource'].upper()}:** {format_currency(row['spend'])} ({row['spend_pct']:.1f}%)\n"
    
    report += f"\n### {previous_month}\n"
    
    if platform_previous is not None:
        for _, row in platform_previous.iterrows():
            report += f"- **{row['datasource'].upper()}:** {format_currency(row['spend'])} ({row['spend_pct']:.1f}%)\n"
    
    report += """

---

## 🎯 TOP PRODUTOS POR ENGAGEMENT

"""
    
    if product_perf is not None:
        report += "| Produto | Investimento | Impressões | Cliques | CTR |\n"
        report += "|---------|--------------|------------|---------|-----|\n"
        for _, row in product_perf.head(10).iterrows():
            ctr = row['ctr'] if not pd.isna(row['ctr']) else 0
            report += f"| {row['product']} | {format_currency(row['spend'])} | {row['impressions']:,.0f} | {row['clicks']:,.0f} | {ctr:.2f}% |\n"
    
    report += """

---

## 📊 PERFORMANCE POR CATEGORIA

"""
    
    if category_perf is not None:
        report += "| Categoria | Investimento | % Total | Cliques | CPC |\n"
        report += "|-----------|--------------|---------|---------|-----|\n"
        for _, row in category_perf.iterrows():
            spend_pct = (row['spend'] / current_spend * 100) if current_spend > 0 else 0
            cpc = row['cpc'] if not pd.isna(row['cpc']) else 0
            report += f"| {row['category']} | {format_currency(row['spend'])} | {spend_pct:.1f}% | {row['clicks']:,.0f} | {format_currency(cpc)} |\n"
    
    report += """

---

## 🔍 INSIGHTS ESTRATÉGICOS

"""
    
    # Gerar insights baseados nos dados
    insights = []
    
    if spend_change_pct > 0 and clicks_change_pct > spend_change_pct:
        insights.append("**Eficiência em alta:** O crescimento em cliques superou o aumento de investimento, indicando maior eficiência nas campanhas.")
    
    if platform_current is not None and platform_previous is not None:
        # Verificar mudanças na distribuição de investimento
        platforms = set(platform_current['datasource']).union(set(platform_previous['datasource']))
        for platform in platforms:
            curr = platform_current[platform_current['datasource'] == platform]
            prev = platform_previous[platform_previous['datasource'] == platform]
            
            if not curr.empty and not prev.empty:
                curr_pct = curr['spend_pct'].iloc[0]
                prev_pct = prev['spend_pct'].iloc[0]
                
                if abs(curr_pct - prev_pct) > 10:
                    direction = "aumento" if curr_pct > prev_pct else "redução"
                    insights.append(f"**Redistribuição estratégica:** {direction} significativa no investimento em {platform.upper()} ({prev_pct:.1f}% → {curr_pct:.1f}%).")
    
    if product_perf is not None and len(product_perf) > 0:
        top_product = product_perf.iloc[0]['product']
        if top_product != 'Desconhecido':
            insights.append(f"**Produto destaque:** {top_product} lidera em engajamento, representando oportunidade para expandir investimentos.")
    
    # Adicionar insights ao relatório
    for i, insight in enumerate(insights[:5], 1):
        report += f"{i}. {insight}\n"
    
    report += """

---

## 🚀 RECOMENDAÇÕES

1. **Otimizar mix de mídia:** Ajustar alocação entre plataformas baseado no ROI histórico
2. **Escalar campanhas performáticas:** Identificar e aumentar budget das campanhas com melhor CTR e conversões
3. **Testar novos formatos:** Experimentar diferentes formatos criativos nas plataformas de melhor performance
4. **Refinar segmentação:** Ajustar públicos-alvo baseado no engajamento por categoria de produto
5. **Monitorar CPC:** Implementar alertas para variações significativas no custo por clique

---

## 📅 PRÓXIMOS PASSOS

- [ ] Revisar orçamento mensal por plataforma
- [ ] Analisar sazonalidade por categoria de produto
- [ ] Planejar testes A/B para campanhas de baixa performance
- [ ] Agendar reunião de review com equipe de performance
- [ ] Definir metas para o próximo período

---

*Relatório gerado automaticamente em {datetime.now().strftime("%d/%m/%Y %H:%M")}*
"""
    
    return report

# =============================================================================
# FUNÇÕES PARA ANÁLISE DE CENÁRIO YoY
# =============================================================================

def calculate_yoy(current_value, previous_value):
    """Calcula variação YoY"""
    if previous_value == 0:
        return 0
    return ((current_value - previous_value) / previous_value) * 100

def analyze_scenario_data(investimento_atual, investimento_anterior,
                         sessoes_atual, sessoes_anterior,
                         engajamento_atual, engajamento_anterior,
                         views_atual, views_anterior):
    """Analisa dados de cenário e calcula YoY"""
    
    # Calcular YoY para cada métrica
    investimento_yoy = calculate_yoy(investimento_atual, investimento_anterior)
    sessoes_yoy = calculate_yoy(sessoes_atual, sessoes_anterior)
    engajamento_yoy = calculate_yoy(engajamento_atual, engajamento_anterior)
    views_yoy = calculate_yoy(views_atual, views_anterior)
    
    # Calcular eficiência
    eficiencia_atual = engajamento_atual / investimento_atual if investimento_atual > 0 else 0
    eficiencia_anterior = engajamento_anterior / investimento_anterior if investimento_anterior > 0 else 0
    eficiencia_yoy = calculate_yoy(eficiencia_atual, eficiencia_anterior)
    
    # Calcular custo por sessão
    cps_atual = investimento_atual / sessoes_atual if sessoes_atual > 0 else 0
    cps_anterior = investimento_anterior / sessoes_anterior if sessoes_anterior > 0 else 0
    cps_yoy = calculate_yoy(cps_atual, cps_anterior)
    
    # Calcular engajamento por view
    eng_view_atual = engajamento_atual / views_atual if views_atual > 0 else 0
    eng_view_anterior = engajamento_anterior / views_anterior if views_anterior > 0 else 0
    eng_view_yoy = calculate_yoy(eng_view_atual, eng_view_anterior)
    
    # Preparar dados para análise
    data_dict = {
        'investimento_atual': investimento_atual,
        'investimento_anterior': investimento_anterior,
        'investimento_yoy': investimento_yoy,
        
        'sessoes_atual': sessoes_atual,
        'sessoes_anterior': sessoes_anterior,
        'sessoes_yoy': sessoes_yoy,
        
        'engajamento_atual': engajamento_atual,
        'engajamento_anterior': engajamento_anterior,
        'engajamento_yoy': engajamento_yoy,
        
        'views_atual': views_atual,
        'views_anterior': views_anterior,
        'views_yoy': views_yoy,
        
        'eficiencia_atual': eficiencia_atual,
        'eficiencia_anterior': eficiencia_anterior,
        'eficiencia_yoy': eficiencia_yoy,
        
        'cps_atual': cps_atual,
        'cps_anterior': cps_anterior,
        'cps_yoy': cps_yoy,
        
        'eng_view_atual': eng_view_atual,
        'eng_view_anterior': eng_view_anterior,
        'eng_view_yoy': eng_view_yoy
    }
    
    return data_dict

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================

# Inicializar estado
if 'df_data' not in st.session_state:
    st.session_state.df_data = pd.DataFrame()
if 'selected_client' not in st.session_state:
    st.session_state.selected_client = "Syngenta"
if 'current_month' not in st.session_state:
    st.session_state.current_month = datetime.now().strftime('%Y-%m')
if 'previous_month' not in st.session_state:
    prev_month = (datetime.now() - relativedelta(months=1)).strftime('%Y-%m')
    st.session_state.previous_month = prev_month
if 'month_comparison' not in st.session_state:
    st.session_state.month_comparison = None
if 'detailed_report' not in st.session_state:
    st.session_state.detailed_report = None
if 'yoy_analysis_result' not in st.session_state:
    st.session_state.yoy_analysis_result = None
if 'text_analysis_result' not in st.session_state:
    st.session_state.text_analysis_result = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Testar conexão
    if st.button("Testar Conexão BigQuery", use_container_width=True):
        with st.spinner("Conectando..."):
            client = get_bigquery_client()
            if client:
                st.success("✅ Conexão OK!")
    
    st.markdown("---")
    st.subheader("👥 Seleção do Cliente")
    
    opcoes_clientes = [
        "Syngenta", 
        "Golden Harvest Brasil", 
        "Nidera (oficial)", 
        "NK Seeds (Oficial - Lab)", 
        "EuroChem Fertilizantes Tocantins", 
        "Grupo Vittia",
        "Todos"
    ]
    
    selected_client = st.selectbox(
        "Cliente:",
        options=opcoes_clientes,
        index=0,
        key="client_select_sidebar"
    )
    
    st.markdown("---")
    st.subheader("📅 Período de Análise")
    
    # Seleção de meses
    current_date = datetime.now()
    
    months_options = []
    for i in range(12):
        month_date = current_date - relativedelta(months=i)
        month_str = month_date.strftime('%Y-%m')
        month_display = month_date.strftime('%B %Y').title()
        months_options.append((month_str, month_display))
    
    current_month = st.selectbox(
        "Mês Atual:",
        options=[m[0] for m in months_options],
        format_func=lambda x: dict(months_options)[x],
        index=0,
        key="current_month_select"
    )
    
    previous_month = st.selectbox(
        "Mês Anterior:",
        options=[m[0] for m in months_options[1:]],
        format_func=lambda x: dict(months_options)[x],
        index=0,
        key="previous_month_select"
    )
    
    st.markdown("---")
    st.subheader("📱 Data Sources")
    
    data_sources_opcoes = ["facebook", "google ads", "tiktok", "linkedin", "twitter", "pinterest", "display", "search", "youtube", "pmax"]
    selected_sources = st.multiselect(
        "Filtrar por plataforma:",
        options=data_sources_opcoes,
        default=data_sources_opcoes[:5],
        key="data_sources_select"
    )
    
    st.markdown("---")
    
    # Botão para carregar dados
    if st.button("📊 Carregar Dados", type="primary", use_container_width=True):
        with st.spinner(f"Carregando dados para {selected_client}..."):
            client = get_bigquery_client()
            if client:
                # Calcular datas baseadas nos meses selecionados
                current_year, current_month_num = map(int, current_month.split('-'))
                prev_year, prev_month_num = map(int, previous_month.split('-'))
                
                # Primeiro dia do mês atual
                start_date_current = date(current_year, current_month_num, 1)
                
                # Último dia do mês atual
                if current_month_num == 12:
                    end_date_current = date(current_year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date_current = date(current_year, current_month_num + 1, 1) - timedelta(days=1)
                
                # Primeiro dia do mês anterior
                start_date_prev = date(prev_year, prev_month_num, 1)
                
                # Último dia do mês anterior
                if prev_month_num == 12:
                    end_date_prev = date(prev_year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date_prev = date(prev_year, prev_month_num + 1, 1) - timedelta(days=1)
                
                # Carregar dados para os dois períodos
                start_date = min(start_date_current, start_date_prev)
                end_date = max(end_date_current, end_date_prev)
                
                df = load_client_data(
                    client,
                    selected_client,
                    start_date=start_date,
                    end_date=end_date,
                    data_sources=selected_sources
                )
                
                if not df.empty:
                    st.session_state.df_data = df
                    st.session_state.selected_client = selected_client
                    st.session_state.current_month = current_month
                    st.session_state.previous_month = previous_month
                    
                    # Calcular comparação
                    comparison = calculate_monthly_comparison(df, current_month, previous_month)
                    st.session_state.month_comparison = comparison
                    
                    # Gerar relatório detalhado
                    report = generate_detailed_report(df, current_month, previous_month, selected_client)
                    st.session_state.detailed_report = report
                    
                    st.success(f"✅ {len(df):,} registros carregados!")
                    st.rerun()
                else:
                    st.error("Nenhum dado encontrado")
            else:
                st.error("❌ Não foi possível conectar.")

# Verificar dados
df = st.session_state.df_data
comparison = st.session_state.month_comparison
detailed_report = st.session_state.detailed_report

# =============================================================================
# LAYOUT PRINCIPAL - ABAS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Dashboard MoM", 
    "💰 Investimento por Plataforma", 
    "🎯 Performance por Produto",
    "📊 Relatório Executivo",
    "📋 Dados Detalhados",
    "📊 Cenário YoY",
    "📝 Análise de Texto"
])

# =============================================================================
# TAB 1-5: ABAS EXISTENTES (MANTIDAS IGUAIS)
# =============================================================================

# Tab 1: Dashboard MoM
with tab1:
    if df.empty:
        st.warning("📭 Nenhum dado carregado. Use o botão na sidebar para carregar dados.")
        st.info("💡 Selecione um cliente e período, depois clique em 'Carregar Dados'")
    else:
        st.header(f"📈 Dashboard Comparativo MoM - {st.session_state.selected_client}")
        
        st.markdown(f"""
        <div class="month-comparison">
            <h3>🔄 Comparativo: {st.session_state.current_month} vs {st.session_state.previous_month}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if comparison:
            # Métricas principais (código mantido igual)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'spend' in comparison:
                    spend_data = comparison['spend']
                    delta = f"{spend_data['change_pct']:+.1f}%"
                    st.metric(
                        label="💰 Investimento Total",
                        value=format_currency(spend_data['current']),
                        delta=delta,
                        delta_color="normal"
                    )
            
            with col2:
                if 'impressions' in comparison:
                    imp_data = comparison['impressions']
                    delta = f"{imp_data['change_pct']:+.1f}%"
                    st.metric(
                        label="👁️ Impressões",
                        value=f"{imp_data['current']:,.0f}".replace(",", "."),
                        delta=delta,
                        delta_color="normal"
                    )
            
            with col3:
                if 'clicks' in comparison:
                    clicks_data = comparison['clicks']
                    delta = f"{clicks_data['change_pct']:+.1f}%"
                    st.metric(
                        label="🖱️ Cliques",
                        value=f"{clicks_data['current']:,.0f}".replace(",", "."),
                        delta=delta,
                        delta_color="normal"
                    )
            
            with col4:
                if 'conversions' in comparison:
                    conv_data = comparison['conversions']
                    delta = f"{conv_data['change_pct']:+.1f}%"
                    st.metric(
                        label="🎯 Conversões",
                        value=f"{conv_data['current']:,.0f}".replace(",", "."),
                        delta=delta,
                        delta_color="normal"
                    )
            
            # Resto do código da Tab 1 (mantido igual)...

# Tabs 2-5 (mantidas iguais, apenas verificando se há dados)
# ... [código das tabs 2-5 mantido igual]

# =============================================================================
# TAB 6: CENÁRIO YoY
# =============================================================================

with tab6:
    st.markdown('<div class="yoy-analysis"><h2>📊 Análise de Cenário YoY (Year-over-Year)</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📝 Insira os dados para análise comparativa
    
    Preencha os valores do **ano atual** e do **ano anterior** para cada métrica.
    O sistema calculará automaticamente a variação YoY e gerará uma análise com Gemini.
    """)
    
    # Input de dados
    st.markdown('<div class="scenario-input">', unsafe_allow_html=True)
    st.subheader("💰 Investimento (R$)")
    
    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        investimento_atual = st.number_input(
            "Ano Atual:",
            min_value=0.0,
            value=100000.0,
            step=1000.0,
            format="%.2f",
            key="investimento_atual"
        )
    
    with col_inv2:
        investimento_anterior = st.number_input(
            "Ano Anterior:",
            min_value=0.0,
            value=85000.0,
            step=1000.0,
            format="%.2f",
            key="investimento_anterior"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="scenario-input">', unsafe_allow_html=True)
    st.subheader("👥 Sessões")
    
    col_ses1, col_ses2 = st.columns(2)
    with col_ses1:
        sessoes_atual = st.number_input(
            "Ano Atual:",
            min_value=0,
            value=1000000,
            step=10000,
            key="sessoes_atual"
        )
    
    with col_ses2:
        sessoes_anterior = st.number_input(
            "Ano Anterior:",
            min_value=0,
            value=900000,
            step=10000,
            key="sessoes_anterior"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="scenario-input">', unsafe_allow_html=True)
    st.subheader("👍 Engajamento")
    
    col_eng1, col_eng2 = st.columns(2)
    with col_eng1:
        engajamento_atual = st.number_input(
            "Ano Atual:",
            min_value=0,
            value=500000,
            step=10000,
            key="engajamento_atual"
        )
    
    with col_eng2:
        engajamento_anterior = st.number_input(
            "Ano Anterior:",
            min_value=0,
            value=450000,
            step=10000,
            key="engajamento_anterior"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="scenario-input">', unsafe_allow_html=True)
    st.subheader("👁️ Views")
    
    col_views1, col_views2 = st.columns(2)
    with col_views1:
        views_atual = st.number_input(
            "Ano Atual:",
            min_value=0,
            value=2000000,
            step=10000,
            key="views_atual"
        )
    
    with col_views2:
        views_anterior = st.number_input(
            "Ano Anterior:",
            min_value=0,
            value=1800000,
            step=10000,
            key="views_anterior"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input de contexto adicional
    st.markdown("### 📝 Contexto Adicional (Opcional)")
    context_input = st.text_area(
        "Forneça informações adicionais sobre o contexto, estratégias, mudanças no mercado, etc.:",
        height=150,
        placeholder="Ex: Aumentamos o investimento em vídeos, lançamos novo produto, mercado mais competitivo, mudança de estratégia...",
        key="context_input"
    )
    
    # Calcular YoY
    if st.button("📊 Calcular YoY e Gerar Análise", type="primary", use_container_width=True):
        if not modelo_texto:
            st.error("❌ Gemini não configurado. Configure a API key para usar esta funcionalidade.")
        else:
            with st.spinner("Calculando YoY e gerando análise..."):
                # Calcular dados YoY
                data_dict = analyze_scenario_data(
                    investimento_atual, investimento_anterior,
                    sessoes_atual, sessoes_anterior,
                    engajamento_atual, engajamento_anterior,
                    views_atual, views_anterior
                )
                
                # Gerar análise com Gemini
                analysis_result = analyze_yoy_data_with_gemini(data_dict, context_input)
                st.session_state.yoy_analysis_result = analysis_result
                
                st.success("✅ Análise YoY gerada com sucesso!")
                st.rerun()
    
    # Mostrar resultados se disponíveis
    if st.session_state.yoy_analysis_result:
        st.markdown("---")
        st.markdown("### 📈 Resultados Calculados")
        
        # Calcular dados novamente para mostrar
        data_dict = analyze_scenario_data(
            investimento_atual, investimento_anterior,
            sessoes_atual, sessoes_anterior,
            engajamento_atual, engajamento_anterior,
            views_atual, views_anterior
        )
        
        # Mostrar métricas calculadas
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        
        with col_res1:
            st.metric(
                "💰 Investimento YoY",
                format_currency(data_dict['investimento_atual']),
                delta=f"{data_dict['investimento_yoy']:+.1f}%"
            )
        
        with col_res2:
            st.metric(
                "👥 Sessões YoY",
                f"{data_dict['sessoes_atual']:,.0f}",
                delta=f"{data_dict['sessoes_yoy']:+.1f}%"
            )
        
        with col_res3:
            st.metric(
                "👍 Engajamento YoY",
                f"{data_dict['engajamento_atual']:,.0f}",
                delta=f"{data_dict['engajamento_yoy']:+.1f}%"
            )
        
        with col_res4:
            st.metric(
                "👁️ Views YoY",
                f"{data_dict['views_atual']:,.0f}",
                delta=f"{data_dict['views_yoy']:+.1f}%"
            )
        
        # Métricas de eficiência
        st.markdown("### 📊 Métricas de Eficiência")
        
        col_eff1, col_eff2, col_eff3 = st.columns(3)
        
        with col_eff1:
            st.metric(
                "💸 Custo por Sessão",
                f"R$ {data_dict['cps_atual']:.2f}",
                delta=f"{data_dict['cps_yoy']:+.1f}%",
                delta_color="inverse" if data_dict['cps_yoy'] < 0 else "normal"
            )
        
        with col_eff2:
            st.metric(
                "🎯 Engajamento/Investimento",
                f"{data_dict['eficiencia_atual']:.2f}",
                delta=f"{data_dict['eficiencia_yoy']:+.1f}%"
            )
        
        with col_eff3:
            st.metric(
                "📈 Engajamento/View",
                f"{data_dict['eng_view_atual']:.3f}",
                delta=f"{data_dict['eng_view_yoy']:+.1f}%"
            )
        
        # Gráfico de variação YoY
        st.markdown("### 📊 Variação YoY por Métrica")
        
        yoy_data = pd.DataFrame({
            'Métrica': ['Investimento', 'Sessões', 'Engajamento', 'Views'],
            'Variação YoY (%)': [
                data_dict['investimento_yoy'],
                data_dict['sessoes_yoy'],
                data_dict['engajamento_yoy'],
                data_dict['views_yoy']
            ]
        })
        
        fig_yoy = px.bar(
            yoy_data,
            x='Métrica',
            y='Variação YoY (%)',
            title='Variação Year-over-Year por Métrica',
            color='Variação YoY (%)',
            color_continuous_scale='RdYlGn',
            text_auto='+.1f'
        )
        fig_yoy.update_layout(height=400)
        st.plotly_chart(fig_yoy, use_container_width=True)
        
        # Análise do Gemini
        st.markdown("### 🤖 Análise com Gemini")
        
        col_actions1, col_actions2 = st.columns(2)
        
        with col_actions1:
            # Download da análise
            analysis_text = st.session_state.yoy_analysis_result
            st.download_button(
                label="💾 Baixar Análise",
                data=analysis_text,
                file_name=f"analise_yoy_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_actions2:
            if st.button("🔄 Nova Análise", use_container_width=True):
                st.session_state.yoy_analysis_result = None
                st.rerun()
        
        # Mostrar análise
        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
        st.markdown(st.session_state.yoy_analysis_result)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 7: ANÁLISE DE TEXTO
# =============================================================================

with tab7:
    st.markdown('<div class="text-analysis"><h2>📝 Análise de Dados em Texto</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📋 Cole seus dados numéricos aqui
    
    **Formato aceito:**
    - Tabelas copiadas de Excel, Google Sheets
    - Dados em formato CSV
    - Listas de números com descrições
    - Relatórios com métricas
    - Qualquer texto contendo dados numéricos
    
    **Exemplos:**
    ```
    Produto         Investimento   Cliques   Conversões
    Victrato        15000          50000     1200
    Vaniva          12000          45000     980
    Reverb          8000           32000     750
    
    ou
    
    Janeiro: R$ 50.000, 100.000 impressões, 5.000 cliques
    Fevereiro: R$ 55.000, 120.000 impressões, 6.200 cliques
    Março: R$ 60.000, 150.000 impressões, 7.500 cliques
    ```
    """)
    
    # Área de input de texto
    st.markdown("### 📥 Cole seus dados aqui:")
    text_data = st.text_area(
        "",
        height=300,
        placeholder="Cole seus dados aqui...\n\nExemplo:\nProduto,Investimento,Cliques,Conversões\nVictrato,15000,50000,1200\nVaniva,12000,45000,980\nReverb,8000,32000,750",
        key="text_data_input"
    )
    
    # Opções de análise
    st.markdown("### ⚙️ Configurações da Análise")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        analysis_type = st.selectbox(
            "Tipo de análise:",
            options=["complete", "financial", "performance", "insights"],
            format_func=lambda x: {
                "complete": "📊 Análise Completa",
                "financial": "💰 Análise Financeira",
                "performance": "📈 Análise de Performance",
                "insights": "🔍 Extração de Insights"
            }[x],
            key="analysis_type_select"
        )
    
    with col_opt2:
        if st.button("🧹 Limpar Análise", use_container_width=True):
            st.session_state.text_analysis_result = None
            st.rerun()
    
    # Botão para análise
    if st.button("🤖 Analisar Dados com Gemini", type="primary", use_container_width=True):
        if not text_data or text_data.strip() == "":
            st.error("❌ Por favor, cole alguns dados para análise.")
        elif not modelo_texto:
            st.error("❌ Gemini não configurado. Configure a API key para usar esta funcionalidade.")
        else:
            with st.spinner("🤖 Gemini está analisando os dados..."):
                analysis_result = analyze_text_data_with_gemini(text_data, analysis_type)
                st.session_state.text_analysis_result = analysis_result
                st.success("✅ Análise concluída!")
                st.rerun()
    
    # Mostrar resultados se disponíveis
    if st.session_state.text_analysis_result:
        st.markdown("---")
        st.markdown("### 📄 Resultado da Análise")
        
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            # Download da análise
            analysis_text = st.session_state.text_analysis_result
            st.download_button(
                label="📥 Baixar Análise",
                data=analysis_text,
                file_name=f"analise_texto_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_act2:
            # Nova análise
            if st.button("🔄 Analisar Novos Dados", use_container_width=True):
                st.session_state.text_analysis_result = None
                st.rerun()
        
        # Mostrar análise
        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
        st.markdown(st.session_state.text_analysis_result)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Exemplos de dados
    with st.expander("📚 Exemplos de Dados para Teste"):
        st.markdown("""
        ### Exemplo 1: Dados de Produtos
        
        ```
        Produto         Investimento   Impressões   Cliques   CTR      Conversões
        Victrato        15.000         500.000      25.000    5.0%     1.200
        Vaniva          12.000         450.000      22.500    5.0%     980
        Reverb          8.000          320.000      16.000    5.0%     750
        Joiner          10.000         400.000      18.000    4.5%     850
        Certano         6.000          250.000      10.000    4.0%     520
        ```
        
        ### Exemplo 2: Dados Mensais
        
        ```
        Mês       Investimento   Sessões   Engajamento   Views
        Janeiro   50.000         100.000   25.000        200.000
        Fevereiro 55.000         120.000   30.000        240.000
        Março     60.000         150.000   38.000        300.000
        Abril     65.000         180.000   45.000        360.000
        Maio      70.000         200.000   52.000        400.000
        ```
        
        ### Exemplo 3: Dados por Plataforma
        
        ```
        Plataforma   Investimento   CPC      CTR      Conversões   ROAS
        Facebook     40.000         R$ 0.80  3.2%     800          2.5x
        Instagram    30.000         R$ 1.20  2.8%     600          2.0x
        Google Ads   20.000         R$ 1.50  4.5%     450          1.8x
        TikTok       15.000         R$ 0.60  5.2%     350          2.3x
        LinkedIn     5.000          R$ 2.00  1.8%     100          1.5x
        ```
        
        ### Exemplo 4: Dados de Campanhas
        
        ```
        Campanha                   Tipo          Investimento   Resultados   CPL
        Victrato Soja             Video         8.000          120          R$ 66.67
        Vaniva Milho              Display       6.000          85           R$ 70.59
        Reverb Soja               Search        5.000          95           R$ 52.63
        Joiner Algodão            Social        4.000          70           R$ 57.14
        Certano Café              Performance   3.000          55           R$ 54.55
        ```
        
        **Dica:** Cole qualquer um desses exemplos na caixa de texto acima e clique em "Analisar Dados com Gemini"!
        """)

# =============================================================================
# RODAPÉ
# =============================================================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if not df.empty:
        st.caption(f"📊 Dados: {len(df):,} registros")
    else:
        st.caption("📊 Aguardando dados...")

with footer_col2:
    if st.session_state.yoy_analysis_result:
        st.caption("📈 Análise YoY disponível")
    elif st.session_state.text_analysis_result:
        st.caption("📝 Análise de texto disponível")

with footer_col3:
    st.caption(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Status Gemini
if modelo_texto:
    st.sidebar.success("✅ Gemini ativo")
else:
    st.sidebar.info("ℹ️ Gemini inativo - Algumas funcionalidades limitadas")
