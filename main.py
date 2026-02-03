# app_completo.py - App Analytics Platform Completo com Classificador Multi-Clientes
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
import re

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
    .campaign-classifier {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .classifier-result {
        background: #d1fae5;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #059669;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .client-filter {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .mom-analysis {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .yoy-scenario {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .pasted-data {
        background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);
        color: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .comparison-table {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    .platform-card {
        background: #f0f9ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        border-left: 4px solid #3b82f6;
    }
    .yoy-metric {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 2px solid #e2e8f0;
        box-shadow: 0 3px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="header-gradient"><h1>📊 Agente Performance</h1></div>', unsafe_allow_html=True)

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
            try:
                if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
                    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
                
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
        
        # Sample data
        try:
            sample_df = df_filtered.head(20).copy()
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

def generate_slides_description(gemini_analysis_report, user_instructions=""):
    """Gera descrição do que colocar em cada slide baseada no relatório Gemini"""
    
    if not modelo_texto:
        return "⚠️ Gemini não configurado."
    
    if not gemini_analysis_report or gemini_analysis_report.startswith("❌") or gemini_analysis_report.startswith("⚠️"):
        return "❌ Nenhuma análise disponível para criar slides."
    
    try:
        prompt = f"""
        # 📊 DESCRIÇÃO PARA SLIDES DE APRESENTAÇÃO
        
        ## RELATÓRIO GEMINI COMPLETO:
        {gemini_analysis_report}
        
        ## INSTRUÇÕES ADICIONAIS:
        {user_instructions if user_instructions else "Baseie-se no relatório para criar uma descrição do que colocar em cada slide."}
        
        ## TAREFA:
        Com base no relatório Gemini acima, crie uma estrutura de relatório em slides (descrição em formato de texto de como cada slide deve vir) de performance de campanha com:

ARQUITETURA DA APRESENTAÇÃO:
1. CAPA: Esta é uma análise da estrutura lógica e do design visual do slide apresentado, focada em sua arquit
2. AGENDA/CONTEXTUALIZAÇÃO: Slide introdutório
3. SLIDES DE DETALHAMENTO: [Descreva padrão identificado]
4. SLIDES ANALÍTICOS: [Descreva padrão identificado]
5. CONCLUSÕES/RECOMENDAÇÕES: Slide final com insights

PADRÕES ESTRUTURAIS IDENTIFICADOS:
- Hierarquia visual: [Descrição da hierarquia]
- Elementos recorrentes: [Lista de elementos]
- Densidade informacional: [Padrão identificado]

DIRETRIZES PARA RELATÓRIO DE PERFORMANCE:
1. Para cada slide, mantenha a estrutura de: [Título claro] + [Dados chave] + [Visualização apropriada] + [Insight breve]
2. Use a progressão lógica: Contexto → Métricas → Análise → Insights → Recomendações
3. Aplique consistência visual em: cores, tipografia, layout de gráficos

Gere uma apresentação completa aplicando ESTA ESTRUTURA ESPECÍFICA a um relatório de performance de campanha digital.
        """
        
        with st.spinner("🤖 Gerando descrição dos slides..."):
            response = modelo_texto.generate_content(prompt)
            return response.text
    
    except Exception as e:
        return f"❌ Erro ao gerar slides: {str(e)[:200]}"

def generate_yoy_analysis(yoy_data, context=""):
    """Gera análise YoY com Gemini"""
    
    if not modelo_texto:
        return "⚠️ Gemini não configurado."
    
    try:
        prompt = f"""
        # 📊 ANÁLISE YoY (Year-over-Year) DE PERFORMANCE
        
        ## 📈 DADOS YoY FORNECIDOS:
        {yoy_data}
        
        ## 📝 CONTEXTO ADICIONAL:
        {context if context else "Nenhum contexto adicional fornecido."}
        
        ## 🎯 TAREFA:
        
        Com base nos dados YoY fornecidos acima, crie uma análise completa em português que inclua:
        
        1. **📊 RESUMO EXECUTIVO** (1-2 parágrafos)
        2. **💰 ANÁLISE DE INVESTIMENTO** (comparativo, eficiência, alocação)
        3. **📈 ANÁLISE DE PERFORMANCE** (sessões, engajamento, views, outras métricas)
        4. **🔍 INSIGHTS PRINCIPAIS** (3-5 insights baseados nos dados)
        5. **🎯 RECOMENDAÇÕES ESTRATÉGICAS** (5-7 recomendações acionáveis)
        6. **📅 PRÓXIMOS PASSOS** (ações sugeridas para o próximo período)
        
        Seja específico, use os números fornecidos e forneça interpretações práticas.
        Foque em insights que ajudem na tomada de decisão.
        """
        
        with st.spinner("🤖 Gemini está analisando os dados YoY..."):
            response = modelo_texto.generate_content(prompt)
            return response.text
    
    except Exception as e:
        return f"❌ Erro na análise YoY: {str(e)[:200]}"

def analyze_pasted_data(pasted_text, analysis_type="overall", context=""):
    """Analisa dados colados com Gemini"""
    
    if not modelo_texto:
        return "⚠️ Gemini não configurado.", None
    
    try:
        # Tentar extrair estrutura dos dados
        prompt_structure = f"""
        Analise o seguinte texto que contém dados numéricos e estrutura-os em formato tabular.
        Identifique cabeçalhos, métricas, valores e períodos.
        
        Dados fornecidos:
        {pasted_text}
        
        Retorne:
        1. Uma descrição da estrutura identificada
        2. Os dados estruturados em formato de tabela (se possível)
        """
        
        with st.spinner("🔍 Estruturando dados..."):
            structure_response = modelo_texto.generate_content(prompt_structure)
        
        # Análise completa
        prompt_analysis = f"""
        # 📊 ANÁLISE DE DADOS NUMÉRICOS
        
        ## 📋 DADOS FORNECIDOS:
        {pasted_text}
        
        ## 🏗️ ESTRUTURA IDENTIFICADA:
        {structure_response.text}
        
        ## 📝 CONTEXTO ADICIONAL:
        {context if context else "Nenhum contexto adicional fornecido."}
        
        ## 🎯 TIPO DE ANÁLISE SOLICITADA:
        {analysis_type}
        
        ## 📊 TAREFA:
        
        Analise os dados fornecidos e:
        
        1. **📈 IDENTIFIQUE AS PRINCIPAIS MÉTRICAS** (quais são as métricas mais importantes)
        2. **📊 CALCULE ESTATÍSTICAS** (totais, médias, variações quando aplicável)
        3. **🔍 EXTRAIA INSIGHTS** (3-5 insights principais dos dados)
        4. **📋 APRESENTE OS DADOS** (formate os dados principais em uma tabela clara)
        5. **🎯 FORNEÇA RECOMENDAÇÕES** (baseadas nos dados analisados)
        
        Seja prático e específico com os números fornecidos.
        """
        
        with st.spinner("🤖 Analisando dados..."):
            analysis_response = modelo_texto.generate_content(prompt_analysis)
        
        return analysis_response.text, structure_response.text
    
    except Exception as e:
        return f"❌ Erro na análise: {str(e)[:200]}", None

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
def load_all_columns_data(_client, data_inicio=None, data_fim=None, data_sources=None, filtro_cliente="Todos", limit=50000):
    """Carrega TODAS as colunas e identifica clientes pelo account_name"""
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
        
        query += f" ORDER BY date DESC"
        
        df = _client.query(query).to_dataframe()
        
        if df.empty:
            st.warning("Nenhum dado encontrado")
            return pd.DataFrame()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        def identificar_cliente_por_account_name(account_name):
            if pd.isna(account_name):
                return "Outros"
            
            account_str = str(account_name).strip()
            account_upper = account_str.upper()
            
            if "SYNGENTA" in account_upper:
                return "Syngenta"
            
            return account_str
        
        if 'account_name' in df.columns:
            df['cliente_identificado'] = df['account_name'].apply(identificar_cliente_por_account_name)
            
            if filtro_cliente != "Todos":
                df = df[df['cliente_identificado'] == filtro_cliente].copy()
        else:
            df['cliente_identificado'] = "Desconhecido"
            
        df = df.head(limit)
        df_classificado = classificar_campanhas_multi_cliente(df)
        
        return df_classificado
    
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
    if df.empty:
        return []
    
    colunas_numericas = []
    
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                colunas_numericas.append(col)
            else:
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
    if df.empty or coluna not in df.columns:
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
    if df.empty or coluna not in df.columns:
        return None
    
    try:
        dados = df[coluna].dropna()
        
        if len(dados) == 0:
            return None
        
        if pd.api.types.is_numeric_dtype(df[coluna]):
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
# FUNÇÕES PARA ANÁLISE MOM E YOY
# =============================================================================

def calculate_mom_analysis(df, cliente, mes_atual, mes_anterior):
    """Calcula análise MoM (Month-over-Month)"""
    
    if df.empty:
        return None
    
    # Filtrar por cliente
    if cliente != "Todos" and 'cliente_identificado' in df.columns:
        df_filtered = df[df['cliente_identificado'] == cliente].copy()
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        return None
    
    # Verificar coluna de data
    if 'date' not in df_filtered.columns:
        return None
    
    # Converter datas
    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
    df_filtered['mes'] = df_filtered['date'].dt.to_period('M')
    
    # Filtrar pelos meses especificados
    df_mes_atual = df_filtered[df_filtered['mes'] == mes_atual]
    df_mes_anterior = df_filtered[df_filtered['mes'] == mes_anterior]
    
    # Análise por plataforma/datasource
    analysis_results = {
        'cliente': cliente,
        'mes_atual': str(mes_atual),
        'mes_anterior': str(mes_anterior),
        'total_mes_atual': len(df_mes_atual),
        'total_mes_anterior': len(df_mes_anterior),
        'platform_analysis': {},
        'metric_analysis': {}
    }
    
    # Análise por plataforma
    if 'datasource' in df_filtered.columns:
        platforms = df_filtered['datasource'].unique()
        
        for platform in platforms:
            platform_current = df_mes_atual[df_mes_atual['datasource'] == platform]
            platform_previous = df_mes_anterior[df_mes_anterior['datasource'] == platform]
            
            # Calcular investimento (spend)
            spend_current = 0
            spend_previous = 0
            
            # Procurar coluna de spend
            spend_cols = [col for col in df_filtered.columns if 'spend' in col.lower() or 'cost' in col.lower()]
            if spend_cols:
                spend_col = spend_cols[0]
                spend_current = pd.to_numeric(platform_current[spend_col], errors='coerce').sum()
                spend_previous = pd.to_numeric(platform_previous[spend_col], errors='coerce').sum()
            
            analysis_results['platform_analysis'][platform] = {
                'spend_current': spend_current,
                'spend_previous': spend_previous,
                'spend_change': spend_current - spend_previous,
                'spend_change_pct': ((spend_current - spend_previous) / spend_previous * 100) if spend_previous > 0 else 0,
                'records_current': len(platform_current),
                'records_previous': len(platform_previous)
            }
    
    # Análise de métricas
    metric_cols = identificar_colunas_numericas(df_filtered)
    priority_metrics = ['spend', 'revenue', 'conversions', 'impressions', 'clicks', 'cpc', 'cpm', 'ctr', 'roas']
    
    for metric_name in priority_metrics:
        for col in metric_cols:
            if metric_name in col.lower():
                metric_current = pd.to_numeric(df_mes_atual[col], errors='coerce').sum()
                metric_previous = pd.to_numeric(df_mes_anterior[col], errors='coerce').sum()
                
                change = metric_current - metric_previous
                change_pct = (change / metric_previous * 100) if metric_previous > 0 else 0
                
                analysis_results['metric_analysis'][col] = {
                    'current': metric_current,
                    'previous': metric_previous,
                    'change': change,
                    'change_pct': change_pct
                }
                break
    
    return analysis_results

def format_currency(value):
    """Formata valor como moeda"""
    try:
        if pd.isna(value):
            return "R$ 0"
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return f"R$ {value}"

def format_percentage(value):
    """Formata valor como porcentagem"""
    try:
        if pd.isna(value):
            return "0.00%"
        return f"{value:.2f}%".replace(".", ",")
    except:
        return f"{value}%"

def create_mom_table(analysis_results):
    """Cria tabela de análise MoM"""
    
    if not analysis_results or 'platform_analysis' not in analysis_results:
        return None
    
    platform_data = analysis_results['platform_analysis']
    
    if not platform_data:
        return None
    
    # Criar DataFrame para tabela
    platforms = list(platform_data.keys())
    
    table_data = {
        'Plataforma': platforms,
        f'Investimento {analysis_results["mes_anterior"]}': [],
        f'Investimento {analysis_results["mes_atual"]}': [],
        'Variação MoM': [],
        'Variação %': [],
        'Registros Anterior': [],
        'Registros Atual': []
    }
    
    total_current = 0
    total_previous = 0
    
    for platform in platforms:
        data = platform_data[platform]
        
        table_data[f'Investimento {analysis_results["mes_anterior"]}'].append(format_currency(data['spend_previous']))
        table_data[f'Investimento {analysis_results["mes_atual"]}'].append(format_currency(data['spend_current']))
        table_data['Variação MoM'].append(format_currency(data['spend_change']))
        table_data['Variação %'].append(format_percentage(data['spend_change_pct']))
        table_data['Registros Anterior'].append(f"{data['records_previous']:,}")
        table_data['Registros Atual'].append(f"{data['records_current']:,}")
        
        total_current += data['spend_current']
        total_previous += data['spend_previous']
    
    # Adicionar linha de total
    table_data['Plataforma'].append('TOTAL')
    table_data[f'Investimento {analysis_results["mes_anterior"]}'].append(format_currency(total_previous))
    table_data[f'Investimento {analysis_results["mes_atual"]}'].append(format_currency(total_current))
    
    total_change = total_current - total_previous
    total_change_pct = (total_change / total_previous * 100) if total_previous > 0 else 0
    
    table_data['Variação MoM'].append(format_currency(total_change))
    table_data['Variação %'].append(format_percentage(total_change_pct))
    table_data['Registros Anterior'].append(f"{analysis_results['total_mes_anterior']:,}")
    table_data['Registros Atual'].append(f"{analysis_results['total_mes_atual']:,}")
    
    df_table = pd.DataFrame(table_data)
    
    return df_table

# =============================================================================
# FUNÇÕES PARA CLASSIFICADOR DE CAMPANHAS MULTI-CLIENTES
# =============================================================================

def extrair_categorias_campanha(nome_campanha):
    """Extrai categorias de campanha usando regex e análise de padrões"""
    if not nome_campanha or pd.isna(nome_campanha):
        return {}
    
    nome_str = str(nome_campanha).upper()
    
    categorias = {
        'iniciativa': None,
        'produto': None,
        'cultura': None,
        'categoria': None,
        'tipo_campanha': None,
        'objetivo': None,
        'etapa_funil': None,
        'editoria': None,
        'po': None,
        'agencia': None,
        'plataforma': None,
        'cliente': None
    }
    
    padroes = {
        'etapa_funil': [
            'UP', 'MID', 'LOWER', 'TOF', 'MOF', 'BOF', 'TOP', 'MIDDLE', 'BOTTOM',
            'AWARENESS', 'CONSIDERATION', 'CONVERSION', 'RETENTION',
            'DESCOBERTA', 'CONSIDERACAO', 'CONVERSAO', 'RETENCAO'
        ],
        
        'tipo_campanha': [
            'VIDEO', 'DISPLAY', 'SEARCH', 'SOCIAL', 'EMAIL', 'SMS', 'PUSH',
            'NATIVO', 'NATIVE', 'PROGRAMATICA', 'PROGRAMMATIC',
            'PERFORMANCE', 'BRANDING', 'BRAND', 'DIRECT', 'DIRECT_RESPONSE'
        ],
        
        'objetivo': [
            'AWARENESS', 'CONSIDERATION', 'CONVERSION', 'LEAD', 'SALES',
            'TRAFFIC', 'ENGAGEMENT', 'INSTALL', 'VIEWS', 'CLICKS',
            'ALCANCE', 'CONVERSAO', 'LEADS', 'VENDAS', 'TRAFEGO',
            'ENGAJAMENTO', 'INSTALACOES', 'VISUALIZACOES', 'CLIQUES'
        ],
        
        'plataforma': [
            'GOOGLE', 'FACEBOOK', 'INSTAGRAM', 'TIKTOK', 'LINKEDIN', 'TWITTER',
            'YOUTUBE', 'PINTEREST', 'SNAPCHAT', 'META', 'TIKTOK', 'BING',
            'DV360', 'TRADEDESK', 'AMAZON', 'APPLE', 'SPOTIFY'
        ],
        
        'agencia': [
            'MACFOR', 'OGILVY', 'PUBLICIS', 'WPP', 'OMNICOM', 'DENTSU',
            'HAVAS', 'IPG', 'ACCENTURE', 'DELOITTE', 'PWC', 'KPMG'
        ],
        
        'cultura': [
            'SOJA', 'MILHO', 'CAFE', 'ALGODAO', 'CANADEACUCAR', 'CANA',
            'TRIGO', 'ARROZ', 'FEIJAO', 'MANDIOCA', 'LARANJA', 'UVA',
            'TOMATE', 'BATATA', 'CEVADA', 'AVEIA', 'GIRASSOL'
        ],
        
        'produto': [
            'ACTARA','ALADE-MITRION',
            'AMISTAR','AMISTAR_TOP',
            'AMPLIGO','ARVATICO',
            'AVICTA_COMPLETO','AXIAL',
            'BRAVONIL','BRAVONIL_TOP',
            'BRAVONIL720','CALARIS',
            'CALARIS_MA','CALIPEN_SC',
            'CLARIVA_SKY','CRUISER_ADVANCED',
            'CRUISER_OPTI',
            'CRUISER_TURBO','CURYOM',
            'CYPRESS','DUAL_GOLD',
            'DURIVO','EDDUS',
            'ELATUS',
            'ELESTAL_NEO',
            'ENGEO_PLENO_S','FORTENZA',
            'FORTENZA_DUO',
            'FORTENZA_ELITE','FORTENZA_VIP_TURBO',
            'GROVER',
            'INFLUX',
            'INSTIVO','INVICT',
            'JOINER',
            'MAXIM_QUATTRO','MINECTO_PRO',
            'MIRAVIS','MIRAVIS_DUO',
            'MIRAVIS_PRO',
            'MODDUS','NEMATOIDES',
            'PERGADO_MZ','PLINAZOLIN',
            'POLYTRIN','PRIORI_TOP',
            'PRIORI_XTRA',
            'PROCLAIM','REBRON',
            'REGLONE',
            'REVUS_OPTI',
            'RIDOMIL_GOLD','SCORE_FLEXI',
            'SPONTA','VERDADERO',
            'VERDAVIS','VOLIAM_FLEXI',
            'VOLIAM_TARGO','CERTANO',
            'RIZOLIQ_LLI','RIZOLIQ_UHC',
            'ALADE',
            'MITRION',
            'POLO_500_SC',
            'ADEPIDYN',
            'ORONDIS_FLEXI','SCORE',
            'ORONDIS_ULTRA','INZAK_ZEON',
            'REVERB','MEGAFOL',
            'AEVO',
            'YIELDON','FRONDEO',
            'FLEXSTAR_GT',
            'ELESTAL','FANTON',
            'JOINER_PRO','INVENCIS',
            'SEEKER','CRESTIVO',
            'VIVA','RIZODERMA',
            'RIZOFOS',
            'SIGNUM','NETURE',
            'MIRAVIS_XTRA','VANIVA',
            'CRUISER_OPTI-CRUISER_ADVANCED',
            'VICTRATO_GOLD',
            'BOUNDARY_EC','VICTRATO'
        ]
    }
    
    po_pattern = r'\bPO[_-]?(\d+)\b'
    po_match = re.search(po_pattern, nome_str, re.IGNORECASE)
    if po_match:
        categorias['po'] = f"PO{po_match.group(1)}"
    
    for agencia in padroes['agencia']:
        if agencia in nome_str:
            categorias['agencia'] = agencia
            break
    
    for plataforma in padroes['plataforma']:
        if plataforma in nome_str:
            categorias['plataforma'] = plataforma
            break
    
    for cultura in padroes['cultura']:
        if cultura in nome_str:
            categorias['cultura'] = cultura
            break
    
    for produto in padroes['produto']:
        if produto in nome_str:
            categorias['produto'] = produto
            break
    
    for tipo in padroes['tipo_campanha']:
        if tipo in nome_str:
            categorias['tipo_campanha'] = tipo
            break
    
    for objetivo in padroes['objetivo']:
        if objetivo in nome_str:
            categorias['objetivo'] = objetivo
            break
    
    for etapa in padroes['etapa_funil']:
        if etapa in nome_str:
            categorias['etapa_funil'] = etapa
            break
    
    separadores = ['_', '-', '|', ' ', '__']
    
    for sep in separadores:
        if sep in nome_str:
            partes = nome_str.split(sep)
            if len(partes) > 0:
                primeira_parte = partes[0]
                if len(primeira_parte) > 3 and primeira_parte not in padroes['plataforma']:
                    categorias['iniciativa'] = primeira_parte
    
    clientes_padroes = {
        'SYNGENTA': ['SYNGENTA', 'CROP', 'AGRO'],
        'BAYER': ['BAYER', 'CROPSCIENCE'],
        'BASF': ['BASF'],
        'CORTEVA': ['CORTEVA', 'PIONEER'],
        'NOVARTIS': ['NOVARTIS'],
        'MONSANTO': ['MONSANTO'],
        'JOHNSON': ['JOHNSON', 'JNJ'],
        'PFIZER': ['PFIZER'],
        'ROCHE': ['ROCHE'],
        'MERCK': ['MERCK'],
        'GLAXOSMITHKLINE': ['GSK', 'GLAXO'],
        'ASTRAZENECA': ['ASTRAZENECA'],
        'SANOFI': ['SANOFI']
    }
    
    for cliente, padroes_cliente in clientes_padroes.items():
        for padrao in padroes_cliente:
            if padrao in nome_str:
                categorias['cliente'] = cliente
                break
        if categorias['cliente']:
            break
    
    return categorias

def classificar_campanhas_multi_cliente(df, coluna_campanha='campaign'):
    """Classifica campanhas para múltiplos clientes"""
    if df.empty or coluna_campanha not in df.columns:
        return df
    
    classificacoes = []
    
    for idx, row in df.iterrows():
        nome_campanha = row[coluna_campanha]
        categorias = extrair_categorias_campanha(nome_campanha)
        
        categorias_preenchidas = sum(1 for v in categorias.values() if v is not None)
        classificado = 'SIM' if categorias_preenchidas >= 3 else 'NÃO'
        
        classificacao = {
            'nome_campanha_original': nome_campanha,
            'classificado': classificado,
            'categorias_identificadas': categorias_preenchidas
        }
        
        for chave, valor in categorias.items():
            classificacao[f'campaign_{chave}'] = valor
        
        classificacoes.append(classificacao)
    
    df_classificado = pd.DataFrame(classificacoes)
    
    if len(df_classificado) == len(df):
        df_resultado = df.copy()
        for col in df_classificado.columns:
            if col != 'nome_campanha_original':
                df_resultado[col] = df_classificado[col]
        
        return df_resultado
    
    return df

def carregar_dicionario_categorias():
    """Carrega dicionário de categorias para sugestões"""
    return {
        'iniciativa': [
            'LANCAMENTO', 'RELANCAMENTO', 'PROMOCAO', 'SAZONAL',
            'EVENTO', 'FEIRA', 'CONGRESSO', 'WORKSHOP',
            'DIA_ESPECIAL', 'NATAL', 'PASCOA', 'BLACKFRIDAY',
            'CYBERMONDAY', 'VERAO', 'INVERNO', 'OUTONO', 'PRIMAVERA'
        ],
        'produto': [
            'PRODUTO_A', 'PRODUTO_B', 'PRODUTO_C', 'PRODUTO_D',
            'LINHA_X', 'LINHA_Y', 'LINHA_Z', 'FAMILIA_A', 'FAMILIA_B'
        ],
        'cultura': [
            'SOJA', 'MILHO', 'CAFE', 'ALGODAO', 'CANA',
            'TRIGO', 'ARROZ', 'FEIJAO', 'FRUTAS', 'HORTALICAS',
            'GRÃOS', 'CEREAIS', 'OLEAGINOSAS'
        ],
        'categoria': [
            'INSETICIDA', 'FUNGICIDA', 'HERBICIDA', 'ADUBO',
            'FERTILIZANTE', 'SEMENTE', 'BIOLOGICO', 'QUIMICO',
            'ORGANICO', 'CONVENCIONAL'
        ],
        'tipo_campanha': [
            'VIDEO', 'DISPLAY', 'SEARCH', 'SOCIAL', 'EMAIL',
            'PERFORMANCE', 'BRANDING', 'DIRECT_RESPONSE',
            'NATIVE', 'PROGRAMMATIC', 'AUDIO', 'TV', 'RADIO'
        ],
        'objetivo': [
            'AWARENESS', 'CONSIDERATION', 'CONVERSION',
            'LEAD_GENERATION', 'SALES', 'TRAFFIC', 'ENGAGEMENT',
            'BRAND_LIFT', 'INSTALLS', 'VIEWS'
        ],
        'etapa_funil': ['TOF', 'MOF', 'BOF', 'UP', 'MID', 'LOWER'],
        'plataforma': [
            'GOOGLE_ADS', 'FACEBOOK', 'INSTAGRAM', 'TIKTOK',
            'LINKEDIN', 'YOUTUBE', 'TWITTER', 'PINTEREST',
            'DV360', 'TRADEDESK', 'AMAZON_DSP'
        ]
    }

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
if 'df_classificado' not in st.session_state:
    st.session_state.df_classificado = pd.DataFrame()
if 'relatorio_classificacao' not in st.session_state:
    st.session_state.relatorio_classificacao = None
if 'filtros_aplicados' not in st.session_state:
    st.session_state.filtros_aplicados = {}
if 'slides_description' not in st.session_state:
    st.session_state.slides_description = None
if 'mom_analysis' not in st.session_state:
    st.session_state.mom_analysis = None
if 'yoy_analysis' not in st.session_state:
    st.session_state.yoy_analysis = None
if 'pasted_data_analysis' not in st.session_state:
    st.session_state.pasted_data_analysis = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    if st.button("Testar Conexão BigQuery"):
        with st.spinner("Conectando..."):
            client = get_bigquery_client()
            if client:
                st.success("✅ Conexão OK!")

    st.subheader("👥 Filtro por Cliente")
    opcoes_clientes = [
        "Todos", 
        "Syngenta", 
        "Golden Harvest Brasil", 
        "Nidera (oficial)", 
        "NK Seeds (Oficial - Lab)", 
        "EuroChem Fertilizantes Tocantins", 
        "Grupo Vittia"
    ]
    
    filtro_cliente = st.selectbox(
        "Selecione o Cliente:",
        options=opcoes_clientes,
        index=0
    )
    
    st.subheader("📱 Data Sources")    
    data_sources_opcoes = ["facebook", "google ads", "tiktok", "linkedin", "twitter", "pinterest"]
    selected_sources = st.multiselect(
        "Data Sources",
        options=data_sources_opcoes,
        default=data_sources_opcoes[:3]
    )
    
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
    
    limite_default = 20000
    if st.session_state.df_completo.empty:
        max_limit = 100000
    else:
        max_limit = min(100000, max(limite_default, len(st.session_state.df_completo)))
    
    limite = st.slider(
        "Limite de registros", 
        1000, 
        max_limit, 
        min(limite_default, max_limit), 
        1000
    )
    
    if st.button("📊 Carregar Dados", use_container_width=True, type="primary"):
        with st.spinner("Carregando..."):
            client = get_bigquery_client()
            if client:
                df = load_all_columns_data(
                    client,
                    data_inicio=data_inicio,
                    data_fim=data_fim,
                    data_sources=selected_sources,
                    filtro_cliente=filtro_cliente,
                    limit=limite
                )
                
                if not df.empty:
                    st.session_state.df_completo = df
                    st.session_state.colunas_numericas = identificar_colunas_numericas(df)
                    
                    df_classificado = classificar_campanhas_multi_cliente(df)
                    st.session_state.df_classificado = df_classificado
                    
                    st.success(f"✅ {len(df):,} registros carregados e classificados")
                    st.session_state.gemini_analysis = None
                    st.session_state.filtros_aplicados = {}
                    st.rerun()
                else:
                    st.error("Nenhum dado encontrado")
            else:
                st.error("❌ Não foi possível conectar.")

# Verificar dados
df = st.session_state.df_completo
colunas_numericas = st.session_state.colunas_numericas
df_classificado = st.session_state.df_classificado

if df.empty:
    st.warning("📭 Nenhum dado carregado. Use o botão na sidebar para carregar dados.")
    st.stop()

# =============================================================================
# SEÇÃO DE FILTROS MULTI-CLIENTE
# =============================================================================

st.markdown("## 🔍 Filtros Avançados por Categoria de Campanha")

filtro_col1, filtro_col2, filtro_col3, filtro_col4 = st.columns(4)

with filtro_col1:
    if 'campaign_produto' in df_classificado.columns:
        todos_produtos = [
            'Todos', 
            'ACTARA',
            'ALADE-MITRION',
            'AMISTAR',
            'AMISTAR_TOP',
            'AMPLIGO',
            'ARVATICO',
            'AVICTA_COMPLETO',
            'AXIAL',
            'BRAVONIL',
            'BRAVONIL_TOP',
            'BRAVONIL720',
            'CALARIS',
            'CALARIS_MA',
            'CALIPEN_SC',
            'CLARIVA_SKY',
            'CRUISER_ADVANCED',
            'CRUISER_OPTI',
            'CRUISER_TURBO',
            'CURYOM',
            'CYPRESS',
            'DUAL_GOLD',
            'DURIVO',
            'EDDUS',
            'ELATUS',
            'ELESTAL_NEO',
            'ENGEO_PLENO_S',
            'FORTENZA',
            'FORTENZA_DUO',
            'FORTENZA_ELITE',
            'FORTENZA_VIP_TURBO',
            'GROVER',
            'INFLUX',
            'INSTIVO',
            'INVICT',
            'JOINER',
            'MAXIM_QUATTRO',
            'MINECTO_PRO',
            'MIRAVIS',
            'MIRAVIS_DUO',
            'MIRAVIS_PRO',
            'MODDUS',
            'NEMATOIDES',
            'PERGADO_MZ',
            'PLINAZOLIN',
            'POLYTRIN',
            'PRIORI_TOP',
            'PRIORI_XTRA',
            'PROCLAIM',
            'REBRON',
            'REGLONE',
            'REVUS_OPTI',
            'RIDOMIL_GOLD',
            'SCORE_FLEXI',
            'SPONTA',
            'VERDADERO',
            'VERDAVIS',
            'VOLIAM_FLEXI',
            'VOLIAM_TARGO',
            'CERTANO',
            'RIZOLIQ_LLI',
            'RIZOLIQ_UHC',
            'ALADE',
            'MITRION',
            'POLO_500_SC',
            'ADEPIDYN',
            'ORONDIS_FLEXI',
            'SCORE',
            'ORONDIS_ULTRA',
            'INZAK_ZEON',
            'REVERB',
            'MEGAFOL',
            'AEVO',
            'YIELDON',
            'FRONDEO',
            'FLEXSTAR_GT',
            'ELESTAL',
            'FANTON',
            'JOINER_PRO',
            'INVENCIS',
            'SEEKER',
            'CRESTIVO',
            'VIVA',
            'RIZODERMA',
            'RIZOFOS',
            'SIGNUM',
            'NETURE',
            'MIRAVIS_XTRA',
            'VANIVA',
            'CRUISER_OPTI-CRUISER_ADVANCED',
            'VICTRATO_GOLD',
            'BOUNDARY_EC',
            'VICTRATO'
        ]
        
        produto_selecionado = st.selectbox(
            "📦 Produto:",
            options=todos_produtos,
            key="produto_selectbox"
        )
        
        if produto_selecionado != 'Todos':
            st.session_state.filtros_aplicados['campaign_produto'] = produto_selecionado
        elif 'campaign_produto' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_produto']
    
    if 'campaign_cultura' in df_classificado.columns:
        culturas = sorted(df_classificado['campaign_cultura'].dropna().unique())
        cultura_selecionada = st.selectbox(
            "🌱 Cultura/Setor:",
            options=['Todas'] + list(culturas)
        )
        if cultura_selecionada != 'Todas':
            st.session_state.filtros_aplicados['campaign_cultura'] = cultura_selecionada
        elif 'campaign_cultura' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_cultura']

with filtro_col2:
    if 'campaign_tipo_campanha' in df_classificado.columns:
        tipos = sorted(df_classificado['campaign_tipo_campanha'].dropna().unique())
        tipo_selecionado = st.selectbox(
            "🎯 Tipo de Campanha:",
            options=['Todos'] + list(tipos)
        )
        if tipo_selecionado != 'Todos':
            st.session_state.filtros_aplicados['campaign_tipo_campanha'] = tipo_selecionado
        elif 'campaign_tipo_campanha' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_tipo_campanha']
    
    if 'campaign_objetivo' in df_classificado.columns:
        objetivos = sorted(df_classificado['campaign_objetivo'].dropna().unique())
        objetivo_selecionado = st.selectbox(
            "🎯 Objetivo:",
            options=['Todos'] + list(objetivos)
        )
        if objetivo_selecionado != 'Todos':
            st.session_state.filtros_aplicados['campaign_objetivo'] = objetivo_selecionado
        elif 'campaign_objetivo' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_objetivo']

with filtro_col3:
    if 'campaign_etapa_funil' in df_classificado.columns:
        etapas = sorted(df_classificado['campaign_etapa_funil'].dropna().unique())
        etapa_selecionada = st.selectbox(
            "📊 Etapa do Funil:",
            options=['Todas'] + list(etapas)
        )
        if etapa_selecionada != 'Todas':
            st.session_state.filtros_aplicados['campaign_etapa_funil'] = etapa_selecionada
        elif 'campaign_etapa_funil' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_etapa_funil']
    
    if 'campaign_iniciativa' in df_classificado.columns:
        iniciativas = sorted(df_classificado['campaign_iniciativa'].dropna().unique())
        iniciativa_selecionada = st.selectbox(
            "🚀 Iniciativa:",
            options=['Todas'] + list(iniciativas)
        )
        if iniciativa_selecionada != 'Todas':
            st.session_state.filtros_aplicados['campaign_iniciativa'] = iniciativa_selecionada
        elif 'campaign_iniciativa' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_iniciativa']

with filtro_col4:
    if 'campaign_plataforma' in df_classificado.columns:
        plataformas = sorted(df_classificado['campaign_plataforma'].dropna().unique())
        plataforma_selecionada = st.selectbox(
            "🖥️ Plataforma:",
            options=['Todas'] + list(plataformas)
        )
        if plataforma_selecionada != 'Todas':
            st.session_state.filtros_aplicados['campaign_plataforma'] = plataforma_selecionada
        elif 'campaign_plataforma' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_plataforma']

    if 'campaign_agencia' in df_classificado.columns:
        agencias = sorted(df_classificado['campaign_agencia'].dropna().unique())
        agencia_selecionada = st.selectbox(
            "🏢 Agência:",
            options=['Todas'] + list(agencias)
        )
        if agencia_selecionada != 'Todas':
            st.session_state.filtros_aplicados['campaign_agencia'] = agencia_selecionada
        elif 'campaign_agencia' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_agencia']

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    if st.button("✅ Aplicar Filtros", use_container_width=True):
        st.rerun()

with col_btn2:
    if st.button("🔄 Limpar Filtros", use_container_width=True):
        st.session_state.filtros_aplicados = {}
        st.rerun()

with col_btn3:
    busca_campanha = st.text_input(
        "",
        placeholder="Digite parte do nome da campanha...",
        key="busca_campanha_input",
        label_visibility="collapsed"
    )
    
    if busca_campanha:
        st.session_state.busca_campanha = busca_campanha
    elif 'busca_campanha' not in st.session_state:
        st.session_state.busca_campanha = ""

df_filtrado = df_classificado.copy()
if st.session_state.filtros_aplicados:
    for coluna, valor in st.session_state.filtros_aplicados.items():
        if coluna in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado[coluna] == valor]

if 'busca_campanha' in st.session_state and st.session_state.busca_campanha:
    busca_termo = st.session_state.busca_campanha.lower().strip()
    if busca_termo and 'campaign' in df_filtrado.columns:
        df_filtrado = df_filtrado[
            df_filtrado['campaign'].astype(str).str.lower().str.contains(busca_termo, na=False)
        ]

filtros_ativos = []

if st.session_state.filtros_aplicados:
    filtros_ativos.extend([f"{k.replace('campaign_', '')}: {v}" for k, v in st.session_state.filtros_aplicados.items()])

if 'busca_campanha' in st.session_state and st.session_state.busca_campanha:
    filtros_ativos.append(f"Busca: '{st.session_state.busca_campanha}'")

if filtros_ativos:
    st.markdown(f"### 📊 Dados Filtrados: {len(df_filtrado):,} registros")
    filtros_texto = " | ".join(filtros_ativos)
    st.info(f"**Filtros ativos:** {filtros_texto}")
    
    st.markdown("**Filtros aplicados:**")
    col_badges = st.columns(min(8, len(filtros_ativos)))
    for idx, filtro in enumerate(filtros_ativos):
        with col_badges[idx % 8]:
            st.markdown(f'<span style="background:#e0f2fe; padding:5px 10px; border-radius:10px; margin:2px; font-size:0.8em">{filtro}</span>', unsafe_allow_html=True)
else:
    st.markdown(f"### 📊 Dados Completos: {len(df_filtrado):,} registros")
    st.info("ℹ️ Nenhum filtro aplicado. Todos os dados estão visíveis.")

# Abas principais
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📋 Visão Geral", 
    "📈 Análise Numérica", 
    "🔍 Explorar Colunas", 
    "📊 Visualizar Dados",
    "🎯 Performance",
    "🤖 Análise com IA",
    "🎪 Classificador Campanhas",
    "📅 Análise MoM",
    "📊 Cenário YoY",
    "📋 Dados Colados"
])

# =============================================================================
# TAB 1: VISÃO GERAL
# =============================================================================

with tab1:
    st.header("📋 Visão Geral das Colunas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        safe_metric("Total de Colunas", len(df_filtrado.columns))
    
    with col2:
        col_numericas_filtradas = identificar_colunas_numericas(df_filtrado)
        safe_metric("Colunas Numéricas", len(col_numericas_filtradas))
    
    with col3:
        safe_metric("Total de Registros", len(df_filtrado))
    
    with col4:
        try:
            memoria_mb = df_filtrado.memory_usage(deep=True).sum() / 1024**2
            safe_metric("Uso de Memória", f"{memoria_mb:.2f} MB")
        except:
            safe_metric("Uso de Memória", "N/A")
    
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
    
    colunas_para_mostrar = []
    
    for col in df_filtrado.columns:
        incluir = True
        
        if tipo_filtro == "Numéricas":
            incluir = col in col_numericas_filtradas
        elif tipo_filtro == "Texto":
            incluir = df_filtrado[col].dtype == 'object' and col not in col_numericas_filtradas
        elif tipo_filtro == "Datas":
            incluir = pd.api.types.is_datetime64_any_dtype(df_filtrado[col])
        
        if pesquisa_coluna and pesquisa_coluna.lower() not in col.lower():
            incluir = False
        
        if incluir:
            colunas_para_mostrar.append(col)
    
    for col in sorted(colunas_para_mostrar)[:50]:
        analise = analisar_coluna(df_filtrado, col)
        
        if analise:
            with st.expander(f"**{col}** ({analise['tipo_detalhado'] if 'tipo_detalhado' in analise else analise['tipo']})"):
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    safe_metric("Tipo", analise['tipo'])
                    safe_metric("Não nulos", analise['nao_nulos'])
                    safe_metric("Valores únicos", analise['valores_unicos'])
                
                with col_info2:
                    safe_metric("Nulos", analise['nulos'])
                    safe_metric("% Nulos", f"{analise['percentual_nulos']:.1f}%")
                
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
    
    col_numericas_filtradas = identificar_colunas_numericas(df_filtrado)
    
    if not col_numericas_filtradas:
        st.warning("Nenhuma coluna numérica")
    else:
        st.success(f"✅ {len(col_numericas_filtradas)} colunas numéricas")
        
        colunas_selecionadas = st.multiselect(
            "Selecione colunas para análise",
            options=col_numericas_filtradas,
            default=col_numericas_filtradas[:min(5, len(col_numericas_filtradas))],
            key="colunas_selecionadas_tab2"
        )
        
        if colunas_selecionadas:
            st.subheader("📊 Estatísticas Descritivas")
            
            df_numeric = df_filtrado[colunas_selecionadas].apply(pd.to_numeric, errors='coerce')
            
            stats_df = df_numeric.describe().T
            stats_df['missing'] = df_numeric.isna().sum()
            stats_df['missing_pct'] = (df_numeric.isna().sum() / len(df_filtrado) * 100)
            
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
            
            if len(colunas_selecionadas) > 0:
                st.subheader("📈 Distribuições")
                
                num_cols = min(3, len(colunas_selecionadas))
                cols_vis = st.columns(num_cols)
                
                for idx, col in enumerate(colunas_selecionadas[:num_cols*3]):
                    with cols_vis[idx % num_cols]:
                        fig = criar_visualizacao_coluna(df_filtrado, col)
                        if fig is not None and fig:
                            st.plotly_chart(
                                fig, 
                                use_container_width=True,
                                key=f"histogram_{col}_{idx}"
                            )

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
                    st.plotly_chart(
                        fig_corr, 
                        use_container_width=True,
                        key="correlation_matrix_tab2"
                    )
                    
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
                        st.dataframe(
                            df_corr, 
                            use_container_width=True,
                            key="strong_correlations_table"
                        )
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
        options=sorted(df_filtrado.columns),
        index=0,
        key="coluna_selecionada_tab3"
    )
    
    if coluna_selecionada:
        analise = analisar_coluna(df_filtrado, coluna_selecionada)
        
        if analise is not None:
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                safe_metric("Total de Valores", analise['total'])
                safe_metric("Valores Não Nulos", analise['nao_nulos'])
                safe_metric("Valores Únicos", analise['valores_unicos'])
            
            with col_info2:
                safe_metric("Valores Nulos", analise['nulos'])
                safe_metric("% Nulos", f"{analise['percentual_nulos']:.1f}%")
            
            st.subheader("📊 Visualização")
            fig = criar_visualizacao_coluna(df_filtrado, coluna_selecionada)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Não foi possível criar visualização para esta coluna")
            
            st.subheader("📋 Amostra de Valores")
            
            col_amostra1, col_amostra2 = st.columns(2)
            
            with col_amostra1:
                st.write("**Primeiros 10:**")
                try:
                    primeiros = df_filtrado[coluna_selecionada].head(10).tolist()
                    primeiros_str = [str(x) for x in primeiros]
                    for val in primeiros_str:
                        st.write(f"- {val}")
                except:
                    st.write("Erro ao mostrar valores")
            
            with col_amostra2:
                st.write("**Últimos 10:**")
                try:
                    ultimos = df_filtrado[coluna_selecionada].tail(10).tolist()
                    ultimos_str = [str(x) for x in ultimos]
                    for val in ultimos_str:
                        st.write(f"- {val}")
                except:
                    st.write("Erro ao mostrar valores")
            
            if analise['tipo_detalhado'] == 'Texto/Categórica' and analise['valores_unicos'] <= 100:
                st.subheader("📊 Distribuição")
                
                try:
                    contagem = df_filtrado[coluna_selecionada].value_counts()
                    df_contagem = pd.DataFrame({
                        'Valor': contagem.index.astype(str),
                        'Contagem': contagem.values,
                        'Percentual': (contagem.values / len(df_filtrado) * 100)
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
    
    colunas_vis = st.multiselect(
        "Selecione colunas para visualizar",
        options=sorted(df_filtrado.columns),
        default=sorted(df_filtrado.columns)[:min(10, len(df_filtrado.columns))],
        key="colunas_vis_tab4"
    )
    
    if colunas_vis:
        st.subheader("🔍 Filtros Adicionais")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        df_filtrado_tab4 = df_filtrado.copy()
        
        with col_f1:
            if 'datasource' in df_filtrado.columns:
                datasources = sorted(df_filtrado['datasource'].dropna().unique())
                ds_selecionados = st.multiselect(
                    "Data Sources",
                    options=datasources,
                    default=datasources[:min(3, len(datasources))],
                    key="ds_selecionados_tab4"
                )
                if ds_selecionados:
                    df_filtrado_tab4 = df_filtrado_tab4[df_filtrado_tab4['datasource'].isin(ds_selecionados)]
        
        with col_f2:
            colunas_num_vis = [c for c in colunas_vis if c in identificar_colunas_numericas(df_filtrado)]
            if colunas_num_vis:
                col_filtro = st.selectbox(
                    "Filtrar por coluna numérica",
                    options=['Nenhum'] + colunas_num_vis,
                    key="col_filtro_tab4"
                )
                if col_filtro != 'Nenhum':
                    try:
                        col_data = pd.to_numeric(df_filtrado_tab4[col_filtro], errors='coerce').dropna()
                        if len(col_data) > 0:
                            min_val = st.number_input(
                                f"Valor mínimo de {col_filtro}",
                                value=float(col_data.min()),
                                key=f"min_val_{col_filtro}_tab4"
                            )
                            df_filtrado_tab4 = df_filtrado_tab4[pd.to_numeric(df_filtrado_tab4[col_filtro], errors='coerce') >= min_val]
                    except:
                        st.warning(f"Não foi possível filtrar por {col_filtro}")
        
        with col_f3:
            limite_linhas = st.slider("Linhas para mostrar", 10, 1000, 100, key="limite_linhas_tab4")
        
        st.subheader(f"📋 Dados ({len(df_filtrado_tab4):,} registros)")
        
        if len(df_filtrado_tab4) > 0:
            total_pages = max(1, len(df_filtrado_tab4) // limite_linhas + 1)
            
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
                st.caption(f"Total: {len(df_filtrado_tab4):,} registros")
            
            start_idx = (page_number - 1) * limite_linhas
            end_idx = min(start_idx + limite_linhas, len(df_filtrado_tab4))
            
            df_display = df_filtrado_tab4[colunas_vis].iloc[start_idx:end_idx].copy()
            
            for col in colunas_vis:
                if col in identificar_colunas_numericas(df_filtrado):
                    try:
                        df_display[col] = df_display[col].apply(
                            lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else ""
                        )
                    except:
                        pass
                elif pd.api.types.is_datetime64_any_dtype(df_filtrado[col]):
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
        
        st.subheader("📥 Exportar")
        
        if len(df_filtrado_tab4) > 0:
            csv = df_filtrado_tab4[colunas_vis].to_csv(index=False)
            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name=f"dados_filtrados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
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
    
    if 'campaign' not in df_filtrado.columns:
        st.error("❌ Coluna 'campaign' não encontrada.")
    else:
        st.subheader("📊 Métricas Gerais")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                num_campaigns = df_filtrado['campaign'].nunique()
                safe_metric("Campanhas", num_campaigns)
            except:
                safe_metric("Campanhas", "Erro")
        
        with col2:
            if 'date' in df_filtrado.columns:
                try:
                    df_date = df_filtrado['date'].dropna()
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
            if 'datasource' in df_filtrado.columns:
                try:
                    sources = df_filtrado['datasource'].nunique()
                    safe_metric("Data Sources", sources)
                except:
                    safe_metric("Data Sources", "Erro")
        
        with col4:
            try:
                num_campaigns_val = df_filtrado['campaign'].nunique()
                records_per_campaign = len(df_filtrado) / num_campaigns_val if num_campaigns_val > 0 else 0
                safe_metric("Média Reg/Camp", f"{records_per_campaign:.1f}")
            except:
                safe_metric("Média Reg/Camp", "Erro")
        
        st.subheader("📈 Top Campanhas")
        
        if 'campaign' in df_filtrado.columns:
            try:
                campaign_stats = df_filtrado['campaign'].value_counts().head(10)
                
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
        
        st.subheader("💰 Métricas Financeiras")
        
        financial_metrics = []
        col_numericas_filtradas = identificar_colunas_numericas(df_filtrado)
        for metric in ['spend', 'revenue', 'conversions', 'roas', 'cpc']:
            for col in col_numericas_filtradas:
                if metric in col.lower():
                    financial_metrics.append(col)
                    break
        
        if financial_metrics:
            cols = st.columns(min(4, len(financial_metrics)))
            for idx, metric in enumerate(financial_metrics[:4]):
                with cols[idx]:
                    if metric in df_filtrado.columns:
                        try:
                            total = pd.to_numeric(df_filtrado[metric], errors='coerce').sum()
                            safe_metric(metric, total)
                        except:
                            safe_metric(metric, "Erro")
        
        st.subheader("📊 Análise por Categoria")
        
        col_cat1, col_cat2 = st.columns(2)
        
        with col_cat1:
            if 'campaign_cliente' in df_filtrado.columns:
                try:
                    cliente_stats = df_filtrado['campaign_cliente'].value_counts().head(10)
                    
                    fig_cliente = px.pie(
                        values=cliente_stats.values,
                        names=cliente_stats.index,
                        title="Distribuição por Cliente",
                        hole=0.3
                    )
                    st.plotly_chart(fig_cliente, use_container_width=True)
                except:
                    pass
        
        with col_cat2:
            if 'campaign_tipo_campanha' in df_filtrado.columns:
                try:
                    tipo_stats = df_filtrado['campaign_tipo_campanha'].value_counts().head(10)
                    
                    fig_tipo = px.bar(
                        x=tipo_stats.index,
                        y=tipo_stats.values,
                        title="Distribuição por Tipo de Campanha"
                    )
                    fig_tipo.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_tipo, use_container_width=True)
                except:
                    pass

# =============================================================================
# TAB 6: ANÁLISE COM IA
# =============================================================================

with tab6:
    st.header("🤖 Análise com Gemini IA")
    
    if not modelo_texto:
        st.error("❌ Gemini não configurado!")
        st.stop()
    
    if df_filtrado.empty:
        st.warning("📭 Nenhum dado carregado.")
        st.stop()
    
    st.markdown("### 🔍 Filtros para Análise")
    
    with st.expander("⚙️ Configurar", expanded=True):
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            if 'datasource' in df_filtrado.columns:
                datasources = sorted(df_filtrado['datasource'].dropna().unique())
                selected_ds = st.multiselect(
                    "Data Sources:",
                    options=datasources,
                    default=datasources[:min(3, len(datasources))],
                    key="selected_ds_tab6"
                )
            else:
                selected_ds = None
            
            if 'date' in df_filtrado.columns:
                try:
                    date_series = df_filtrado['date'].dropna()
                    if len(date_series) > 0:
                        if not pd.api.types.is_datetime64_any_dtype(date_series):
                            date_series = pd.to_datetime(date_series, errors='coerce')
                        
                        min_date = date_series.min().date()
                        max_date = date_series.max().date()
                        
                        date_range = st.date_input(
                            "Período:",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key="date_range_tab6"
                        )
                    else:
                        date_range = None
                except Exception as e:
                    date_range = None
            else:
                date_range = None
        
        with col_filter2:
            if 'campaign' in df_filtrado.columns:
                campaigns = sorted(df_filtrado['campaign'].dropna().unique())
                selected_campaigns = st.multiselect(
                    "Campanhas (opcional):",
                    options=campaigns,
                    key="selected_campaigns_tab6"
                )
            else:
                selected_campaigns = None
            
            df_len = len(df_filtrado)
            max_records_value = min(10000, max(100, df_len)) if df_len > 0 else 5000
            max_records = st.slider(
                "Máximo de registros:",
                min_value=100,
                max_value=min(10000, df_len) if df_len > 0 else 10000,
                value=min(5000, max_records_value),
                step=100,
                key="max_records_tab6"
            )
    
    df_filtered_ia = df_filtrado.copy()
    
    if selected_ds and 'datasource' in df_filtered_ia.columns and len(selected_ds) > 0:
        df_filtered_ia = df_filtered_ia[df_filtered_ia['datasource'].isin(selected_ds)]
    
    if date_range and len(date_range) == 2 and 'date' in df_filtered_ia.columns:
        start_date, end_date = date_range
        
        if not pd.api.types.is_datetime64_any_dtype(df_filtered_ia['date']):
            df_filtered_ia['date'] = pd.to_datetime(df_filtered_ia['date'], errors='coerce')
        
        mask = df_filtered_ia['date'].notna()
        
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        df_filtered_ia = df_filtered_ia[
            mask & 
            (df_filtered_ia['date'] >= start_dt) & 
            (df_filtered_ia['date'] <= end_dt)
        ]
    
    if selected_campaigns and 'campaign' in df_filtered_ia.columns and len(selected_campaigns) > 0:
        df_filtered_ia = df_filtered_ia[df_filtered_ia['campaign'].isin(selected_campaigns)]
    
    df_filtered_ia = df_filtered_ia.head(max_records)
    
    st.markdown("### 📊 Dados Selecionados")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        safe_metric("Registros", len(df_filtered_ia))
    
    with col_stat2:
        if 'campaign' in df_filtered_ia.columns:
            try:
                num_campaigns = df_filtered_ia['campaign'].nunique()
                safe_metric("Campanhas", num_campaigns)
            except:
                safe_metric("Campanhas", "Erro")
    
    with col_stat3:
        if 'datasource' in df_filtered_ia.columns:
            try:
                num_sources = df_filtered_ia['datasource'].nunique()
                safe_metric("Data Sources", num_sources)
            except:
                safe_metric("Data Sources", "Erro")
    
    with col_stat4:
        if 'date' in df_filtered_ia.columns:
            try:
                date_series = df_filtered_ia['date'].dropna()
                if len(date_series) > 0:
                    if not pd.api.types.is_datetime64_any_dtype(date_series):
                        date_series = pd.to_datetime(date_series, errors='coerce')
                    period_days = (date_series.max() - date_series.min()).days + 1
                    safe_metric("Dias", period_days)
                else:
                    safe_metric("Dias", 0)
            except:
                safe_metric("Dias", "Erro")
    
    st.markdown("### 🎯 Configuração")
    
    analysis_focus = st.selectbox(
        "Foco da Análise:",
        options=["overall", "trends", "efficiency", "complete"],
        format_func=lambda x: {
            "overall": "📈 Performance Geral",
            "trends": "📊 Tendências", 
            "efficiency": "💰 Eficiência",
            "complete": "🏆 Análise Completa"
        }[x],
        key="analysis_focus_tab6"
    )
    
    user_instructions = st.text_area(
        "📝 Instruções (opcional):",
        placeholder="Ex: Foque no ROI, identifique as melhores campanhas, analise tendências por data source...",
        height=100,
        key="user_instructions_tab6"
    )
    
    st.markdown("### 🚀 Gerar Análise")
    
    generate_button = st.button("🤖 Gerar Análise com Gemini", type="primary", use_container_width=True, key="generate_button_tab6")
    
    if generate_button:
        if df_filtered_ia.empty:
            st.error("❌ Nenhum dado após filtros.")
        else:
            with st.spinner(f"🤖 Analisando {len(df_filtered_ia):,} registros..."):
                try:
                    analysis_result = generate_gemini_analysis(
                        df_filtered_ia, 
                        analysis_focus, 
                        user_instructions
                    )
                    st.session_state.gemini_analysis = analysis_result
                    st.success("✅ Análise concluída!")
                except Exception as e:
                    st.error(f"❌ Erro ao gerar análise: {str(e)[:200]}")
    
    if st.session_state.gemini_analysis:
        st.markdown("---")
        st.markdown("### 📄 Relatório de Análise")
        
        col_actions1, col_actions2, col_actions3 = st.columns(3)
        
        with col_actions1:
            analysis_text = st.session_state.gemini_analysis
            st.download_button(
                label="💾 Baixar Relatório",
                data=analysis_text,
                file_name=f"analise_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_report_tab6"
            )
        
        with col_actions2:
            if st.button("🎬 Gerar Descrição dos Slides", use_container_width=True, type="secondary", key="generate_slides_tab6"):
                with st.spinner("Gerando descrição para slides..."):
                    slides_desc = generate_slides_description(
                        st.session_state.gemini_analysis,
                        user_instructions
                    )
                    
                    st.session_state.slides_description = slides_desc
                    st.success("✅ Descrição dos slides gerada!")
                    st.rerun()
        
        with col_actions3:
            if st.button("🔄 Nova Análise", use_container_width=True, key="new_analysis_tab6"):
                st.session_state.gemini_analysis = None
                st.session_state.slides_description = None
                st.rerun()
    
        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
        st.markdown(st.session_state.gemini_analysis)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.slides_description:
            st.markdown("---")
            st.markdown("### 🎬 Descrição para Slides")
            
            slides_text = st.session_state.slides_description
            st.download_button(
                label="📥 Baixar Descrição dos Slides",
                data=slides_text,
                file_name=f"slides_descricao_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_slides_tab6"
            )
            
            st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
            st.markdown(st.session_state.slides_description)
            st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 7: CLASSIFICADOR DE CAMPANHAS MULTI-CLIENTES
# =============================================================================

with tab7:
    st.markdown('<div class="campaign-classifier"><h2>🎪 Classificador de Campanhas Multi-Clientes</h2></div>', unsafe_allow_html=True)
    
    dicionario_categorias = carregar_dicionario_categorias()
    
    col_intro1, col_intro2 = st.columns(2)
    
    with col_intro1:
        st.markdown("### 📋 Sobre o Sistema")
    
    with col_intro2:
        st.markdown("### 🔍 Status da Classificação")
        
        if 'classificado' in df_classificado.columns:
            total = len(df_classificado)
            classificadas = df_classificado[df_classificado['classificado'] == 'SIM'].shape[0]
            taxa = (classificadas / total * 100) if total > 0 else 0
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                safe_metric("Total", total)
                safe_metric("Classificadas", classificadas)
            with col_stat2:
                safe_metric("Taxa", f"{taxa:.1f}%")
                
                if 'campaign_cliente' in df_classificado.columns:
                    clientes_unicos = df_classificado['campaign_cliente'].nunique()
                    safe_metric("Clientes", clientes_unicos)
        else:
            st.warning("Nenhuma classificação disponível")
    
    st.markdown("### 📊 Distribuição por Categoria")
    
    col_dist1, col_dist2, col_dist3 = st.columns(3)
    
    with col_dist1:
        if 'campaign_cliente' in df_classificado.columns:
            try:
                cliente_counts = df_classificado['campaign_cliente'].value_counts().head(10)
                
                fig_clientes = px.bar(
                    x=cliente_counts.index,
                    y=cliente_counts.values,
                    title="Top 10 Clientes",
                    color=cliente_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_clientes.update_xaxes(tickangle=45)
                st.plotly_chart(fig_clientes, use_container_width=True)
            except:
                pass
    
    with col_dist2:
        if 'campaign_tipo_campanha' in df_classificado.columns:
            try:
                tipo_counts = df_classificado['campaign_tipo_campanha'].value_counts().head(10)
                
                fig_tipos = px.pie(
                    values=tipo_counts.values,
                    names=tipo_counts.index,
                    title="Tipos de Campanha",
                    hole=0.3
                )
                st.plotly_chart(fig_tipos, use_container_width=True)
            except:
                pass
    
    with col_dist3:
        if 'campaign_etapa_funil' in df_classificado.columns:
            try:
                etapa_counts = df_classificado['campaign_etapa_funil'].value_counts()
                fig_etapas = px.bar(
                    x=etapa_counts.index,
                    y=etapa_counts.values,
                    title="Etapas do Funil",
                    color=etapa_counts.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_etapas, use_container_width=True)
            except:
                pass
    
    st.markdown("### 🔍 Explorador de Campanhas")
    
    col_explorer1, col_explorer2 = st.columns(2)
    
    with col_explorer1:
        if 'campaign' in df_classificado.columns:
            campanhas = sorted(df_classificado['campaign'].dropna().unique())
            campanha_selecionada = st.selectbox(
                "Selecione uma campanha para análise:",
                options=campanhas[:100],
                key="campanha_selecionada_tab7"
            )
            
            if campanha_selecionada:
                campanha_data = df_classificado[df_classificado['campaign'] == campanha_selecionada].iloc[0]
                
                st.markdown("#### 📋 Detalhes da Campanha")
                st.write(f"**Nome:** {campanha_selecionada}")
                
                categorias_identificadas = {}
                for col in df_classificado.columns:
                    if col.startswith('campaign_') and col != 'campaign_classificado' and col != 'categorias_identificadas':
                        valor = campanha_data[col]
                        if pd.notna(valor):
                            nome_categoria = col.replace('campaign_', '').replace('_', ' ').title()
                            categorias_identificadas[nome_categoria] = valor
                
                if categorias_identificadas:
                    st.markdown("#### 🏷️ Categorias Identificadas")
                    for categoria, valor in categorias_identificadas.items():
                        st.write(f"**{categoria}:** {valor}")
                else:
                    st.info("Nenhuma categoria identificada para esta campanha")
    
    with col_explorer2:
        st.markdown("#### 📈 Estatísticas de Classificação")
        
        if 'classificado' in df_classificado.columns:
            status_counts = df_classificado['classificado'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Status de Classificação",
                color=status_counts.values,
                color_discrete_sequence=['#10b981', '#ef4444']
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        if st.button("🔄 Reclassificar Campanhas", use_container_width=True, key="reclassificar_tab7"):
            with st.spinner("Reclassificando campanhas..."):
                df_classificado_novo = classificar_campanhas_multi_cliente(df)
                st.session_state.df_classificado = df_classificado_novo
                st.success("✅ Campanhas reclassificadas!")
                st.rerun()
    
    st.markdown("### 📥 Exportar Dados Classificados")
    
    if len(df_classificado) > 0:
        colunas_classificadas = [col for col in df_classificado.columns if col.startswith('campaign_')]
        colunas_base = ['campaign', 'date', 'datasource'] if all(col in df_classificado.columns for col in ['campaign', 'date', 'datasource']) else []
        colunas_exportar = colunas_base + colunas_classificadas
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            csv_data = df_classificado[colunas_exportar].to_csv(index=False)
            st.download_button(
                label="📥 Baixar Todos os Dados Classificados",
                data=csv_data,
                file_name=f"campanhas_classificadas_completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_all_tab7"
            )
        
        with col_export2:
            if 'classificado' in df_classificado.columns:
                nao_classificadas = df_classificado[df_classificado['classificado'] == 'NÃO']
                if len(nao_classificadas) > 0:
                    csv_nao_classificadas = nao_classificadas[['campaign']].to_csv(index=False)
                    st.download_button(
                        label="📥 Baixar Campanhas Não Classificadas",
                        data=csv_nao_classificadas,
                        file_name=f"campanhas_nao_classificadas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_unclassified_tab7"
                    )
    
    if modelo_texto and len(df_classificado) > 0:
        st.markdown("### 🤖 Análise de Padrões com Gemini")
        
        if st.button("🔍 Analisar Padrões de Nomenclatura", use_container_width=True, key="analisar_padroes_tab7"):
            with st.spinner("Analisando padrões de nomenclatura..."):
                try:
                    sample_size = min(50, len(df_classificado))
                    sample_campaigns = df_classificado['campaign'].dropna().sample(sample_size).tolist()
                    
                    clientes_identificados = []
                    if 'campaign_cliente' in df_classificado.columns:
                        clientes_identificados = df_classificado['campaign_cliente'].dropna().unique().tolist()
                    
                    prompt = f"""
                    Analise os seguintes nomes de campanhas de marketing e identifique:
                    
                    1. Padrões comuns de nomenclatura
                    2. Estruturas mais frequentes
                    3. Componentes principais encontrados
                    4. Clientes identificados: {', '.join(clientes_identificados[:10]) if clientes_identificados else 'Nenhum'}
                    5. Problemas de padronização
                    6. Sugestões para melhorar a classificação automática
                    
                    Amostra de nomes de campanhas:
                    {', '.join([str(c) for c in sample_campaigns])}
                    
                    Forneça uma análise detalhada em português com:
                    - Identificação de padrões estruturais
                    - Componentes mais comuns (cliente, produto, objetivo, etc.)
                    - Problemas de inconsistência
                    - Recomendações para padronização futura
                    - Sugestões para melhorar a taxonomia
                    """
                    
                    response = modelo_texto.generate_content(prompt)
                    
                    st.markdown("### 📄 Análise de Padrões")
                    st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
                    st.markdown(response.text)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Erro na análise: {str(e)[:200]}")

# =============================================================================
# TAB 8: ANÁLISE MOM (MONTH-OVER-MONTH)
# =============================================================================

with tab8:
    st.markdown('<div class="mom-analysis"><h2>📅 Análise MoM (Month-over-Month)</h2></div>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Configuração da Análise")
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        # Cliente para análise
        clientes_disponiveis = ["Todos"] + sorted(df['cliente_identificado'].unique().tolist()) if 'cliente_identificado' in df.columns else ["Todos"]
        cliente_analise = st.selectbox(
            "👥 Cliente para análise:",
            options=clientes_disponiveis,
            index=0,
            key="cliente_analise_tab8"
        )
    
    with col_config2:
        # Mês atual
        if 'date' in df.columns:
            df_dates = df['date'].dropna()
            if len(df_dates) > 0:
                if not pd.api.types.is_datetime64_any_dtype(df_dates):
                    df_dates = pd.to_datetime(df_dates, errors='coerce')
                
                meses_disponiveis = sorted(df_dates.dt.to_period('M').unique(), reverse=True)
                meses_str = [str(m) for m in meses_disponiveis]
                
                mes_atual_period = st.selectbox(
                    "📅 Mês Atual:",
                    options=meses_str[:12] if len(meses_str) > 0 else [],
                    key="mes_atual_tab8"
                )
            else:
                mes_atual_period = None
                st.info("Sem datas disponíveis")
        else:
            mes_atual_period = None
            st.info("Coluna de data não encontrada")
    
    with col_config3:
        # Mês anterior
        if mes_atual_period:
            # Calcular mês anterior
            mes_atual_dt = pd.Period(mes_atual_period).to_timestamp()
            mes_anterior_dt = mes_atual_dt - pd.DateOffset(months=1)
            mes_anterior_period = pd.Period(mes_anterior_dt, freq='M')
            
            st.write(f"**Mês Anterior:** {mes_anterior_period}")
        else:
            mes_anterior_period = None
    
    st.markdown("### 📊 Executar Análise")
    
    if st.button("📈 Calcular Análise MoM", use_container_width=True, type="primary", key="calcular_mom_tab8"):
        if mes_atual_period and mes_anterior_period:
            with st.spinner(f"Calculando análise MoM para {cliente_analise}..."):
                try:
                    mom_result = calculate_mom_analysis(
                        df, 
                        cliente_analise, 
                        pd.Period(mes_atual_period),
                        pd.Period(mes_anterior_period)
                    )
                    
                    if mom_result:
                        st.session_state.mom_analysis = mom_result
                        st.success("✅ Análise MoM calculada!")
                    else:
                        st.error("❌ Não foi possível calcular a análise MoM")
                except Exception as e:
                    st.error(f"❌ Erro ao calcular MoM: {str(e)[:200]}")
        else:
            st.error("❌ Selecione o mês atual para análise")
    
    if st.session_state.mom_analysis:
        mom_data = st.session_state.mom_analysis
        
        st.markdown("### 📋 Resultados da Análise")
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        
        with col_res1:
            safe_metric("Cliente", mom_data['cliente'])
        
        with col_res2:
            safe_metric("Mês Anterior", mom_data['mes_anterior'])
        
        with col_res3:
            safe_metric("Mês Atual", mom_data['mes_atual'])
        
        with col_res4:
            total_change = mom_data['total_mes_atual'] - mom_data['total_mes_anterior']
            change_pct = (total_change / mom_data['total_mes_anterior'] * 100) if mom_data['total_mes_anterior'] > 0 else 0
            safe_metric("Variação Registros", total_change, f"{change_pct:.1f}%")
        
        st.markdown("### 📊 Análise por Plataforma")
        
        # Criar tabela de análise
        df_mom_table = create_mom_table(mom_data)
        
        if df_mom_table is not None:
            st.dataframe(
                df_mom_table.style.format(precision=2),
                use_container_width=True
            )
            
            # Gráfico de distribuição por plataforma
            if 'platform_analysis' in mom_data and mom_data['platform_analysis']:
                platform_current = {}
                platform_previous = {}
                
                for platform, data in mom_data['platform_analysis'].items():
                    platform_current[platform] = data['spend_current']
                    platform_previous[platform] = data['spend_previous']
                
                # Criar DataFrame para gráfico
                df_platforms = pd.DataFrame({
                    'Plataforma': list(platform_current.keys()),
                    'Mês Anterior': list(platform_previous.values()),
                    'Mês Atual': list(platform_current.values())
                })
                
                # Gráfico de barras
                fig_platforms = go.Figure()
                
                fig_platforms.add_trace(go.Bar(
                    name='Mês Anterior',
                    x=df_platforms['Plataforma'],
                    y=df_platforms['Mês Anterior'],
                    marker_color='#6366f1'
                ))
                
                fig_platforms.add_trace(go.Bar(
                    name='Mês Atual',
                    x=df_platforms['Plataforma'],
                    y=df_platforms['Mês Atual'],
                    marker_color='#10b981'
                ))
                
                fig_platforms.update_layout(
                    title="Investimento por Plataforma - Comparativo MoM",
                    barmode='group',
                    xaxis_title="Plataforma",
                    yaxis_title="Investimento (R$)",
                    height=500
                )
                
                st.plotly_chart(fig_platforms, use_container_width=True)
        
        st.markdown("### 📈 Análise de Métricas")
        
        if 'metric_analysis' in mom_data and mom_data['metric_analysis']:
            metric_data = mom_data['metric_analysis']
            
            if metric_data:
                cols_metrics = st.columns(min(3, len(metric_data)))
                
                for idx, (metric_name, metric_info) in enumerate(metric_data.items()):
                    if idx < 9:  # Mostrar até 9 métricas
                        with cols_metrics[idx % 3]:
                            st.markdown(f'<div class="yoy-metric">', unsafe_allow_html=True)
                            st.subheader(metric_name)
                            
                            col_curr, col_prev = st.columns(2)
                            with col_curr:
                                st.metric(
                                    "Atual", 
                                    format_currency(metric_info['current']) if 'spend' in metric_name.lower() or 'cost' in metric_name.lower() or 'revenue' in metric_name.lower() else f"{metric_info['current']:,.0f}"
                                )
                            
                            with col_prev:
                                st.metric(
                                    "Anterior", 
                                    format_currency(metric_info['previous']) if 'spend' in metric_name.lower() or 'cost' in metric_name.lower() or 'revenue' in metric_name.lower() else f"{metric_info['previous']:,.0f}"
                                )
                            
                            change_color = "green" if metric_info['change'] > 0 else "red"
                            st.markdown(f"**Variação:** <span style='color:{change_color}'>{format_currency(metric_info['change']) if 'spend' in metric_name.lower() or 'cost' in metric_name.lower() or 'revenue' in metric_name.lower() else f"{metric_info['change']:,.0f}"} ({metric_info['change_pct']:.1f}%)</span>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📝 Relatório de Análise")
        
        if modelo_texto:
            if st.button("📄 Gerar Relatório com Gemini", use_container_width=True, key="gerar_relatorio_mom_tab8"):
                with st.spinner("Gerando relatório de análise MoM..."):
                    try:
                        # Preparar dados para Gemini
                        analysis_text = f"""
                        CLIENTE: {mom_data['cliente']}
                        PERÍODO: {mom_data['mes_anterior']} vs {mom_data['mes_atual']}
                        
                        RESUMO GERAL:
                        - Total de registros mês anterior: {mom_data['total_mes_anterior']:,}
                        - Total de registros mês atual: {mom_data['total_mes_atual']:,}
                        - Variação: {mom_data['total_mes_atual'] - mom_data['total_mes_anterior']:,} ({((mom_data['total_mes_atual'] - mom_data['total_mes_anterior']) / mom_data['total_mes_anterior'] * 100):.1f}%)
                        
                        ANÁLISE POR PLATAFORMA:
                        """
                        
                        for platform, data in mom_data.get('platform_analysis', {}).items():
                            analysis_text += f"""
                            - {platform}:
                              * Investimento anterior: R$ {data['spend_previous']:,.2f}
                              * Investimento atual: R$ {data['spend_current']:,.2f}
                              * Variação: R$ {data['spend_change']:,.2f} ({data['spend_change_pct']:.1f}%)
                              * Registros anterior: {data['records_previous']:,}
                              * Registros atual: {data['records_current']:,}
                            """
                        
                        analysis_text += """
                        
                        ANÁLISE DE MÉTRICAS:
                        """
                        
                        for metric, data in mom_data.get('metric_analysis', {}).items():
                            analysis_text += f"""
                            - {metric}:
                              * Anterior: {data['previous']:,.2f}
                              * Atual: {data['current']:,.2f}
                              * Variação: {data['change']:,.2f} ({data['change_pct']:.1f}%)
                            """
                        
                        prompt = f"""
                        # 📊 RELATÓRIO DE ANÁLISE MoM (Month-over-Month)
                        
                        {analysis_text}
                        
                        ## 🎯 TAREFA:
                        
                        Com base nos dados MoM acima, crie um relatório executivo em português que inclua:
                        
                        1. **📈 RESUMO EXECUTIVO** (1-2 parágrafos com os principais achados)
                        2. **💰 ANÁLISE DE INVESTIMENTO** (comparativo por plataforma, eficiência)
                        3. **📊 ANÁLISE DE PERFORMANCE** (principais métricas e suas variações)
                        4. **🔍 INSIGHTS ESTRATÉGICOS** (3-5 insights baseados nos dados)
                        5. **🎯 RECOMENDAÇÕES** (5-7 recomendações acionáveis para o próximo mês)
                        6. **📅 PRÓXIMOS PASSOS** (plano de ação sugerido)
                        
                        Foque em:
                        - Evolução da redistribuição estratégica de investimento por plataforma
                        - Estratégia eficaz com menor investimento (se aplicável)
                        - Eficiência e diminuição nos custos das campanhas de mídia paga
                        - Performance geral e engajamento
                        
                        Seja específico com os números e forneça interpretações práticas.
                        """
                        
                        response = modelo_texto.generate_content(prompt)
                        
                        st.markdown("### 📄 Relatório Gemini")
                        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
                        st.markdown(response.text)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar relatório: {str(e)[:200]}")

# =============================================================================
# TAB 9: CENÁRIO YOY (YEAR-OVER-YEAR)
# =============================================================================

with tab9:
    st.markdown('<div class="yoy-scenario"><h2>📊 Cenário YoY (Year-over-Year)</h2></div>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Configurar Cenário YoY")
    
    # Cliente para análise
    col_yoy1, col_yoy2 = st.columns(2)
    
    with col_yoy1:
        cliente_yoy = st.selectbox(
            "👥 Cliente:",
            options=["Syngenta", "Golden Harvest Brasil", "Nidera", "NK Seeds", "EuroChem", "Grupo Vittia", "Outros"],
            index=0,
            key="cliente_yoy_tab9"
        )
        
        ano_atual = st.number_input(
            "📅 Ano Atual:",
            min_value=2020,
            max_value=2030,
            value=2025,
            step=1,
            key="ano_atual_tab9"
        )
        
        ano_anterior = st.number_input(
            "📅 Ano Anterior:",
            min_value=2020,
            max_value=2030,
            value=2024,
            step=1,
            key="ano_anterior_tab9"
        )
    
    with col_yoy2:
        mes_yoy = st.selectbox(
            "📅 Mês:",
            options=[
                "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
                "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
            ],
            index=10,  # Novembro
            key="mes_yoy_tab9"
        )
        
        periodo_yoy = st.selectbox(
            "⏱️ Período:",
            options=["Mês Completo", "Primeira Quinzena", "Segunda Quinzena", "Semana Específica"],
            index=0,
            key="periodo_yoy_tab9"
        )
    
    st.markdown("### 📈 Inserir Dados YoY")
    
    with st.expander("💰 Dados de Investimento", expanded=True):
        st.markdown("#### Investimento por Plataforma")
        
        col_inv1, col_inv2, col_inv3, col_inv4 = st.columns(4)
        
        with col_inv1:
            fb_atual = st.number_input("FB/Insta Atual (R$):", min_value=0.0, value=60817.0, step=1000.0, key="fb_atual_tab9")
            fb_anterior = st.number_input("FB/Insta Anterior (R$):", min_value=0.0, value=60357.0, step=1000.0, key="fb_anterior_tab9")
        
        with col_inv2:
            tiktok_atual = st.number_input("TikTok Atual (R$):", min_value=0.0, value=45831.0, step=1000.0, key="tiktok_atual_tab9")
            tiktok_anterior = st.number_input("TikTok Anterior (R$):", min_value=0.0, value=33004.0, step=1000.0, key="tiktok_anterior_tab9")
        
        with col_inv3:
            display_atual = st.number_input("Display Atual (R$):", min_value=0.0, value=93618.0, step=1000.0, key="display_atual_tab9")
            display_anterior = st.number_input("Display Anterior (R$):", min_value=0.0, value=58313.0, step=1000.0, key="display_anterior_tab9")
        
        with col_inv4:
            youtube_atual = st.number_input("YouTube Atual (R$):", min_value=0.0, value=81890.0, step=1000.0, key="youtube_atual_tab9")
            youtube_anterior = st.number_input("YouTube Anterior (R$):", min_value=0.0, value=60379.0, step=1000.0, key="youtube_anterior_tab9")
    
    col_inv5, col_inv6, col_inv7 = st.columns(3)
    
    with col_inv5:
        pmax_atual = st.number_input("PMax Atual (R$):", min_value=0.0, value=14632.0, step=1000.0, key="pmax_atual_tab9")
        pmax_anterior = st.number_input("PMax Anterior (R$):", min_value=0.0, value=17.0, step=1000.0, key="pmax_anterior_tab9")
    
    with col_inv6:
        search_atual = st.number_input("Search Atual (R$):", min_value=0.0, value=3451.15, step=1000.0, key="search_atual_tab9")
        search_anterior = st.number_input("Search Anterior (R$):", min_value=0.0, value=0.0, step=1000.0, key="search_anterior_tab9")
    
    with col_inv7:
        outras_atual = st.number_input("Outras Atual (R$):", min_value=0.0, value=0.0, step=1000.0, key="outras_atual_tab9")
        outras_anterior = st.number_input("Outras Anterior (R$):", min_value=0.0, value=0.0, step=1000.0, key="outras_anterior_tab9")
    
    with st.expander("📊 Dados de Performance", expanded=True):
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        
        with col_perf1:
            sessoes_atual = st.number_input("Sessões Atual:", min_value=0, value=938000, step=1000, key="sessoes_atual_tab9")
            sessoes_anterior = st.number_input("Sessões Anterior:", min_value=0, value=1050000, step=1000, key="sessoes_anterior_tab9")
            
            tempo_medio_atual = st.number_input("Tempo Médio Atual (min):", min_value=0.0, value=2.0, step=0.1, key="tempo_medio_atual_tab9")
            tempo_medio_anterior = st.number_input("Tempo Médio Anterior (min):", min_value=0.0, value=2.06, step=0.1, key="tempo_medio_anterior_tab9")
        
        with col_perf2:
            engajamento_atual = st.number_input("Engajamento Atual:", min_value=0, value=22500000, step=100000, key="engajamento_atual_tab9")
            engajamento_anterior = st.number_input("Engajamento Anterior:", min_value=0, value=20250000, step=100000, key="engajamento_anterior_tab9")
            
            views_atual = st.number_input("Views Atual:", min_value=0, value=15000000, step=100000, key="views_atual_tab9")
            views_anterior = st.number_input("Views Anterior:", min_value=0, value=11200000, step=100000, key="views_anterior_tab9")
        
        with col_perf3:
            conversoes_atual = st.number_input("Conversões Atual:", min_value=0, value=1500, step=10, key="conversoes_atual_tab9")
            conversoes_anterior = st.number_input("Conversões Anterior:", min_value=0, value=1200, step=10, key="conversoes_anterior_tab9")
            
            cpc_atual = st.number_input("CPC Atual (R$):", min_value=0.0, value=1.85, step=0.01, key="cpc_atual_tab9")
            cpc_anterior = st.number_input("CPC Anterior (R$):", min_value=0.0, value=1.95, step=0.01, key="cpc_anterior_tab9")
    
    with st.expander("🎯 Dados de Produtos (Top 10)", expanded=False):
        st.info("Insira dados dos top 10 produtos (opcional)")
        
        produtos_data = []
        for i in range(1, 6):
            col_prod1, col_prod2 = st.columns(2)
            with col_prod1:
                produto_nome = st.text_input(f"Produto {i}:", value="", key=f"produto_{i}_nome_tab9")
            with col_prod2:
                if produto_nome:
                    eng_prod = st.number_input(f"Engajamento {produto_nome}:", min_value=0, value=0, step=1000, key=f"eng_prod_{i}_tab9")
                    produtos_data.append({"produto": produto_nome, "engajamento": eng_prod})
    
    st.markdown("### 📝 Contexto Adicional")
    
    contexto_yoy = st.text_area(
        "📋 Informações de contexto (opcional):",
        placeholder="Ex: Campanha de lançamento de novo produto, aumento sazonal de custos, mudança de estratégia...",
        height=100,
        key="contexto_yoy_tab9"
    )
    
    st.markdown("### 🚀 Calcular e Analisar")
    
    if st.button("📊 Calcular Análise YoY", use_container_width=True, type="primary", key="calcular_yoy_tab9"):
        with st.spinner("Calculando análise YoY..."):
            try:
                # Calcular totais
                total_invest_atual = fb_atual + tiktok_atual + display_atual + youtube_atual + pmax_atual + search_atual + outras_atual
                total_invest_anterior = fb_anterior + tiktok_anterior + display_anterior + youtube_anterior + pmax_anterior + search_anterior + outras_anterior
                
                # Calcular variações
                var_invest = total_invest_atual - total_invest_anterior
                var_invest_pct = (var_invest / total_invest_anterior * 100) if total_invest_anterior > 0 else 0
                
                var_sessoes = sessoes_atual - sessoes_anterior
                var_sessoes_pct = (var_sessoes / sessoes_anterior * 100) if sessoes_anterior > 0 else 0
                
                var_engajamento = engajamento_atual - engajamento_anterior
                var_engajamento_pct = (var_engajamento / engajamento_anterior * 100) if engajamento_anterior > 0 else 0
                
                var_tempo = tempo_medio_atual - tempo_medio_anterior
                var_tempo_pct = (var_tempo / tempo_medio_anterior * 100) if tempo_medio_anterior > 0 else 0
                
                var_views = views_atual - views_anterior
                var_views_pct = (var_views / views_anterior * 100) if views_anterior > 0 else 0
                
                var_conversoes = conversoes_atual - conversoes_anterior
                var_conversoes_pct = (var_conversoes / conversoes_anterior * 100) if conversoes_anterior > 0 else 0
                
                var_cpc = cpc_atual - cpc_anterior
                var_cpc_pct = (var_cpc / cpc_anterior * 100) if cpc_anterior > 0 else 0
                
                # Preparar dados para Gemini
                yoy_data = f"""
                # 📊 DADOS YoY - {cliente_yoy}
                
                ## 📅 PERÍODO:
                - Mês: {mes_yoy}
                - Ano Atual: {ano_atual} vs Ano Anterior: {ano_anterior}
                - Período: {periodo_yoy}
                
                ## 💰 INVESTIMENTO POR PLATAFORMA:
                
                ### ANO {ano_anterior} ({mes_yoy}):
                - FB/Instagram: R$ {fb_anterior:,.2f}
                - TikTok: R$ {tiktok_anterior:,.2f}
                - Display: R$ {display_anterior:,.2f}
                - YouTube: R$ {youtube_anterior:,.2f}
                - PMax: R$ {pmax_anterior:,.2f}
                - Search: R$ {search_anterior:,.2f}
                - Outras: R$ {outras_anterior:,.2f}
                - **TOTAL ANTERIOR: R$ {total_invest_anterior:,.2f}**
                
                ### ANO {ano_atual} ({mes_yoy}):
                - FB/Instagram: R$ {fb_atual:,.2f}
                - TikTok: R$ {tiktok_atual:,.2f}
                - Display: R$ {display_atual:,.2f}
                - YouTube: R$ {youtube_atual:,.2f}
                - PMax: R$ {pmax_atual:,.2f}
                - Search: R$ {search_atual:,.2f}
                - Outras: R$ {outras_atual:,.2f}
                - **TOTAL ATUAL: R$ {total_invest_atual:,.2f}**
                
                ## 📈 VARIAÇÃO INVESTIMENTO:
                - Variação Total: R$ {var_invest:,.2f} ({var_invest_pct:.1f}%)
                
                ## 📊 PERFORMANCE:
                
                ### SESSÕES:
                - {ano_anterior}: {sessoes_anterior:,}
                - {ano_atual}: {sessoes_atual:,}
                - Variação: {var_sessoes:+,} ({var_sessoes_pct:+.1f}%)
                
                ### TEMPO MÉDIO:
                - {ano_anterior}: {tempo_medio_anterior:.2f} min
                - {ano_atual}: {tempo_medio_atual:.2f} min
                - Variação: {var_tempo:+.2f} min ({var_tempo_pct:+.1f}%)
                
                ### ENGAJAMENTO:
                - {ano_anterior}: {engajamento_anterior:,}
                - {ano_atual}: {engajamento_atual:,}
                - Variação: {var_engajamento:+,} ({var_engajamento_pct:+.1f}%)
                
                ### VIEWS:
                - {ano_anterior}: {views_anterior:,}
                - {ano_atual}: {views_atual:,}
                - Variação: {var_views:+,} ({var_views_pct:+.1f}%)
                
                ### CONVERSÕES:
                - {ano_anterior}: {conversoes_anterior:,}
                - {ano_atual}: {conversoes_atual:,}
                - Variação: {var_conversoes:+,} ({var_conversoes_pct:+.1f}%)
                
                ### CPC (Custo por Clique):
                - {ano_anterior}: R$ {cpc_anterior:.2f}
                - {ano_atual}: R$ {cpc_atual:.2f}
                - Variação: R$ {var_cpc:+.2f} ({var_cpc_pct:+.1f}%)
                """
                
                if produtos_data:
                    yoy_data += "\n\n## 🏆 TOP PRODUTOS (por engajamento):\n"
                    for prod in produtos_data:
                        yoy_data += f"- **{prod['produto']}**: {prod['engajamento']:,} engajamentos\n"
                
                st.session_state.yoy_analysis = yoy_data
                st.success("✅ Dados YoY calculados!")
                
            except Exception as e:
                st.error(f"❌ Erro ao calcular YoY: {str(e)[:200]}")
    
    if st.session_state.yoy_analysis:
        yoy_data = st.session_state.yoy_analysis
        
        st.markdown("### 📋 Resumo dos Dados")
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        with col_sum1:
            # Extrair totais do texto
            import re
            
            total_atual_match = re.search(r'TOTAL ATUAL: R\$ ([\d,]+\.\d{2})', yoy_data)
            total_anterior_match = re.search(r'TOTAL ANTERIOR: R\$ ([\d,]+\.\d{2})', yoy_data)
            
            if total_atual_match and total_anterior_match:
                total_atual = float(total_atual_match.group(1).replace(',', ''))
                total_anterior = float(total_anterior_match.group(1).replace(',', ''))
                var_total = total_atual - total_anterior
                var_total_pct = (var_total / total_anterior * 100) if total_anterior > 0 else 0
                
                safe_metric("Investimento Atual", format_currency(total_atual))
                safe_metric("Investimento Anterior", format_currency(total_anterior))
                safe_metric("Variação Investimento", format_currency(var_total), f"{var_total_pct:.1f}%")
        
        with col_sum2:
            # Extrair sessões
            sessoes_match = re.search(r'SESSÕES:\s*\n.*?' + str(ano_anterior) + r': ([\d,]+)\s*\n.*?' + str(ano_atual) + r': ([\d,]+)', yoy_data, re.DOTALL)
            if sessoes_match:
                sessoes_ant = int(sessoes_match.group(1).replace(',', ''))
                sessoes_at = int(sessoes_match.group(2).replace(',', ''))
                var_sess = sessoes_at - sessoes_ant
                var_sess_pct = (var_sess / sessoes_ant * 100) if sessoes_ant > 0 else 0
                
                safe_metric("Sessões Atual", f"{sessoes_at:,}")
                safe_metric("Sessões Anterior", f"{sessoes_ant:,}")
                safe_metric("Variação Sessões", f"{var_sess:+,}", f"{var_sess_pct:+.1f}%")
        
        with col_sum3:
            # Extrair engajamento
            eng_match = re.search(r'ENGAJAMENTO:\s*\n.*?' + str(ano_anterior) + r': ([\d,]+)\s*\n.*?' + str(ano_atual) + r': ([\d,]+)', yoy_data, re.DOTALL)
            if eng_match:
                eng_ant = int(eng_match.group(1).replace(',', ''))
                eng_at = int(eng_match.group(2).replace(',', ''))
                var_eng = eng_at - eng_ant
                var_eng_pct = (var_eng / eng_ant * 100) if eng_ant > 0 else 0
                
                safe_metric("Engajamento Atual", f"{eng_at:,}")
                safe_metric("Engajamento Anterior", f"{eng_ant:,}")
                safe_metric("Variação Engajamento", f"{var_eng:+,}", f"{var_eng_pct:+.1f}%")
        
        st.markdown("### 📊 Tabela de Comparativo")
        
        # Criar tabela comparativa
        comparativo_data = {
            'Métrica': ['Investimento Total', 'Sessões', 'Tempo Médio', 'Engajamento', 'Views', 'Conversões', 'CPC'],
            f'{ano_anterior}': [
                format_currency(total_anterior) if 'total_anterior' in locals() else 'N/A',
                f"{sessoes_anterior:,}" if 'sessoes_anterior' in locals() else 'N/A',
                f"{tempo_medio_anterior:.2f} min" if 'tempo_medio_anterior' in locals() else 'N/A',
                f"{engajamento_anterior:,}" if 'engajamento_anterior' in locals() else 'N/A',
                f"{views_anterior:,}" if 'views_anterior' in locals() else 'N/A',
                f"{conversoes_anterior:,}" if 'conversoes_anterior' in locals() else 'N/A',
                f"R$ {cpc_anterior:.2f}" if 'cpc_anterior' in locals() else 'N/A'
            ],
            f'{ano_atual}': [
                format_currency(total_atual) if 'total_atual' in locals() else 'N/A',
                f"{sessoes_atual:,}" if 'sessoes_atual' in locals() else 'N/A',
                f"{tempo_medio_atual:.2f} min" if 'tempo_medio_atual' in locals() else 'N/A',
                f"{engajamento_atual:,}" if 'engajamento_atual' in locals() else 'N/A',
                f"{views_atual:,}" if 'views_atual' in locals() else 'N/A',
                f"{conversoes_atual:,}" if 'conversoes_atual' in locals() else 'N/A',
                f"R$ {cpc_atual:.2f}" if 'cpc_atual' in locals() else 'N/A'
            ],
            'Variação': [
                f"{var_invest_pct:+.1f}%" if 'var_invest_pct' in locals() else 'N/A',
                f"{var_sessoes_pct:+.1f}%" if 'var_sessoes_pct' in locals() else 'N/A',
                f"{var_tempo_pct:+.1f}%" if 'var_tempo_pct' in locals() else 'N/A',
                f"{var_engajamento_pct:+.1f}%" if 'var_engajamento_pct' in locals() else 'N/A',
                f"{var_views_pct:+.1f}%" if 'var_views_pct' in locals() else 'N/A',
                f"{var_conversoes_pct:+.1f}%" if 'var_conversoes_pct' in locals() else 'N/A',
                f"{var_cpc_pct:+.1f}%" if 'var_cpc_pct' in locals() else 'N/A'
            ]
        }
        
        df_comparativo = pd.DataFrame(comparativo_data)
        st.dataframe(df_comparativo, use_container_width=True)
        
        st.markdown("### 📈 Gráfico de Comparativo")
        
        # Gráfico de barras para métricas principais
        metricas_grafico = ['Investimento', 'Sessões', 'Engajamento']
        valores_anterior = [total_anterior/1000, sessoes_anterior/1000, engajamento_anterior/1000000] if all(var in locals() for var in ['total_anterior', 'sessoes_anterior', 'engajamento_anterior']) else [0, 0, 0]
        valores_atual = [total_atual/1000, sessoes_atual/1000, engajamento_atual/1000000] if all(var in locals() for var in ['total_atual', 'sessoes_atual', 'engajamento_atual']) else [0, 0, 0]
        
        fig_yoy = go.Figure()
        
        fig_yoy.add_trace(go.Bar(
            name=f'{ano_anterior}',
            x=metricas_grafico,
            y=valores_anterior,
            marker_color='#6366f1'
        ))
        
        fig_yoy.add_trace(go.Bar(
            name=f'{ano_atual}',
            x=metricas_grafico,
            y=valores_atual,
            marker_color='#10b981'
        ))
        
        fig_yoy.update_layout(
            title=f"Comparativo YoY - {mes_yoy}",
            barmode='group',
            xaxis_title="Métrica",
            yaxis_title="Valor",
            height=500
        )
        
        # Adicionar anotações com variação percentual
        for i, (ant, at) in enumerate(zip(valores_anterior, valores_atual)):
            if ant > 0:
                var_pct = ((at - ant) / ant * 100)
                fig_yoy.add_annotation(
                    x=i,
                    y=max(ant, at) * 1.05,
                    text=f"{var_pct:+.1f}%",
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
        
        st.plotly_chart(fig_yoy, use_container_width=True)
        
        st.markdown("### 🤖 Gerar Análise com Gemini")
        
        if modelo_texto:
            if st.button("📄 Gerar Análise Completa", use_container_width=True, type="primary", key="gerar_analise_yoy_tab9"):
                with st.spinner("Gerando análise YoY com Gemini..."):
                    try:
                        analysis_result = generate_yoy_analysis(yoy_data, contexto_yoy)
                        
                        st.markdown("### 📄 Análise YoY Completa")
                        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
                        st.markdown(analysis_result)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Botão para download
                        st.download_button(
                            label="💾 Baixar Análise YoY",
                            data=analysis_result,
                            file_name=f"analise_yoy_{cliente_yoy}_{mes_yoy}_{ano_atual}_vs_{ano_anterior}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key="download_yoy_tab9"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar análise: {str(e)[:200]}")

# =============================================================================
# TAB 10: DADOS COLADOS
# =============================================================================

with tab10:
    st.markdown('<div class="pasted-data"><h2>📋 Análise de Dados Colados</h2></div>', unsafe_allow_html=True)
    
    st.markdown("### 📝 Cole seus dados aqui")
    
    col_paste1, col_paste2 = st.columns([2, 1])
    
    with col_paste1:
        pasted_text = st.text_area(
            "Cole dados numéricos, tabelas, ou qualquer informação para análise:",
            height=300,
            placeholder="Exemplo:\n\nINVESTIMENTO MENSAL OUTUBRO\nFERRAMENTA FB INSTA TIKTOK DISPLAY Youtube PMax Total\nINVESTIMENTO R$ 60,357 R$ 121,923 R$ 33,004 R$ 58,313 R$ 60,379 R$ 17 R$ 333,992\nPORCENTAGEM 18.07% 36.50% 9.88% 17.46% 18.08% 0.01% 100.00%\n\nINVESTIMENTO MENSAL NOVEMBRO\nFERRAMENTA FB INSTA TIKTOK DISPLAY Youtube PMax Search Total\nINVESTIMENTO R$ 60.817 R$ 92.307 R$ 45.831 R$ 93.618 R$ 81.890 R$ 14.632 R$ 3.451,15 R$ 389.095\nPORCENTAGEM 15,63% 23,5% 11,6% 23,8% 20,8% 3,7% 0,9% 100.00%",
            key="pasted_text_tab10"
        )
    
    with col_paste2:
        st.markdown("### ⚙️ Configuração")
        
        analysis_type_paste = st.selectbox(
            "Tipo de Análise:",
            options=["overall", "financial", "performance", "comparative", "insights"],
            format_func=lambda x: {
                "overall": "📊 Análise Geral",
                "financial": "💰 Análise Financeira",
                "performance": "📈 Análise de Performance",
                "comparative": "🔄 Análise Comparativa",
                "insights": "🔍 Extrair Insights"
            }[x],
            key="analysis_type_paste_tab10"
        )
        
        context_paste = st.text_area(
            "Contexto (opcional):",
            height=100,
            placeholder="Ex: Dados de campanhas de Novembro 2025, análise de performance por plataforma...",
            key="context_paste_tab10"
        )
    
    st.markdown("### 🚀 Analisar Dados")
    
    if st.button("🤖 Analisar Dados Colados", use_container_width=True, type="primary", key="analisar_paste_tab10"):
        if not pasted_text or pasted_text.strip() == "":
            st.error("❌ Cole alguns dados para análise")
        elif not modelo_texto:
            st.error("❌ Gemini não está configurado")
        else:
            with st.spinner("Analisando dados colados..."):
                try:
                    analysis_result, structure_result = analyze_pasted_data(
                        pasted_text, 
                        analysis_type_paste, 
                        context_paste
                    )
                    
                    st.session_state.pasted_data_analysis = analysis_result
                    st.session_state.pasted_structure = structure_result
                    
                    st.success("✅ Análise concluída!")
                    
                except Exception as e:
                    st.error(f"❌ Erro na análise: {str(e)[:200]}")
    
    if st.session_state.pasted_data_analysis:
        analysis_result = st.session_state.pasted_data_analysis
        structure_result = st.session_state.pasted_structure if 'pasted_structure' in st.session_state else None
        
        st.markdown("### 🏗️ Estrutura Identificada")
        
        if structure_result:
            with st.expander("Ver estrutura dos dados"):
                st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
                st.markdown(structure_result)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📄 Análise Completa")
        
        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
        st.markdown(analysis_result)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_actions1, col_actions2, col_actions3 = st.columns(3)
        
        with col_actions1:
            st.download_button(
                label="💾 Baixar Análise",
                data=analysis_result,
                file_name=f"analise_dados_colados_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_analysis_paste_tab10"
            )
        
        with col_actions2:
            if structure_result:
                st.download_button(
                    label="📊 Baixar Estrutura",
                    data=structure_result,
                    file_name=f"estrutura_dados_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_structure_paste_tab10"
                )
        
        with col_actions3:
            if st.button("🔄 Nova Análise", use_container_width=True, key="nova_analise_paste_tab10"):
                st.session_state.pasted_data_analysis = None
                st.session_state.pasted_structure = None
                st.rerun()

# =============================================================================
# RODAPÉ
# =============================================================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if not df_filtrado.empty:
        st.caption(f"📊 Dados: {len(df_filtrado):,} registros")
        if st.session_state.filtros_aplicados:
            filtros_count = len(st.session_state.filtros_aplicados)
            st.caption(f"🔍 Filtros: {filtros_count} ativos")

with footer_col2:
    if 'campaign' in df_filtrado.columns:
        try:
            num_campaigns = df_filtrado['campaign'].nunique()
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
