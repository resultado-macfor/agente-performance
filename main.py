# app_completo.py - App Analytics Platform Completo com Classificador Multi-Clientes e Sistema de Login
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
import hashlib
import hmac
import base64
import time

# Tentar importar Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# =============================================================================
# CONFIGURAÇÃO DE AUTENTICAÇÃO
# =============================================================================

# Credenciais de usuário (em produção, use uma base de dados ou variáveis de ambiente)
USERS = {
    "admin": {
        "password_hash": "fc7d5e82c7a8e9b8b5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c",  # senha1234 em hash
        "role": "admin",
        "name": "Administrador"
    }
}

# Função para verificar senha (simplificada para exemplo)
def verify_password(input_password, stored_hash):
    # Em produção, use bcrypt ou similar
    # Aqui usamos uma verificação simples para demonstração
    expected_hash = hashlib.sha256(f"admin{input_password}".encode()).hexdigest()
    return hmac.compare_digest(expected_hash, stored_hash)

# Inicializar estado de autenticação
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.login_attempts = 0
    st.session_state.last_login_attempt = 0

# =============================================================================
# PÁGINA DE LOGIN
# =============================================================================

def login_page():
    st.set_page_config(
        layout="wide",
        page_title="Login - Agente Performance",
        page_icon="🔐"
    )
    
    # CSS personalizado para página de login
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .login-header h1 {
            color: #4f46e5;
            margin-bottom: 10px;
        }
        .login-form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .login-input {
            padding: 12px 15px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        .login-input:focus {
            outline: none;
            border-color: #4f46e5;
        }
        .login-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .login-button:hover {
            transform: translateY(-2px);
        }
        .login-error {
            background: #fee2e2;
            color: #dc2626;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;
            display: none;
        }
        .login-error.show {
            display: block;
        }
        .watermark {
            position: fixed;
            bottom: 20px;
            right: 20px;
            color: #9ca3af;
            font-size: 12px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Container principal
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Cabeçalho
    st.markdown('<div class="login-header">', unsafe_allow_html=True)
    st.markdown('<h1>🔐 Agente Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p>Acesso Restrito à Equipe Autorizada</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Formulário de login
    with st.form("login_form"):
        username = st.text_input("👤 Usuário", placeholder="Digite seu nome de usuário")
        password = st.text_input("🔒 Senha", placeholder="Digite sua senha", type="password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("🚀 Entrar", use_container_width=True)
        with col2:
            remember_me = st.checkbox("Lembrar-me")
        
        # Verificar tentativas recentes
        current_time = time.time()
        if st.session_state.login_attempts >= 3:
            time_since_last = current_time - st.session_state.last_login_attempt
            if time_since_last < 300:  # 5 minutos de bloqueio
                st.error(f"⚠️ Muitas tentativas falhas. Tente novamente em {int((300 - time_since_last)/60)} minutos.")
                st.stop()
        
        if submit_button:
            if username in USERS:
                if verify_password(password, USERS[username]["password_hash"]):
                    # Login bem-sucedido
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = USERS[username]["role"]
                    st.session_state.login_attempts = 0
                    
                    if remember_me:
                        # Em produção, use cookies seguros
                        st.success("✅ Login realizado com sucesso!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.success("✅ Login realizado com sucesso!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                else:
                    # Senha incorreta
                    st.session_state.login_attempts += 1
                    st.session_state.last_login_attempt = current_time
                    st.error(f"❌ Senha incorreta. Tentativa {st.session_state.login_attempts} de 3.")
            else:
                # Usuário não encontrado
                st.session_state.login_attempts += 1
                st.session_state.last_login_attempt = current_time
                st.error(f"❌ Usuário não encontrado. Tentativa {st.session_state.login_attempts} de 3.")
    
    # Informações de ajuda
    st.markdown("---")
    with st.expander("ℹ️ Precisa de ajuda?"):
        st.info("""
        **Credenciais padrão:**
        - 👤 Usuário: admin
        - 🔒 Senha: senha1234
        
        **Notas de segurança:**
        - Após 3 tentativas falhas, o acesso será bloqueado por 5 minutos
        - Não compartilhe suas credenciais
        - Em produção, use autenticação de dois fatores
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Watermark
    st.markdown('<div class="watermark">🔒 Sistema Seguro v1.0</div>', unsafe_allow_html=True)

# =============================================================================
# VERIFICAR AUTENTICAÇÃO
# =============================================================================

def check_auth():
    if not st.session_state.authenticated:
        login_page()
        st.stop()
    return True

# =============================================================================
# APLICAÇÃO PRINCIPAL (APÓS LOGIN)
# =============================================================================

def main_app():
    # Configuração da página principal
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
        .user-info {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #0ea5e9;
        }
        .logout-button {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 8px 20px !important;
            font-weight: 500 !important;
            border: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # =============================================================================
    # BARRA SUPERIOR COM INFORMAÇÕES DO USUÁRIO
    # =============================================================================
    
    col_user, col_logout = st.columns([3, 1])
    
    with col_user:
        st.markdown(f"""
        <div class="user-info">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0;">👤 {USERS[st.session_state.username]['name']}</h4>
                    <p style="margin: 0; font-size: 0.9em; color: #6b7280;">
                        Logado como: <strong>{st.session_state.username}</strong> | 
                        Perfil: <span style="color: #4f46e5;">{st.session_state.role.upper()}</span> | 
                        Sessão iniciada: {datetime.now().strftime('%H:%M')}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_logout:
        if st.button("🚪 Sair", key="logout_button", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()
    
    # Título principal
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
            
            # =====================================================================
            # FUNÇÃO ATUALIZADA PARA IDENTIFICAR CLIENTE
            # =====================================================================
            def identificar_cliente_por_account_name(account_name):
                if pd.isna(account_name):
                    return "Outros"
                
                account_str = str(account_name).strip()
                account_upper = account_str.upper()
                
                # Regra Especial: Agrupamento Syngenta
                if "SYNGENTA" in account_upper:
                    return "Syngenta"
                
                # Retorna o nome exato para os outros (ou você pode mapear nomes específicos aqui)
                return account_str
            
            if 'account_name' in df.columns:
                df['cliente_identificado'] = df['account_name'].apply(identificar_cliente_por_account_name)
                
                # Aplicar filtro de cliente baseado no dropdown da sidebar
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
    # FUNÇÕES PARA CLASSIFICADOR DE CAMPANHAS MULTI-CLIENTES
    # =============================================================================

    def extrair_categorias_campanha(nome_campanha):
        """Extrai categorias de campanha usando regex e análise de padrões"""
        if not nome_campanha or pd.isna(nome_campanha):
            return {}
        
        nome_str = str(nome_campanha).upper()
        
        # Inicializar dicionário de resultados
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
        
        # Padrões comuns para identificar componentes
        padroes = {
            # Etapas do funil (em várias línguas)
            'etapa_funil': [
                'UP', 'MID', 'LOWER', 'TOF', 'MOF', 'BOF', 'TOP', 'MIDDLE', 'BOTTOM',
                'AWARENESS', 'CONSIDERATION', 'CONVERSION', 'RETENTION',
                'DESCOBERTA', 'CONSIDERACAO', 'CONVERSAO', 'RETENCAO'
            ],
            
            # Tipos de campanha
            'tipo_campanha': [
                'VIDEO', 'DISPLAY', 'SEARCH', 'SOCIAL', 'EMAIL', 'SMS', 'PUSH',
                'NATIVO', 'NATIVE', 'PROGRAMATICA', 'PROGRAMMATIC',
                'PERFORMANCE', 'BRANDING', 'BRAND', 'DIRECT', 'DIRECT_RESPONSE'
            ],
            
            # Objetivos
            'objetivo': [
                'AWARENESS', 'CONSIDERATION', 'CONVERSION', 'LEAD', 'SALES',
                'TRAFFIC', 'ENGAGEMENT', 'INSTALL', 'VIEWS', 'CLICKS',
                'ALCANCE', 'CONVERSAO', 'LEADS', 'VENDAS', 'TRAFEGO',
                'ENGAJAMENTO', 'INSTALACOES', 'VISUALIZACOES', 'CLIQUES'
            ],
            
            # Plataformas
            'plataforma': [
                'GOOGLE', 'FACEBOOK', 'INSTAGRAM', 'TIKTOK', 'LINKEDIN', 'TWITTER',
                'YOUTUBE', 'PINTEREST', 'SNAPCHAT', 'META', 'TIKTOK', 'BING',
                'DV360', 'TRADEDESK', 'AMAZON', 'APPLE', 'SPOTIFY'
            ],
            
            # Agências (adicionar mais conforme necessário)
            'agencia': [
                'MACFOR', 'OGILVY', 'PUBLICIS', 'WPP', 'OMNICOM', 'DENTSU',
                'HAVAS', 'IPG', 'ACCENTURE', 'DELOITTE', 'PWC', 'KPMG'
            ],
            
            # Culturas/produtos agrícolas
            'cultura': [
                'SOJA', 'MILHO', 'CAFE', 'ALGODAO', 'CANADEACUCAR', 'CANA',
                'TRIGO', 'ARROZ', 'FEIJAO', 'MANDIOCA', 'LARANJA', 'UVA',
                'TOMATE', 'BATATA', 'CEVADA', 'AVEIA', 'GIRASSOL'
            ],
            
            # Marcas/Produtos genéricos
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
        
        # Identificar PO (Purchase Order)
        po_pattern = r'\bPO[_-]?(\d+)\b'
        po_match = re.search(po_pattern, nome_str, re.IGNORECASE)
        if po_match:
            categorias['po'] = f"PO{po_match.group(1)}"
        
        # Identificar agência
        for agencia in padroes['agencia']:
            if agencia in nome_str:
                categorias['agencia'] = agencia
                break
        
        # Identificar plataforma
        for plataforma in padroes['plataforma']:
            if plataforma in nome_str:
                categorias['plataforma'] = plataforma
                break
        
        # Identificar cultura
        for cultura in padroes['cultura']:
            if cultura in nome_str:
                categorias['cultura'] = cultura
                break
        
        # Identificar produto
        for produto in padroes['produto']:
            if produto in nome_str:
                categorias['produto'] = produto
                break
        
        # Identificar tipo de campanha
        for tipo in padroes['tipo_campanha']:
            if tipo in nome_str:
                categorias['tipo_campanha'] = tipo
                break
        
        # Identificar objetivo
        for objetivo in padroes['objetivo']:
            if objetivo in nome_str:
                categorias['objetivo'] = objetivo
                break
        
        # Identificar etapa do funil
        for etapa in padroes['etapa_funil']:
            if etapa in nome_str:
                categorias['etapa_funil'] = etapa
                break
        
        # Tentar identificar iniciativa (primeira parte antes de separadores comuns)
        separadores = ['_', '-', '|', ' ', '__']
        
        # Extrair possíveis iniciativas
        for sep in separadores:
            if sep in nome_str:
                partes = nome_str.split(sep)
                if len(partes) > 0:
                    # Primeira parte como possível iniciativa
                    primeira_parte = partes[0]
                    if len(primeira_parte) > 3 and primeira_parte not in padroes['plataforma']:
                        categorias['iniciativa'] = primeira_parte
        
        # Identificar cliente (se houver prefixo ou sufixo específico)
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
            
            # Determinar se foi classificado (pelo menos 3 categorias identificadas)
            categorias_preenchidas = sum(1 for v in categorias.values() if v is not None)
            classificado = 'SIM' if categorias_preenchidas >= 3 else 'NÃO'
            
            classificacao = {
                'nome_campanha_original': nome_campanha,
                'classificado': classificado,
                'categorias_identificadas': categorias_preenchidas
            }
            
            # Adicionar todas as categorias
            for chave, valor in categorias.items():
                classificacao[f'campaign_{chave}'] = valor
            
            classificacoes.append(classificacao)
        
        df_classificado = pd.DataFrame(classificacoes)
        
        # Combinar com dados originais
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

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Informações do usuário na sidebar
        st.markdown(f"""
        <div style="background: #f0f9ff; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <p style="margin: 0; font-size: 0.9em;"><strong>👤 Usuário:</strong> {st.session_state.username}</p>
            <p style="margin: 0; font-size: 0.9em;"><strong>🏷️ Perfil:</strong> {st.session_state.role}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Testar conexão
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
        
        # Data sources
        st.subheader("📱 Data Sources")    
        data_sources_opcoes = ["facebook", "google ads", "tiktok", "linkedin", "twitter", "pinterest"]
        selected_sources = st.multiselect(
            "Data Sources",
            options=data_sources_opcoes,
            default=data_sources_opcoes[:3]
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
        
        # Limite - CORREÇÃO DO ERRO: usar valor padrão se df vazio
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
                        filtro_cliente=filtro_cliente,
                        limit=limite
                    )
                    
                    if not df.empty:
                        st.session_state.df_completo = df
                        st.session_state.colunas_numericas = identificar_colunas_numericas(df)
                        
                        # Classificar campanhas automaticamente
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

    # =============================================================================
    # SEÇÃO PRINCIPAL (mostrar apenas se houver dados)
    # =============================================================================

    if df.empty:
        st.warning("📭 Nenhum dado carregado. Use o botão na sidebar para carregar dados.")
        st.stop()

    # =============================================================================
    # SEÇÃO DE FILTROS MULTI-CLIENTE (ACIMA DAS ABAS)
    # =============================================================================

    st.markdown("## 🔍 Filtros Avançados por Categoria de Campanha")

    # Criar colunas para filtros
    filtro_col1, filtro_col2, filtro_col3, filtro_col4 = st.columns(4)



    with filtro_col1:
        # Filtro por Produto 
        if 'campaign_produto' in df_classificado.columns:
            # Lista completa de produtos
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
        
        # Filtro por Cultura
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
        # Filtro por Tipo de Campanha
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
        
        # Filtro por Objetivo
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
        # Filtro por Etapa do Funil
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
        
        # Filtro por Iniciativa
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
        # Filtro por Plataforma
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

        # Filtro por Agência
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

    # Botões de ação para filtros
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

    with col_btn1:
        if st.button("✅ Aplicar Filtros", use_container_width=True):
            st.rerun()

    with col_btn2:
        if st.button("🔄 Limpar Filtros", use_container_width=True):
            st.session_state.filtros_aplicados = {}
            st.rerun()


    with col_btn3:
        # Adicionar campo de busca por nome de campanha
        busca_campanha = st.text_input(
            "",
            placeholder="Digite parte do nome da campanha...",
            key="busca_campanha_input",
            label_visibility="collapsed"  # Esconde completamente o label
        )
        
        # Armazenar busca no session_state
        if busca_campanha:
            st.session_state.busca_campanha = busca_campanha
        elif 'busca_campanha' not in st.session_state:
            st.session_state.busca_campanha = ""




    # Aplicar filtros aos dados
    df_filtrado = df_classificado.copy()
    if st.session_state.filtros_aplicados:
        for coluna, valor in st.session_state.filtros_aplicados.items():
            if coluna in df_filtrado.columns:
                df_filtrado = df_filtrado[df_filtrado[coluna] == valor]


    # Depois aplicar busca por nome da campanha (se existir)
    if 'busca_campanha' in st.session_state and st.session_state.busca_campanha:
        busca_termo = st.session_state.busca_campanha.lower().strip()
        if busca_termo and 'campaign' in df_filtrado.columns:
            # Buscar em qualquer parte do nome da campanha (case insensitive)
            df_filtrado = df_filtrado[
                df_filtrado['campaign'].astype(str).str.lower().str.contains(busca_termo, na=False)
            ]

    # Mostrar status dos filtros e busca
    filtros_ativos = []


    if st.session_state.filtros_aplicados:
        filtros_ativos.extend([f"{k.replace('campaign_', '')}: {v}" for k, v in st.session_state.filtros_aplicados.items()])

    if 'busca_campanha' in st.session_state and st.session_state.busca_campanha:
        filtros_ativos.append(f"Busca: '{st.session_state.busca_campanha}'")

    if filtros_ativos:
        st.markdown(f"### 📊 Dados Filtrados: {len(df_filtrado):,} registros")
        filtros_texto = " | ".join(filtros_ativos)
        st.info(f"**Filtros ativos:** {filtros_texto}")
        
        # Mostrar resumo dos filtros em badges
        st.markdown("**Filtros aplicados:**")
        col_badges = st.columns(min(8, len(filtros_ativos)))
        for idx, filtro in enumerate(filtros_ativos):
            with col_badges[idx % 8]:
                st.markdown(f'<span style="background:#e0f2fe; padding:5px 10px; border-radius:10px; margin:2px; font-size:0.8em">{filtro}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f"### 📊 Dados Completos: {len(df_filtrado):,} registros")
        st.info("ℹ️ Nenhum filtro aplicado. Todos os dados estão visíveis.")

    # Abas principais (usando df_filtrado em vez de df)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Visão Geral", 
        "📈 Análise Numérica", 
        "🔍 Explorar Colunas", 
        "📊 Visualizar Dados",
        "🎯 Performance",
        "🤖 Análise com IA",
        "🎪 Classificador Campanhas"
    ])

    # =============================================================================
    # TAB 1: VISÃO GERAL (COM DADOS FILTRADOS)
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
        
        # Mostrar informações
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
    # TAB 2: ANÁLISE NUMÉRICA (COM DADOS FILTRADOS)
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
                # Estatísticas
                st.subheader("📊 Estatísticas Descritivas")
                
                # Converter para numérico
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
                            # Criar gráfico sem mostrar mensagem de erro
                            fig = criar_visualizacao_coluna(df_filtrado, col)
                            if fig is not None and fig:
                                # Adicionar key única para cada gráfico
                                st.plotly_chart(
                                    fig, 
                                    use_container_width=True,
                                    key=f"histogram_{col}_{idx}"
                                )


                # Correlações - CORREÇÃO: Adicionar key única
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
    # TAB 3: EXPLORAR COLUNAS (COM DADOS FILTRADOS)
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
                
                # Visualização
                st.subheader("📊 Visualização")
                fig = criar_visualizacao_coluna(df_filtrado, coluna_selecionada)
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
                
                # Distribuição
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
    #=============================================================================
    # TAB 4: VISUALIZAR DADOS (COM DADOS FILTRADOS)
    # =============================================================================

    with tab4:
        st.header("📊 Visualizar Dados Completos")
        
        # Selecionar colunas
        colunas_vis = st.multiselect(
            "Selecione colunas para visualizar",
            options=sorted(df_filtrado.columns),
            default=sorted(df_filtrado.columns)[:min(10, len(df_filtrado.columns))],
            key="colunas_vis_tab4"
        )
        
        if colunas_vis:
            # Filtros adicionais dentro da tab
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
            
            # Mostrar dados
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
                
                # Calcular índice
                start_idx = (page_number - 1) * limite_linhas
                end_idx = min(start_idx + limite_linhas, len(df_filtrado_tab4))
                
                # Formatar DataFrame
                df_display = df_filtrado_tab4[colunas_vis].iloc[start_idx:end_idx].copy()
                
                # Formatar números e datas
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
            
            # Download
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
    # TAB 5: PERFORMANCE (COM DADOS FILTRADOS)
    # =============================================================================

    with tab5:
        st.header("🎯 Análise de Performance")
        
        if 'campaign' not in df_filtrado.columns:
            st.error("❌ Coluna 'campaign' não encontrada.")
        else:
            # Métricas gerais
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
                        # Garantir que é datetime
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
            
            # Análise por campanha
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
            
            # Métricas financeiras
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
            
            # Análise por categoria (se disponível)
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
    # TAB 6: ANÁLISE COM IA (COM DADOS FILTRADOS)
    # =============================================================================

    with tab6:
        st.header("🤖 Análise com Gemini IA")
        
        if not modelo_texto:
            st.error("❌ Gemini não configurado!")
            st.info("Configure a chave do Gemini nas variáveis de ambiente ou secrets.")
            st.stop()
        
        if df_filtrado.empty:
            st.warning("📭 Nenhum dado carregado.")
            st.stop()
        
        # Filtros para análise
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
                        # Garantir que a coluna date é datetime
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
                            st.info("Sem datas disponíveis")
                            date_range = None
                    except Exception as e:
                        st.error(f"Erro com datas: {str(e)[:100]}")
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
                
                # CORREÇÃO DO ERRO: usar valor seguro para o slider
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
        
        # Aplicar filtros adicionais
        df_filtered_ia = df_filtrado.copy()
        
        # Filtro por datasource
        if selected_ds and 'datasource' in df_filtered_ia.columns and len(selected_ds) > 0:
            df_filtered_ia = df_filtered_ia[df_filtered_ia['datasource'].isin(selected_ds)]
        
        # Filtro por data
        if date_range and len(date_range) == 2 and 'date' in df_filtered_ia.columns:
            start_date, end_date = date_range
            
            # Garantir que a coluna date é datetime
            if not pd.api.types.is_datetime64_any_dtype(df_filtered_ia['date']):
                df_filtered_ia['date'] = pd.to_datetime(df_filtered_ia['date'], errors='coerce')
            
            # Filtrar apenas datas válidas
            mask = df_filtered_ia['date'].notna()
            
            # Converter start_date e end_date para datetime
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            
            # Aplicar filtro de data
            df_filtered_ia = df_filtered_ia[
                mask & 
                (df_filtered_ia['date'] >= start_dt) & 
                (df_filtered_ia['date'] <= end_dt)
            ]
        
        # Filtro por campanha
        if selected_campaigns and 'campaign' in df_filtered_ia.columns and len(selected_campaigns) > 0:
            df_filtered_ia = df_filtered_ia[df_filtered_ia['campaign'].isin(selected_campaigns)]
        
        # Limitar registros
        df_filtered_ia = df_filtered_ia.head(max_records)
        
        # Estatísticas
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
                    # Garantir que é datetime
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
            }[x],
            key="analysis_focus_tab6"
        )
        
        user_instructions = st.text_area(
            "📝 Instruções (opcional):",
            placeholder="Ex: Foque no ROI, identifique as melhores campanhas, analise tendências por data source...",
            height=100,
            key="user_instructions_tab6"
        )
        
        # Gerar análise
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
        
        # Mostrar análise
        if st.session_state.gemini_analysis:
            st.markdown("---")
            st.markdown("### 📄 Relatório de Análise")
            
            # Ações
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
                        # Gerar descrição dos slides baseada no relatório
                        slides_desc = generate_slides_description(
                            st.session_state.gemini_analysis,
                            user_instructions  # Reutilizar instruções da análise
                        )
                        
                        st.session_state.slides_description = slides_desc
                        st.success("✅ Descrição dos slides gerada!")
                        st.rerun()
            
            with col_actions3:
                if st.button("🔄 Nova Análise", use_container_width=True, key="new_analysis_tab6"):
                    st.session_state.gemini_analysis = None
                    st.session_state.slides_description = None
                    st.rerun()
        
            # Mostrar análise Gemini
            st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
            st.markdown(st.session_state.gemini_analysis)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.slides_description:
                st.markdown("---")
                st.markdown("### 🎬 Descrição para Slides")
                
                # Botão para download
                slides_text = st.session_state.slides_description
                st.download_button(
                    label="📥 Baixar Descrição dos Slides",
                    data=slides_text,
                    file_name=f"slides_descricao_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_slides_tab6"
                )
                
                # Mostrar descrição dos slides
                st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
                st.markdown(st.session_state.slides_description)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Instruções
            st.info("""
            ## 📋 Como usar:
            
            1. **Ajuste os filtros** acima
            2. **Selecione o foco** da análise
            3. **Adicione instruções** se desejar
            4. **Clique em 'Gerar Análise'**
            
            ## 🎯 Você receberá:
            
            - 📊 Resumo executivo
            - 🎯 Análise de campanhas
            - 💰 Insights financeiros
            - 📈 Tendências identificadas
            - 🚀 Recomendações acionáveis
            """)

    # =============================================================================
    # TAB 7: CLASSIFICADOR DE CAMPANHAS MULTI-CLIENTES
    # =============================================================================

    with tab7:
        st.markdown('<div class="campaign-classifier"><h2>🎪 Classificador de Campanhas Multi-Clientes</h2></div>', unsafe_allow_html=True)
        
        # Carregar dicionário de categorias
        dicionario_categorias = carregar_dicionario_categorias()
        
        col_intro1, col_intro2 = st.columns(2)
        
        with col_intro1:
            st.markdown("### 📋 Sobre o Sistema")
            st.info("""
            Este classificador analisa nomes de campanhas de **múltiplos clientes**
            identificando automaticamente categorias como:
            
            - 👥 **Cliente** (Syngenta, Bayer, Basf, etc.)
            - 🚀 **Iniciativa** (lançamento, promoção, evento)
            - 📦 **Produto** (linhas e famílias de produtos)
            - 🌱 **Cultura/Setor** (soja, milho, café, etc.)
            - 🎯 **Tipo de Campanha** (vídeo, display, search)
            - 🎯 **Objetivo** (awareness, conversão, leads)
            - 📊 **Etapa do Funil** (TOF, MOF, BOF)
            - 🖥️ **Plataforma** (Google, Facebook, Instagram)
            """)
        
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
                    
                    # Contar clientes identificados
                    if 'campaign_cliente' in df_classificado.columns:
                        clientes_unicos = df_classificado['campaign_cliente'].nunique()
                        safe_metric("Clientes", clientes_unicos)
            else:
                st.warning("Nenhuma classificação disponível")
        
        # Análise de distribuição
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
        
        # Explorador de campanhas
        st.markdown("### 🔍 Explorador de Campanhas")
        
        col_explorer1, col_explorer2 = st.columns(2)
        
        with col_explorer1:
            # Selecionar uma campanha para análise detalhada
            if 'campaign' in df_classificado.columns:
                campanhas = sorted(df_classificado['campaign'].dropna().unique())
                campanha_selecionada = st.selectbox(
                    "Selecione uma campanha para análise:",
                    options=campanhas[:100],  # Limitar para performance
                    key="campanha_selecionada_tab7"
                )
                
                if campanha_selecionada:
                    campanha_data = df_classificado[df_classificado['campaign'] == campanha_selecionada].iloc[0]
                    
                    st.markdown("#### 📋 Detalhes da Campanha")
                    st.write(f"**Nome:** {campanha_selecionada}")
                    
                    # Mostrar categorias identificadas
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
            # Estatísticas de classificação
            st.markdown("#### 📈 Estatísticas de Classificação")
            
            if 'classificado' in df_classificado.columns:
                # Gráfico de classificação
                status_counts = df_classificado['classificado'].value_counts()
                fig_status = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Status de Classificação",
                    color=status_counts.values,
                    color_discrete_sequence=['#10b981', '#ef4444']
                )
                st.plotly_chart(fig_status, use_container_width=True)
            
            # Botão para reclassificar
            if st.button("🔄 Reclassificar Campanhas", use_container_width=True, key="reclassificar_tab7"):
                with st.spinner("Reclassificando campanhas..."):
                    df_classificado_novo = classificar_campanhas_multi_cliente(df)
                    st.session_state.df_classificado = df_classificado_novo
                    st.success("✅ Campanhas reclassificadas!")
                    st.rerun()
        
        # Exportar dados classificados
        st.markdown("### 📥 Exportar Dados Classificados")
        
        if len(df_classificado) > 0:
            # Selecionar colunas para exportar
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
                # Exportar apenas as não classificadas para análise
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
        
        # Análise com Gemini sobre padrões
        if modelo_texto and len(df_classificado) > 0:
            st.markdown("### 🤖 Análise de Padrões com Gemini")
            
            if st.button("🔍 Analisar Padrões de Nomenclatura", use_container_width=True, key="analisar_padroes_tab7"):
                with st.spinner("Analisando padrões de nomenclatura..."):
                    try:
                        # Preparar amostra de campanhas
                        sample_size = min(50, len(df_classificado))
                        sample_campaigns = df_classificado['campaign'].dropna().sample(sample_size).tolist()
                        
                        # Contar clientes identificados
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
        st.caption(f"👤 {st.session_state.username}")

    # Status Gemini
    if modelo_texto:
        st.sidebar.success("✅ Gemini ativo")
    else:
        st.sidebar.info("ℹ️ Gemini inativo")

# =============================================================================
# PONTO DE ENTRADA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()
