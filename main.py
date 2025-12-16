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
import google.generativeai as genai

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

# Configuração da API do Gemini para análise
gemini_api_key = os.getenv("GEM_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    modelo_texto = genai.GenerativeModel("gemini-2.0-flash")
else:
    st.warning("GEM_API_KEY não encontrada. A funcionalidade de análise com IA estará limitada.")
    modelo_texto = None

# =============================================================================
# CONEXÃO E CARREGAMENTO - TODAS AS COLUNAS (COM VARIÁVEIS DE AMBIENTE)
# =============================================================================

@st.cache_resource
def get_bigquery_client():
    """Cria cliente BigQuery usando variáveis de ambiente"""
    try:
        # OPÇÃO 1: Variáveis de ambiente individuais
        if all(key in os.environ for key in ['type', 'project_id', 'private_key', 'client_email', 'token_uri']):
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
        
        # OPÇÃO 2: JSON string completo em variável de ambiente
        elif 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
            credentials_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
            service_account_info = json.loads(credentials_json)
        
        # OPÇÃO 3: Streamlit Secrets
        elif 'gcp_service_account' in st.secrets:
            service_account_info = dict(st.secrets["gcp_service_account"])
            if isinstance(service_account_info.get("private_key"), str):
                service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
        
        else:
            st.error("""
            ❌ Credenciais não encontradas!
            
            Configure uma das seguintes opções:
            
            1. **Variáveis de ambiente individuais**:
               - `type`
               - `project_id`
               - `private_key`
               - `client_email`
               - `token_uri`
            
            2. **JSON completo em variável de ambiente**:
               - `GOOGLE_APPLICATION_CREDENTIALS_JSON`
            
            3. **Streamlit Secrets** (no formato TOML):
               ```toml
               [gcp_service_account]
               type = "service_account"
               project_id = "seu-project"
               private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"
               client_email = "email@project.iam.gserviceaccount.com"
               token_uri = "https://oauth2.googleapis.com/token"
               ```
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
        # Primeiro, vamos descobrir quais colunas existem
        st.info("🔍 Analisando estrutura da tabela...")
        
        # Query para obter todas as colunas
        query_schema = """
        SELECT column_name, data_type 
        FROM `macfor-media-flow.ads.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'app_view_campaigns'
        """
        
        try:
            schema_df = _client.query(query_schema).to_dataframe()
            todas_colunas = schema_df['column_name'].tolist()
            st.success(f"✅ Encontradas {len(todas_colunas)} colunas na tabela")
        except Exception as schema_error:
            # Se falhar, usar colunas padrão
            st.warning(f"Não foi possível obter schema automático: {schema_error}")
            todas_colunas = "*"
        
        # Construir query dinâmica
        if isinstance(todas_colunas, list):
            colunas_query = ",\n            ".join(todas_colunas)
        else:
            colunas_query = "*"
        
        query = f"""
        SELECT 
            {colunas_query}
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
        
        st.info("📥 Carregando dados... Isso pode levar alguns instantes")
        df = _client.query(query).to_dataframe()
        
        if df.empty:
            st.warning("Nenhum dado encontrado")
            return pd.DataFrame()
        
        st.success(f"✅ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        
        # Identificar e converter colunas numéricas
        colunas_numericas = []
        colunas_texto = []
        colunas_data = []
        
        for col in df.columns:
            # Tentar inferir tipo
            try:
                # Primeiro tenta converter para numérico
                amostra = df[col].dropna().head(100)
                if len(amostra) > 0:
                    # Testa se parece numérico
                    if pd.api.types.is_numeric_dtype(df[col]):
                        colunas_numericas.append(col)
                        # Converter para numérico
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif 'date' in col.lower() or 'data' in col.lower():
                        colunas_data.append(col)
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
                    else:
                        colunas_texto.append(col)
            except:
                colunas_texto.append(col)
        
        # Log das colunas identificadas
        with st.expander("📋 Informações das Colunas"):
            st.write(f"**Numéricas ({len(colunas_numericas)}):** {', '.join(colunas_numericas[:10])}...")
            st.write(f"**Texto ({len(colunas_texto)}):** {', '.join(colunas_texto[:10])}...")
            st.write(f"**Datas ({len(colunas_data)}):** {', '.join(colunas_data)}")
        
        return df
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# FUNÇÕES ESPECÍFICAS PARA ANÁLISE DE CAMPANHAS
# =============================================================================

def load_campaign_performance_data(_client, start_date=None, end_date=None, datasources=None, campaigns=None):
    """Carrega dados específicos para análise de performance de campanhas"""
    try:
        query = """
        SELECT 
            date,
            campaign,
            datasource,
            impressions,
            clicks,
            spend,
            conversions,
            conversion_value,
            cpc,
            cpm,
            ctr,
            conversion_rate,
            roas,
            reach,
            frequency,
            video_views,
            video_view_rate,
            video_plays,
            engagements,
            engagement_rate,
            landing_page_views,
            add_to_cart,
            purchases,
            revenue
        FROM `macfor-media-flow.ads.app_view_campaigns`
        WHERE 1=1
        """
        
        conditions = []
        
        if start_date:
            conditions.append(f"DATE(date) >= DATE('{start_date}')")
        if end_date:
            conditions.append(f"DATE(date) <= DATE('{end_date}')")
        if datasources and len(datasources) > 0:
            ds_str = ", ".join([f"'{ds}'" for ds in datasources])
            conditions.append(f"datasource IN ({ds_str})")
        if campaigns and len(campaigns) > 0:
            camp_str = ", ".join([f"'{camp}'" for camp in campaigns])
            conditions.append(f"campaign IN ({camp_str})")
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY date DESC"
        
        df = _client.query(query).to_dataframe()
        
        if not df.empty:
            # Converter colunas de data
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Converter colunas numéricas
            numeric_cols = ['impressions', 'clicks', 'spend', 'conversions', 'conversion_value',
                          'cpc', 'cpm', 'ctr', 'conversion_rate', 'roas', 'reach', 'frequency',
                          'video_views', 'video_view_rate', 'video_plays', 'engagements',
                          'engagement_rate', 'landing_page_views', 'add_to_cart', 'purchases', 'revenue']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados de campanhas: {str(e)}")
        return pd.DataFrame()

def generate_campaign_analysis_with_ai(df_campaigns, selected_campaign, date_range, metric_focus, client=None):
    """Gera análise detalhada da campanha usando Gemini IA"""
    
    if df_campaigns.empty:
        return "Nenhum dado disponível para análise."
    
    # Preparar dados para análise
    campaign_data = df_campaigns[df_campaigns['campaign'] == selected_campaign].copy()
    
    if campaign_data.empty:
        return f"Nenhum dado encontrado para a campanha '{selected_campaign}'."
    
    # Ordenar por data
    if 'date' in campaign_data.columns:
        campaign_data = campaign_data.sort_values('date')
    
    # Calcular métricas resumidas
    total_spend = campaign_data['spend'].sum() if 'spend' in campaign_data.columns else 0
    total_conversions = campaign_data['conversions'].sum() if 'conversions' in campaign_data.columns else 0
    total_revenue = campaign_data['revenue'].sum() if 'revenue' in campaign_data.columns else 0
    total_impressions = campaign_data['impressions'].sum() if 'impressions' in campaign_data.columns else 0
    total_clicks = campaign_data['clicks'].sum() if 'clicks' in campaign_data.columns else 0
    
    avg_cpc = campaign_data['cpc'].mean() if 'cpc' in campaign_data.columns else 0
    avg_ctr = campaign_data['ctr'].mean() if 'ctr' in campaign_data.columns else 0
    avg_roas = campaign_data['roas'].mean() if 'roas' in campaign_data.columns else 0
    avg_conversion_rate = campaign_data['conversion_rate'].mean() if 'conversion_rate' in campaign_data.columns else 0
    
    # Calcular tendências
    if len(campaign_data) > 1 and 'date' in campaign_data.columns:
        # Última semana vs semana anterior
        latest_date = campaign_data['date'].max()
        week_ago = latest_date - timedelta(days=7)
        
        last_week_data = campaign_data[campaign_data['date'] > week_ago]
        previous_week_data = campaign_data[
            (campaign_data['date'] > week_ago - timedelta(days=14)) & 
            (campaign_data['date'] <= week_ago)
        ]
        
        if not last_week_data.empty and not previous_week_data.empty:
            last_week_spend = last_week_data['spend'].sum() if 'spend' in last_week_data.columns else 0
            previous_week_spend = previous_week_data['spend'].sum() if 'spend' in previous_week_data.columns else 0
            spend_change = ((last_week_spend - previous_week_spend) / previous_week_spend * 100) if previous_week_spend > 0 else 0
            
            last_week_conversions = last_week_data['conversions'].sum() if 'conversions' in last_week_data.columns else 0
            previous_week_conversions = previous_week_data['conversions'].sum() if 'conversions' in previous_week_data.columns else 0
            conversions_change = ((last_week_conversions - previous_week_conversions) / previous_week_conversions * 100) if previous_week_conversions > 0 else 0
        else:
            spend_change = 0
            conversions_change = 0
    else:
        spend_change = 0
        conversions_change = 0
    
    # Preparar prompt para Gemini
    prompt = f"""
    # ANÁLISE DE PERFORMANCE DE CAMPANHA - RELATÓRIO ESPECIALIZADO
    
    ## CONTEXTO DA ANÁLISE:
    - **Campanha Analisada:** {selected_campaign}
    - **Período de Análise:** {date_range}
    - **Foco Principal:** {metric_focus}
    - **Total de Registros:** {len(campaign_data)} dias
    
    ## DADOS RESUMIDOS DA CAMPANHA:
    
    ### MÉTRICAS DE INVESTIMENTO:
    - **Investimento Total:** R$ {total_spend:,.2f}
    - **Mudança Semanal:** {'+' if spend_change >= 0 else ''}{spend_change:.1f}%
    
    ### MÉTRICAS DE PERFORMANCE:
    - **Conversões Totais:** {total_conversions:,.0f}
    - **Mudança Semanal:** {'+' if conversions_change >= 0 else ''}{conversions_change:.1f}%
    - **Receita Total:** R$ {total_revenue:,.2f}
    
    ### MÉTRICAS DE EFICIÊNCIA:
    - **CPC Médio:** R$ {avg_cpc:,.2f}
    - **CTR Médio:** {avg_ctr:.2f}%
    - **Taxa de Conversão Média:** {avg_conversion_rate:.2f}%
    - **ROAS Médio:** {avg_roas:.2f}x
    
    ### MÉTRICAS DE ALCANCE:
    - **Impressões Totais:** {total_impressions:,.0f}
    - **Cliques Totais:** {total_clicks:,.0f}
    
    ## TENDÊNCIAS IDENTIFICADAS:
    {campaign_data.tail(30).to_string() if len(campaign_data) > 0 else "Dados insuficientes para análise de tendências"}
    
    ## INSTRUÇÕES PARA A ANÁLISE:
    
    Você é um analista de marketing digital especializado. Crie um relatório de performance completo com:
    
    1. **📊 RESUMO EXECUTIVO** (3-4 parágrafos)
       - Performance geral da campanha
       - Principais conquistas e destaques
       - Pontos de atenção críticos
       - Avaliação geral (Excelente/Boa/Regular/Precisa Melhorar)
    
    2. **🎯 ANÁLISE DA MÉTRICA PRINCIPAL: {metric_focus}**
       - Tendência ao longo do tempo (melhorando/piorando/estável)
       - Comparação com benchmarks do setor
       - Fatores que influenciam esta métrica
       - Insights específicos e ações recomendadas
    
    3. **💰 ANÁLISE DE CUSTOS E ROI**
       - Eficiência do investimento (CPC, CPA, ROAS)
       - Comparação com períodos anteriores
       - Identificação de oportunidades de otimização
       - Sugestões de ajuste no orçamento
    
    4. **📈 ANÁLISE DE ENGAGEMENT E ALCANCE**
       - Performance de CTR e taxas de interação
       - Qualidade do tráfego gerado
       - Retenção e engajamento do público
       - Oportunidades para aumentar o alcance
    
    5. **🔄 ANÁLISE DE CONVERSÃO**
       - Taxas de conversão e qualidade
       - Valor por conversão (LTV)
       - Funil de conversão identificado
       - Pontos de fricção e oportunidades
    
    6. **🚀 RECOMENDAÇÕES ESTRATÉGICAS** (TOP 5)
       - 1-2 recomendações para CURTO PRAZO (próximos 7 dias)
       - 2-3 recomendações para MÉDIO PRAZO (próximos 30 dias)
       - 1-2 recomendações para LONGO PRAZO (próximos 90 dias)
       - Priorização clara com impacto esperado
    
    7. **🔮 PREVISÕES E OPORTUNIDADES**
       - Projeções baseadas em tendências atuais
       - Oportunidades de crescimento identificadas
       - Cenários de otimização (conservador/agressivo)
       - KPI's críticos para monitoramento contínuo
    
    ## FORMATO DO RELATÓRIO:
    - Use markdown com formatação profissional
    - Inclua emojis relevantes para cada seção (📊, 🎯, 💰, etc.)
    - Seja direto, baseado em dados e prático
    - Evite jargões excessivos, explique termos técnicos
    - Use formatação como **negrito** para destaques
    - Inclua listas com bullets para clareza
    - Mantenha um tom profissional mas acessível
    """
    
    try:
        if modelo_texto:
            response = modelo_texto.generate_content(prompt)
            return response.text
        else:
            # Fallback para análise básica se Gemini não estiver disponível
            return generate_basic_campaign_analysis(campaign_data, selected_campaign, date_range, metric_focus)
    except Exception as e:
        return f"Erro ao gerar análise: {str(e)}"

def generate_basic_campaign_analysis(campaign_data, selected_campaign, date_range, metric_focus):
    """Gera análise básica se o Gemini não estiver disponível"""
    
    analysis = f"""
    # 📊 RELATÓRIO DE PERFORMANCE - {selected_campaign}
    
    ## 📅 Período: {date_range}
    ## 🎯 Foco de Análise: {metric_focus}
    
    ## 📈 MÉTRICAS PRINCIPAIS:
    
    ### Investimento e Resultados:
    - **Total Investido:** R$ {campaign_data['spend'].sum():,.2f}
    - **Conversões:** {campaign_data['conversions'].sum():,.0f}
    - **Receita Gerada:** R$ {campaign_data['revenue'].sum():,.2f}
    
    ### Eficiência:
    - **CPC Médio:** R$ {campaign_data['cpc'].mean():,.2f}
    - **CTR Médio:** {campaign_data['ctr'].mean():.2f}%
    - **ROAS Médio:** {campaign_data['roas'].mean():.2f}x
    
    ## 📊 TENDÊNCIAS:
    
    ### Últimos 7 dias vs 7 dias anteriores:
    - Investimento: {((campaign_data.tail(7)['spend'].sum() - campaign_data.iloc[-14:-7]['spend'].sum()) / campaign_data.iloc[-14:-7]['spend'].sum() * 100 if campaign_data.iloc[-14:-7]['spend'].sum() > 0 else 0):.1f}%
    - Conversões: {((campaign_data.tail(7)['conversions'].sum() - campaign_data.iloc[-14:-7]['conversions'].sum()) / campaign_data.iloc[-14:-7]['conversions'].sum() * 100 if campaign_data.iloc[-14:-7]['conversions'].sum() > 0 else 0):.1f}%
    
    ## 💡 RECOMENDAÇÕES:
    
    1. **Monitorar diariamente** a métrica {metric_focus}
    2. **Ajustar orçamento** baseado no ROAS
    3. **Otimizar criativos** para melhor CTR
    4. **Testar diferentes públicos** para aumentar conversões
    5. **Analisar concorrência** e benchmarks do setor
    
    ## 🚀 PRÓXIMOS PASSOS:
    
    - Revisar esta análise semanalmente
    - Implementar as recomendações prioritárias
    - Definir metas realistas para o próximo período
    - Monitorar KPIs-chave diariamente
    """
    
    return analysis

def create_campaign_visualizations(df_campaigns, selected_campaign):
    """Cria visualizações para a campanha selecionada"""
    
    if df_campaigns.empty:
        return None
    
    campaign_data = df_campaigns[df_campaigns['campaign'] == selected_campaign].copy()
    
    if campaign_data.empty:
        return None
    
    visualizations = {}
    
    # 1. Gráfico de tendência de gastos e conversões
    if 'spend' in campaign_data.columns and 'conversions' in campaign_data.columns and 'date' in campaign_data.columns:
        fig1 = go.Figure()
        
        # Adicionar linha de gastos
        fig1.add_trace(go.Scatter(
            x=campaign_data['date'],
            y=campaign_data['spend'],
            name='Investimento (R$)',
            yaxis='y',
            line=dict(color='#FF6B6B', width=3),
            mode='lines+markers'
        ))
        
        # Adicionar barras de conversões
        fig1.add_trace(go.Bar(
            x=campaign_data['date'],
            y=campaign_data['conversions'],
            name='Conversões',
            yaxis='y2',
            marker_color='#4ECDC4',
            opacity=0.7
        ))
        
        fig1.update_layout(
            title=f"📈 Investimento vs Conversões - {selected_campaign}",
            xaxis_title="Data",
            yaxis_title="Investimento (R$)",
            yaxis=dict(
                title="Investimento (R$)",
                titlefont=dict(color="#FF6B6B"),
                tickfont=dict(color="#FF6B6B")
            ),
            yaxis2=dict(
                title="Conversões",
                titlefont=dict(color="#4ECDC4"),
                tickfont=dict(color="#4ECDC4"),
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        visualizations['spend_vs_conversions'] = fig1
    
    # 2. Gráfico de ROAS ao longo do tempo
    if 'roas' in campaign_data.columns and 'date' in campaign_data.columns:
        fig2 = px.line(
            campaign_data,
            x='date',
            y='roas',
            title=f"📊 ROAS ao Longo do Tempo - {selected_campaign}",
            markers=True
        )
        
        # Adicionar linha de break-even
        fig2.add_hline(
            y=1, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Break-even", 
            annotation_position="bottom right"
        )
        
        # Adicionar média
        avg_roas = campaign_data['roas'].mean()
        fig2.add_hline(
            y=avg_roas,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Média: {avg_roas:.2f}x",
            annotation_position="top right"
        )
        
        fig2.update_layout(
            yaxis_title="ROAS (x)",
            template='plotly_white',
            height=400
        )
        visualizations['roas_trend'] = fig2
    
    # 3. Gráfico de métricas de eficiência
    efficiency_metrics = ['ctr', 'conversion_rate', 'cpc']
    available_metrics = [m for m in efficiency_metrics if m in campaign_data.columns]
    
    if available_metrics and 'date' in campaign_data.columns:
        fig3 = go.Figure()
        
        colors = ['#667eea', '#764ba2', '#10b981']
        
        for idx, metric in enumerate(available_metrics):
            fig3.add_trace(go.Scatter(
                x=campaign_data['date'],
                y=campaign_data[metric],
                name=metric.upper(),
                mode='lines+markers',
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
        
        fig3.update_layout(
            title=f"📋 Métricas de Eficiência - {selected_campaign}",
            xaxis_title="Data",
            yaxis_title="Valor",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        visualizations['efficiency_metrics'] = fig3
    
    # 4. Heatmap de correlação
    numeric_cols = campaign_data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        correlation_matrix = campaign_data[numeric_cols].corr()
        
        fig4 = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=list(correlation_matrix.columns),
            y=list(correlation_matrix.index),
            annotation_text=np.round(correlation_matrix.values, 2),
            colorscale='RdBu',
            showscale=True,
            hoverinfo='z'
        )
        
        fig4.update_layout(
            title=f"🔥 Correlação entre Métricas - {selected_campaign}",
            template='plotly_white',
            height=500,
            xaxis=dict(tickangle=45)
        )
        visualizations['correlation_heatmap'] = fig4
    
    # 5. Gráfico de pizza por data source (se disponível)
    if 'datasource' in df_campaigns.columns:
        source_distribution = df_campaigns['datasource'].value_counts()
        
        fig5 = px.pie(
            values=source_distribution.values,
            names=source_distribution.index,
            title="📱 Distribuição por Data Source",
            hole=0.4
        )
        
        fig5.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        fig5.update_layout(
            template='plotly_white',
            height=400
        )
        visualizations['source_distribution'] = fig5
    
    return visualizations

def create_performance_dashboard(df_campaigns, selected_campaign):
    """Cria dashboard de performance com métricas-chave"""
    
    if df_campaigns.empty:
        return None
    
    campaign_data = df_campaigns[df_campaigns['campaign'] == selected_campaign].copy()
    
    if campaign_data.empty:
        return None
    
    metrics = {}
    
    # Calcular métricas básicas
    metrics['total_spend'] = campaign_data['spend'].sum() if 'spend' in campaign_data.columns else 0
    metrics['total_conversions'] = campaign_data['conversions'].sum() if 'conversions' in campaign_data.columns else 0
    metrics['total_revenue'] = campaign_data['revenue'].sum() if 'revenue' in campaign_data.columns else 0
    metrics['total_impressions'] = campaign_data['impressions'].sum() if 'impressions' in campaign_data.columns else 0
    metrics['total_clicks'] = campaign_data['clicks'].sum() if 'clicks' in campaign_data.columns else 0
    
    # Calcular métricas de eficiência
    metrics['avg_cpc'] = campaign_data['cpc'].mean() if 'cpc' in campaign_data.columns else 0
    metrics['avg_ctr'] = campaign_data['ctr'].mean() if 'ctr' in campaign_data.columns else 0
    metrics['avg_roas'] = campaign_data['roas'].mean() if 'roas' in campaign_data.columns else 0
    metrics['avg_conversion_rate'] = campaign_data['conversion_rate'].mean() if 'conversion_rate' in campaign_data.columns else 0
    
    # Calcular CPA
    metrics['cpa'] = metrics['total_spend'] / metrics['total_conversions'] if metrics['total_conversions'] > 0 else 0
    
    return metrics

# =============================================================================
# FUNÇÕES ORIGINAIS DO APP (MANTIDAS)
# =============================================================================

def identificar_colunas_numericas(df):
    """Identifica automaticamente colunas numéricas"""
    colunas_numericas = []
    
    for col in df.columns:
        try:
            # Tenta converter para numérico
            if pd.api.types.is_numeric_dtype(df[col]):
                colunas_numericas.append(col)
            elif df[col].dropna().apply(lambda x: isinstance(x, (int, float, np.number))).any():
                colunas_numericas.append(col)
        except:
            continue
    
    return colunas_numericas

def analisar_coluna(df, coluna):
    """Analisa uma coluna específica"""
    if coluna not in df.columns:
        return None
    
    dados_coluna = df[coluna]
    analise = {
        'nome': coluna,
        'tipo': str(dados_coluna.dtype),
        'total': len(dados_coluna),
        'nao_nulos': dados_coluna.notna().sum(),
        'nulos': dados_coluna.isna().sum(),
        'percentual_nulos': (dados_coluna.isna().sum() / len(dados_coluna)) * 100,
        'valores_unicos': dados_coluna.nunique()
    }
    
    # Se for numérica
    if pd.api.types.is_numeric_dtype(dados_coluna):
        dados_validos = dados_coluna.dropna()
        if len(dados_validos) > 0:
            analise.update({
                'tipo_detalhado': 'Numérica',
                'min': dados_validos.min(),
                'max': dados_validos.max(),
                'media': dados_validos.mean(),
                'mediana': dados_validos.median(),
                'desvio_padrao': dados_validos.std(),
                'q1': dados_validos.quantile(0.25),
                'q3': dados_validos.quantile(0.75),
                'assimetria': dados_validos.skew(),
                'curtose': dados_validos.kurt()
            })
    # Se for categórica/texto
    elif dados_coluna.dtype == 'object':
        analise.update({
            'tipo_detalhado': 'Texto/Categórica',
            'valores_mais_comuns': dados_coluna.value_counts().head(10).to_dict(),
            'valor_mais_frequente': dados_coluna.mode().iloc[0] if not dados_coluna.mode().empty else None,
            'frequencia_valor_mais_comum': dados_coluna.value_counts().iloc[0] if not dados_coluna.empty else 0
        })
    # Se for data
    elif pd.api.types.is_datetime64_any_dtype(dados_coluna):
        dados_validos = dados_coluna.dropna()
        if len(dados_validos) > 0:
            analise.update({
                'tipo_detalhado': 'Data',
                'data_minima': dados_validos.min(),
                'data_maxima': dados_validos.max(),
                'intervalo_dias': (dados_validos.max() - dados_validos.min()).days
            })
    
    return analise

def criar_visualizacao_coluna(df, coluna):
    """Cria visualização adequada para o tipo de coluna"""
    if coluna not in df.columns:
        return None
    
    dados = df[coluna].dropna()
    
    # Coluna numérica
    if pd.api.types.is_numeric_dtype(df[coluna]):
        fig = px.histogram(
            df, 
            x=coluna,
            nbins=50,
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
        contagem_diaria = df.groupby(df[coluna].dt.date).size().reset_index()
        contagem_diaria.columns = ['data', 'contagem']
        
        fig = px.line(
            contagem_diaria,
            x='data',
            y='contagem',
            title=f"Frequência por Data - {coluna}"
        )
        return fig
    
    return None

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================

# Inicializar estado
if 'df_completo' not in st.session_state:
    st.session_state.df_completo = pd.DataFrame()
if 'colunas_numericas' not in st.session_state:
    st.session_state.colunas_numericas = []
if 'df_campaigns' not in st.session_state:
    st.session_state.df_campaigns = pd.DataFrame()
if 'campaign_analysis' not in st.session_state:
    st.session_state.campaign_analysis = None
if 'selected_campaign' not in st.session_state:
    st.session_state.selected_campaign = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Verificação de variáveis de ambiente
    st.subheader("🔧 Configuração de Credenciais")
    
    # Botão para verificar configuração
    if st.button("🔍 Verificar Configuração Atual"):
        with st.expander("Configurações Detectadas"):
            # Verificar métodos disponíveis
            metodos = []
            if all(key in os.environ for key in ['type', 'project_id', 'private_key', 'client_email']):
                metodos.append("✅ Variáveis de ambiente individuais")
            if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
                metodos.append("✅ JSON em variável de ambiente")
            if 'gcp_service_account' in st.secrets:
                metodos.append("✅ Streamlit Secrets")
            
            if metodos:
                st.write("**Métodos disponíveis:**")
                for metodo in metodos:
                    st.write(f"- {metodo}")
                
                # Mostrar algumas informações (sem expor credenciais sensíveis)
                if 'project_id' in os.environ:
                    st.write(f"**Project ID:** {os.environ['project_id']}")
                if 'client_email' in os.environ:
                    st.write(f"**Client Email:** {os.environ['client_email']}")
            else:
                st.error("❌ Nenhum método de autenticação configurado")
    
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
    col_load1, col_load2 = st.columns(2)
    
    with col_load1:
        if st.button("📊 Carregar Dados Gerais", use_container_width=True):
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
    
    with col_load2:
        if st.button("🎯 Carregar Dados Campanhas", use_container_width=True, type="primary"):
            with st.spinner("Carregando dados de campanhas..."):
                client = get_bigquery_client()
                if client:
                    df_campaigns = load_campaign_performance_data(
                        client,
                        start_date=data_inicio,
                        end_date=data_fim,
                        datasources=selected_sources
                    )
                    
                    if not df_campaigns.empty:
                        st.session_state.df_campaigns = df_campaigns
                        st.success(f"✅ {len(df_campaigns):,} registros de campanhas carregados")
                        
                        # Listar campanhas disponíveis
                        campaigns = df_campaigns['campaign'].unique() if 'campaign' in df_campaigns.columns else []
                        st.session_state.available_campaigns = campaigns
                        st.info(f"📋 {len(campaigns)} campanhas disponíveis")
                    else:
                        st.error("Nenhum dado de campanha encontrado")
                else:
                    st.error("❌ Não foi possível conectar ao BigQuery.")

# Verificar se há dados carregados
df = st.session_state.df_completo
colunas_numericas = st.session_state.colunas_numericas
df_campaigns = st.session_state.df_campaigns

if df.empty and df_campaigns.empty:
    st.warning("📭 Nenhum dado carregado. Use os botões na sidebar para carregar dados.")
    st.stop()

# Abas principais - AGORA COM A NOVA ABA DE PERFORMANCE
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Visão Geral das Colunas", 
    "📈 Análise Numérica", 
    "🔍 Explorar Colunas", 
    "📊 Visualizar Dados",
    "📐 Análise Estatística",
    "🎯 Performance de Campanhas"  # NOVA ABA
])

# =============================================================================
# TAB 1: VISÃO GERAL DAS COLUNAS (MANTIDO IGUAL)
# =============================================================================

with tab1:
    st.header("📋 Visão Geral de TODAS as Colunas")
    
    if df.empty:
        st.info("ℹ️ Nenhum dado geral carregado. Carregue dados gerais na sidebar.")
    else:
        # Estatísticas gerais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Colunas", len(df.columns))
        
        with col2:
            st.metric("Colunas Numéricas", len(colunas_numericas))
        
        with col3:
            st.metric("Total de Registros", f"{len(df):,}")
        
        with col4:
            memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Uso de Memória", f"{memoria_mb:.1f} MB")
        
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
                    st.write(f"**Tipo:** {analise['tipo']}")
                    st.write(f"**Não nulos:** {analise['nao_nulos']:,} ({analise['percentual_nulos']:.1f}% nulos)")
                    st.write(f"**Valores únicos:** {analise['valores_unicos']:,}")
                
                with col_info2:
                    if analise['tipo_detalhado'] == 'Numérica':
                        st.write(f"**Média:** {analise.get('media', 'N/A'):.2f}")
                        st.write(f"**Min:** {analise.get('min', 'N/A'):.2f}")
                        st.write(f"**Max:** {analise.get('max', 'N/A'):.2f}")
                    elif analise['tipo_detalhado'] == 'Texto/Categórica':
                        if analise.get('valor_mais_frequente'):
                            st.write(f"**Valor mais comum:** {analise['valor_mais_frequente']}")
                            st.write(f"**Frequência:** {analise['frequencia_valor_mais_comum']:,}")
                    elif analise['tipo_detalhado'] == 'Data':
                        st.write(f"**Período:** {analise.get('data_minima', 'N/A')} a {analise.get('data_maxima', 'N/A')}")
                
                # Botão para visualizar
                if st.button(f"📊 Visualizar {col}", key=f"viz_{col}_tab1"):
                    st.session_state.coluna_selecionada = col
                    st.rerun()

# =============================================================================
# TAB 2: ANÁLISE NUMÉRICA (MANTIDO IGUAL)
# =============================================================================

with tab2:
    st.header("📈 Análise de Colunas Numéricas")
    
    if df.empty:
        st.info("ℹ️ Nenhum dado geral carregado. Carregue dados gerais na sidebar.")
    elif not colunas_numericas:
        st.warning("Nenhuma coluna numérica identificada")
    else:
        st.success(f"✅ {len(colunas_numericas)} colunas numéricas disponíveis para análise")
        
        # Selecionar colunas para análise
        colunas_selecionadas = st.multiselect(
            "Selecione colunas numéricas para análise",
            options=colunas_numericas,
            default=colunas_numericas[:5] if len(colunas_numericas) >= 5 else colunas_numericas,
            key="colunas_selecionadas_tab2"
        )
        
        if colunas_selecionadas:
            # Estatísticas descritivas
            st.subheader("📊 Estatísticas Descritivas")
            
            stats_df = df[colunas_selecionadas].describe().T
            stats_df['missing'] = df[colunas_selecionadas].isna().sum()
            stats_df['missing_pct'] = (df[colunas_selecionadas].isna().sum() / len(df) * 100)
            
            # Formatar
            def formatar_numero(x):
                if isinstance(x, (int, np.integer)):
                    return f"{x:,}"
                elif isinstance(x, (float, np.floating)):
                    if abs(x) < 0.01:
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
                
                # Calcular correlações
                correlacao = df[colunas_selecionadas].corr()
                
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
                        if abs(corr) > 0.3 and not pd.isna(corr):
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

# =============================================================================
# TAB 3: EXPLORAR COLUNAS (MANTIDO IGUAL)
# =============================================================================

with tab3:
    st.header("🔍 Explorar Colunas Individualmente")
    
    if df.empty:
        st.info("ℹ️ Nenhum dado geral carregado. Carregue dados gerais na sidebar.")
    else:
        coluna_selecionada = st.selectbox(
            "Selecione uma coluna para explorar",
            options=sorted(df.columns),
            index=0,
            key="coluna_selecionada_tab3"
        )
        
        if coluna_selecionada:
            analise = analisar_coluna(df, coluna_selecionada)
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.metric("Total de Valores", analise['total'], key="total_valores_tab3")
                st.metric("Valores Não Nulos", f"{analise['nao_nulos']:,}", key="nao_nulos_tab3")
                st.metric("Valores Únicos", f"{analise['valores_unicos']:,}", key="valores_unicos_tab3")
            
            with col_info2:
                st.metric("Valores Nulos", f"{analise['nulos']:,}", key="nulos_tab3")
                st.metric("% Nulos", f"{analise['percentual_nulos']:.1f}%", key="percentual_nulos_tab3")
            
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
# TAB 4: VISUALIZAR DADOS (MANTIDO IGUAL)
# =============================================================================

with tab4:
    st.header("📊 Visualizar Dados Completos")
    
    if df.empty:
        st.info("ℹ️ Nenhum dado geral carregado. Carregue dados gerais na sidebar.")
    else:
        # Selecionar colunas para visualizar
        colunas_vis = st.multiselect(
            "Selecione colunas para visualizar",
            options=sorted(df.columns),
            default=sorted(df.columns)[:10],
            key="colunas_vis_tab4"
        )
        
        if colunas_vis:
            # Filtros
            st.subheader("🔍 Filtros")
            
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                # Filtro por datasource se existir
                if 'datasource' in df.columns:
                    datasources = sorted(df['datasource'].unique())
                    ds_selecionados = st.multiselect(
                        "Data Sources",
                        options=datasources,
                        default=datasources[:min(3, len(datasources))],
                        key="ds_selecionados_tab4"
                    )
                    if ds_selecionados:
                        df_filtrado = df[df['datasource'].isin(ds_selecionados)]
                    else:
                        df_filtrado = df.copy()
                else:
                    df_filtrado = df.copy()
            
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
                        min_val = st.number_input(
                            f"Valor mínimo de {col_filtro}",
                            value=float(df_filtrado[col_filtro].min()),
                            min_value=float(df_filtrado[col_filtro].min()),
                            max_value=float(df_filtrado[col_filtro].max()),
                            key=f"min_val_{col_filtro}_tab4"
                        )
                        df_filtrado = df_filtrado[df_filtrado[col_filtro] >= min_val]
            
            with col_f3:
                # Limite de linhas
                limite_linhas = st.slider("Linhas para mostrar", 10, 1000, 100, key="limite_linhas_tab4")
            
            # Mostrar dados
            st.subheader(f"📋 Dados ({len(df_filtrado):,} registros filtrados)")
            
            # Paginação
            total_pages = max(1, len(df_filtrado) // limite_linhas + 1)
            
            col_pg1, col_pg2, col_pg3 = st.columns([1, 2, 1])
            
            with col_pg1:
                page_number = st.number_input("Página", 1, total_pages, 1, key="page_number_tab4")
            
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
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else x
                    )
            
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400
            )
            
            # Download
            st.subheader("📥 Exportar Dados")
            
            csv = df_filtrado[colunas_vis].to_csv(index=False)
            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name=f"dados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="download_csv_tab4"
            )

# =============================================================================
# TAB 5: ANÁLISE ESTATÍSTICA AVANÇADA (MANTIDO IGUAL)
# =============================================================================

with tab5:
    st.header("📐 Análise Estatística Avançada")
    
    if df.empty:
        st.info("ℹ️ Nenhum dado geral carregado. Carregue dados gerais na sidebar.")
    elif not colunas_numericas:
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
                options=colunas_numericas,
                help="Variáveis que podem explicar a variável resposta",
                key="variaveis_explicativas_tab5"
            )
        
        # Remover variável resposta das explicativas se selecionada
        if variavel_resposta != 'Nenhuma' and variavel_resposta in variaveis_explicativas:
            variaveis_explicativas = [v for v in variaveis_explicativas if v != variavel_resposta]
        
        if variavel_resposta != 'Nenhuma' and variaveis_explicativas:
            # Preparar dados
            dados_analise = df[[variavel_resposta] + variaveis_explicativas].dropna()
            
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
                    st.subheader("📈 Relações Principais")
                    
                    top_correl = df_correl.head(3)
                    
                    if not top_correl.empty:
                        cols_graf = st.columns(min(3, len(top_correl)))
                        
                        for idx, row in top_correl.iterrows():
                            with cols_graf[idx % len(cols_graf)]:
                                fig_scatter = px.scatter(
                                    dados_analise,
                                    x=row['Variável'],
                                    y=variavel_resposta,
                                    trendline="ols",
                                    title=f"{variavel_resposta} vs {row['Variável']}",
                                    labels={
                                        row['Variável']: row['Variável'],
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
                        nbins=50,
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
                    st.metric("Q1 (25%)", f"{Q1:.2f}", key="q1_tab5")
                
                with col_out2:
                    st.metric("Q3 (75%)", f"{Q3:.2f}", key="q3_tab5")
                
                with col_out3:
                    st.metric("IQR", f"{IQR:.2f}", key="iqr_tab5")
                
                with col_out4:
                    st.metric("Outliers", f"{len(outliers)} ({len(outliers)/len(dados_analise)*100:.1f}%)", key="outliers_tab5")
                
                if not outliers.empty:
                    with st.expander("📋 Ver Outliers"):
                        st.dataframe(outliers, use_container_width=True)

# =============================================================================
# TAB 6: NOVA ABA - PERFORMANCE DE CAMPANHAS
# =============================================================================

with tab6:
    st.header("🎯 Performance de Campanhas")
    st.markdown("Análise detalhada de campanhas de marketing com insights baseados em IA")
    
    if df_campaigns.empty:
        st.warning("📭 Nenhum dado de campanha carregado. Use o botão 'Carregar Dados Campanhas' na sidebar para começar.")
        
        with st.expander("ℹ️ Como usar esta funcionalidade"):
            st.markdown("""
            ### 🎯 O que é a Análise de Performance de Campanhas?
            
            Esta funcionalidade permite analisar campanhas de marketing digital com:
            
            1. **📊 Dashboard de Métricas**: Visualização de KPIs principais
            2. **📈 Análise de Tendências**: Evolução temporal das campanhas
            3. **🤖 Análise com IA**: Insights automáticos usando Gemini
            4. **📋 Relatórios Detalhados**: Análises completas por campanha
            
            ### 📥 Como começar:
            
            1. **Configure as credenciais** do BigQuery na sidebar
            2. **Selecione os Data Sources** (Facebook, Google Ads, TikTok)
            3. **Defina o período** de análise
            4. **Clique em 'Carregar Dados Campanhas'**
            
            ### 📊 Métricas Analisadas:
            
            - **Investimento (Spend)**: Total gasto nas campanhas
            - **Conversões**: Número de conversões geradas
            - **ROAS (Return on Ad Spend)**: Retorno sobre investimento
            - **CPC (Cost per Click)**: Custo por clique
            - **CTR (Click-Through Rate)**: Taxa de cliques
            - **Taxa de Conversão**: Percentual de cliques que convertem
            
            ### 🚀 Benefícios:
            
            - **Tomada de decisão baseada em dados**
            - **Identificação de oportunidades de otimização**
            - **Comparação entre diferentes campanhas**
            - **Previsões e recomendações automáticas**
            """)
        
        st.stop()
    
    # Sidebar interna para configuração de análise de campanha
    with st.sidebar.expander("🎯 Configuração da Análise", expanded=True):
        st.subheader("Selecionar Campanha")
        
        # Listar campanhas disponíveis
        available_campaigns = df_campaigns['campaign'].unique() if 'campaign' in df_campaigns.columns else []
        
        if len(available_campaigns) == 0:
            st.error("Nenhuma campanha encontrada nos dados carregados")
            st.stop()
        
        selected_campaign = st.selectbox(
            "Escolha a campanha para análise:",
            options=available_campaigns,
            index=0,
            help="Selecione a campanha que deseja analisar"
        )
        
        st.session_state.selected_campaign = selected_campaign
        
        # Métrica de foco
        st.subheader("Foco da Análise")
        metric_focus = st.selectbox(
            "Métrica principal para análise:",
            options=["ROAS (Retorno sobre Investimento)", "Conversões", "CTR (Taxa de Cliques)", 
                    "CPC (Custo por Clique)", "Investimento Total", "Receita Gerada"],
            index=0,
            help="Selecione a métrica que será o foco da análise"
        )
        
        # Período de análise
        st.subheader("Período da Análise")
        analysis_period = st.radio(
            "Período para análise:",
            ["Todo o período carregado", "Últimos 30 dias", "Últimos 60 dias", "Últimos 90 dias"],
            index=0,
            horizontal=True
        )
        
        # Botão para gerar análise
        generate_analysis = st.button(
            "🚀 Gerar Análise com IA",
            type="primary",
            use_container_width=True,
            help="Clique para gerar análise detalhada usando IA"
        )
    
    # Layout principal da análise
    col_main1, col_main2 = st.columns([2, 1])
    
    with col_main1:
        st.markdown(f"### 📊 Análise da Campanha: **{selected_campaign}**")
        
        # Dashboard de métricas
        st.subheader("📈 Dashboard de Performance")
        
        # Calcular métricas
        campaign_data = df_campaigns[df_campaigns['campaign'] == selected_campaign]
        
        if not campaign_data.empty:
            # Métricas em cards
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            
            with col_metric1:
                total_spend = campaign_data['spend'].sum() if 'spend' in campaign_data.columns else 0
                st.metric(
                    "💰 Investimento Total",
                    f"R$ {total_spend:,.0f}",
                    help="Total investido na campanha"
                )
            
            with col_metric2:
                total_conversions = campaign_data['conversions'].sum() if 'conversions' in campaign_data.columns else 0
                st.metric(
                    "🔄 Total de Conversões",
                    f"{total_conversions:,.0f}",
                    help="Número total de conversões"
                )
            
            with col_metric3:
                avg_roas = campaign_data['roas'].mean() if 'roas' in campaign_data.columns else 0
                st.metric(
                    "📈 ROAS Médio",
                    f"{avg_roas:.2f}x",
                    help="Retorno médio sobre investimento"
                )
            
            with col_metric4:
                avg_cpc = campaign_data['cpc'].mean() if 'cpc' in campaign_data.columns else 0
                st.metric(
                    "🎯 CPC Médio",
                    f"R$ {avg_cpc:.2f}",
                    help="Custo médio por clique"
                )
            
            # Mais métricas
            col_metric5, col_metric6, col_metric7, col_metric8 = st.columns(4)
            
            with col_metric5:
                total_revenue = campaign_data['revenue'].sum() if 'revenue' in campaign_data.columns else 0
                st.metric(
                    "💸 Receita Gerada",
                    f"R$ {total_revenue:,.0f}",
                    help="Receita total gerada"
                )
            
            with col_metric6:
                avg_ctr = campaign_data['ctr'].mean() if 'ctr' in campaign_data.columns else 0
                st.metric(
                    "👆 CTR Médio",
                    f"{avg_ctr:.2f}%",
                    help="Taxa média de cliques"
                )
            
            with col_metric7:
                total_clicks = campaign_data['clicks'].sum() if 'clicks' in campaign_data.columns else 0
                st.metric(
                    "🖱️ Total de Cliques",
                    f"{total_clicks:,.0f}",
                    help="Número total de cliques"
                )
            
            with col_metric8:
                conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
                st.metric(
                    "📊 Taxa de Conversão",
                    f"{conversion_rate:.2f}%",
                    help="Percentual de cliques que convertem"
                )
    
    with col_main2:
        st.markdown("### 📋 Informações da Campanha")
        
        # Informações básicas
        with st.container():
            st.markdown("#### 📅 Período Ativo")
            if 'date' in campaign_data.columns:
                start_date = campaign_data['date'].min()
                end_date = campaign_data['date'].max()
                days_active = (end_date - start_date).days
                
                st.write(f"**Início:** {start_date.strftime('%d/%m/%Y')}")
                st.write(f"**Término:** {end_date.strftime('%d/%m/%Y')}")
                st.write(f"**Dias ativa:** {days_active} dias")
            
            st.markdown("#### 📱 Data Sources")
            if 'datasource' in campaign_data.columns:
                sources = campaign_data['datasource'].unique()
                for source in sources:
                    st.write(f"- {source}")
            
            st.markdown("#### 🎯 Status da Campanha")
            
            # Avaliação simples baseada no ROAS
            if avg_roas > 3:
                st.success("✅ **Excelente Performance**")
                st.write("ROAS acima de 3x indica campanha muito eficiente")
            elif avg_roas > 1.5:
                st.info("📈 **Boa Performance**")
                st.write("ROAS entre 1.5x e 3x indica campanha rentável")
            elif avg_roas > 1:
                st.warning("⚠️ **Performance Regular**")
                st.write("ROAS entre 1x e 1.5x indica campanha no break-even")
            else:
                st.error("❌ **Performance Insatisfatória**")
                st.write("ROAS abaixo de 1x indica prejuízo")
    
    # Visualizações
    st.markdown("---")
    st.subheader("📊 Visualizações da Performance")
    
    if generate_analysis or st.session_state.campaign_analysis:
        # Gerar visualizações
        visualizations = create_campaign_visualizations(df_campaigns, selected_campaign)
        
        if visualizations:
            # Layout das visualizações
            for viz_name, fig in visualizations.items():
                if viz_name == 'spend_vs_conversions':
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_viz2:
                        # Análise de tendência
                        st.markdown("#### 📈 Análise de Tendência")
                        
                        if 'spend' in campaign_data.columns and 'conversions' in campaign_data.columns:
                            # Calcular correlação
                            correlation = campaign_data['spend'].corr(campaign_data['conversions'])
                            
                            if correlation > 0.7:
                                st.success("**Correlação Forte Positiva**")
                                st.write("Investimento e conversões estão fortemente relacionados")
                            elif correlation > 0.3:
                                st.info("**Correlação Moderada Positiva**")
                                st.write("Há relação moderada entre investimento e conversões")
                            elif correlation > -0.3:
                                st.warning("**Correlação Fraca**")
                                st.write("Pouca relação entre investimento e conversões")
                            else:
                                st.error("**Correlação Negativa**")
                                st.write("Aumento no investimento está associado a menos conversões")
                
                elif viz_name == 'roas_trend':
                    col_viz3, col_viz4 = st.columns(2)
                    
                    with col_viz3:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_viz4:
                        st.markdown("#### 💰 Análise de ROAS")
                        
                        roas_stats = {
                            "Mínimo": campaign_data['roas'].min() if 'roas' in campaign_data.columns else 0,
                            "Máximo": campaign_data['roas'].max() if 'roas' in campaign_data.columns else 0,
                            "Média": avg_roas,
                            "Mediana": campaign_data['roas'].median() if 'roas' in campaign_data.columns else 0
                        }
                        
                        for stat_name, stat_value in roas_stats.items():
                            st.write(f"**{stat_name}:** {stat_value:.2f}x")
                        
                        # Dias acima e abaixo do break-even
                        if 'roas' in campaign_data.columns:
                            days_above = (campaign_data['roas'] > 1).sum()
                            days_below = (campaign_data['roas'] <= 1).sum()
                            total_days = len(campaign_data)
                            
                            st.write(f"**Dias acima do break-even:** {days_above} ({days_above/total_days*100:.1f}%)")
                            st.write(f"**Dias abaixo do break-even:** {days_below} ({days_below/total_days*100:.1f}%)")
                
                elif viz_name == 'efficiency_metrics':
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_name == 'correlation_heatmap':
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_name == 'source_distribution':
                    st.plotly_chart(fig, use_container_width=True)
    
    # Análise com IA
    st.markdown("---")
    st.subheader("🤖 Análise com Inteligência Artificial")
    
    if generate_analysis:
        with st.spinner("🧠 Analisando dados e gerando insights com IA..."):
            client = get_bigquery_client()
            analysis = generate_campaign_analysis_with_ai(
                df_campaigns,
                selected_campaign,
                analysis_period,
                metric_focus,
                client
            )
            
            st.session_state.campaign_analysis = analysis
            
            # Mostrar análise
            st.markdown(analysis)
            
            # Botão para download
            st.download_button(
                label="📥 Baixar Relatório Completo",
                data=analysis,
                file_name=f"relatorio_campanha_{selected_campaign}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                help="Baixe o relatório completo em formato Markdown"
            )
    
    elif st.session_state.campaign_analysis:
        st.markdown(st.session_state.campaign_analysis)
        
        # Botão para download
        st.download_button(
            label="📥 Baixar Relatório Completo",
            data=st.session_state.campaign_analysis,
            file_name=f"relatorio_campanha_{selected_campaign}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            help="Baixe o relatório completo em formato Markdown"
        )
    
    # Seção de recomendações rápidas
    st.markdown("---")
    st.subheader("💡 Recomendações Rápidas")
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        with st.container():
            st.markdown("#### 🚀 Para Aumentar ROAS")
            st.write("1. Otimizar segmentação de público")
            st.write("2. Testar diferentes criativos")
            st.write("3. Ajustar lances por dispositivo")
            st.write("4. Melhorar landing pages")
    
    with col_rec2:
        with st.container():
            st.markdown("#### 📉 Para Reduzir CPC")
            st.write("1. Refinar palavras-chave")
            st.write("2. Melhorar Quality Score")
            st.write("3. Ajustar horários de veiculação")
            st.write("4. Testar diferentes formatos")
    
    with col_rec3:
        with st.container():
            st.markdown("#### 📊 Para Aumentar Conversões")
            st.write("1. Otimizar call-to-action")
            st.write("2. Simplificar formulários")
            st.write("3. Melhorar velocidade do site")
            st.write("4. Implementar remarketing")

# =============================================================================
# RODAPÉ
# =============================================================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if not df.empty:
        st.caption(f"📊 Dados Gerais: {len(df):,} registros • {len(df.columns)} colunas")

with footer_col2:
    if not df_campaigns.empty:
        st.caption(f"🎯 Campanhas: {len(df_campaigns['campaign'].unique()):,} campanhas ativas")

with footer_col3:
    st.caption(f"⏰ Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Nota sobre IA
if gemini_api_key and (st.session_state.campaign_analysis is not None):
    st.sidebar.success("🤖 IA Gemini disponível para análises")
elif not gemini_api_key:
    st.sidebar.warning("⚠️ Gemini não configurado - Análises básicas apenas")
