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
        # Primeiro, vamos descobrir quais colunas existem na tabela
        query_schema = """
        SELECT column_name, data_type 
        FROM `macfor-media-flow.ads.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'app_view_campaigns'
        AND column_name IN (
            'date', 'campaign', 'datasource', 'impressions', 'clicks', 
            'spend', 'conversions', 'conversion_value', 'cpc', 'cpm', 
            'ctr', 'conversion_rate', 'roas', 'reach', 'frequency',
            'video_views', 'video_view_rate', 'video_plays', 'engagements',
            'engagement_rate', 'landing_page_views', 'add_to_cart', 
            'purchases', 'revenue'
        )
        """
        
        try:
            schema_df = _client.query(query_schema).to_dataframe()
            colunas_existentes = schema_df['column_name'].tolist()
            
            # Verificar se a coluna conversions existe
            if 'conversions' not in colunas_existentes:
                st.warning("⚠️ A coluna 'conversions' não existe na tabela. Vou usar colunas alternativas.")
            
            # Construir a query apenas com as colunas que existem
            colunas_query = ", ".join(colunas_existentes)
            
        except Exception as schema_error:
            # Se não conseguir obter o schema, usar colunas padrão
            st.warning(f"Não foi possível obter schema: {schema_error}")
            colunas_query = """
                date, campaign, datasource, impressions, clicks, spend, 
                conversion_value, cpc, cpm, ctr, conversion_rate, roas, 
                reach, frequency, video_views, video_view_rate, video_plays, 
                engagements, engagement_rate, landing_page_views, add_to_cart, 
                purchases, revenue
            """
        
        query = f"""
        SELECT 
            {colunas_query}
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
            
            # Criar coluna conversions se não existir (pode ser purchases ou outra)
            if 'conversions' not in df.columns:
                if 'purchases' in df.columns:
                    df['conversions'] = df['purchases'].fillna(0)
                    st.info("ℹ️ Usando a coluna 'purchases' como 'conversions'")
                elif 'add_to_cart' in df.columns:
                    df['conversions'] = df['add_to_cart'].fillna(0)
                    st.info("ℹ️ Usando a coluna 'add_to_cart' como 'conversions'")
                elif 'landing_page_views' in df.columns:
                    df['conversions'] = df['landing_page_views'].fillna(0)
                    st.info("ℹ️ Usando a coluna 'landing_page_views' como 'conversions'")
                else:
                    df['conversions'] = 0
                    st.warning("⚠️ Nenhuma coluna de conversão encontrada. Usando valor 0.")
            
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

def generate_campaign_analysis_with_ai(df_campaigns, selected_campaign, date_range, metric_focus):
    """Gera análise detalhada da campanha usando Gemini IA"""
    
    if df_campaigns.empty:
        return "Nenhum dado disponível para análise."
    
    # Preparar dados para análise
    campaign_data = df_campaigns[df_campaigns['campaign'] == selected_campaign].copy()
    
    if campaign_data.empty:
        return f"Nenhum dado encontrado para a campanha '{selected_campaign}'."
    
    # Ordenar por data se a coluna existir
    if 'date' in campaign_data.columns:
        campaign_data = campaign_data.sort_values('date')
    
    # Calcular métricas disponíveis
    metrics_summary = {}
    
    # Métricas básicas
    for metric in ['spend', 'impressions', 'clicks', 'conversions', 'revenue']:
        if metric in campaign_data.columns:
            metrics_summary[f'total_{metric}'] = campaign_data[metric].sum()
            metrics_summary[f'avg_{metric}_per_day'] = campaign_data[metric].mean()
    
    # Métricas calculadas
    if 'spend' in campaign_data.columns and 'clicks' in campaign_data.columns:
        total_clicks = metrics_summary.get('total_clicks', 0)
        total_spend = metrics_summary.get('total_spend', 0)
        if total_clicks > 0:
            metrics_summary['cpc'] = total_spend / total_clicks
    
    if 'clicks' in campaign_data.columns and 'impressions' in campaign_data.columns:
        total_impressions = metrics_summary.get('total_impressions', 0)
        if total_impressions > 0:
            metrics_summary['ctr'] = (metrics_summary.get('total_clicks', 0) / total_impressions) * 100
    
    if 'spend' in campaign_data.columns and 'revenue' in campaign_data.columns:
        total_revenue = metrics_summary.get('total_revenue', 0)
        if total_spend > 0:
            metrics_summary['roas'] = total_revenue / total_spend
    
    if 'clicks' in campaign_data.columns and 'conversions' in campaign_data.columns:
        if total_clicks > 0:
            metrics_summary['conversion_rate'] = (metrics_summary.get('total_conversions', 0) / total_clicks) * 100
    
    # Preparar prompt adaptado às métricas disponíveis
    prompt = f"""
    # ANÁLISE DE PERFORMANCE DE CAMPANHA - RELATÓRIO ESPECIALIZADO
    
    ## CONTEXTO DA ANÁLISE:
    - **Campanha Analisada:** {selected_campaign}
    - **Período de Análise:** {date_range}
    - **Foco Principal:** {metric_focus}
    - **Total de Registros:** {len(campaign_data)} dias
    
    ## DADOS DISPONÍVEIS:
    {campaign_data.head(20).to_string() if len(campaign_data) > 0 else "Dados insuficientes"}
    
    ## INSTRUÇÕES PARA A ANÁLISE:
    
    Você é um analista de marketing digital especializado. Crie um relatório de performance completo com base nos dados disponíveis.
    
    Inclua:
    1. **📊 RESUMO EXECUTIVO** - Performance geral
    2. **🎯 ANÁLISE DA MÉTRICA PRINCIPAL** - {metric_focus}
    3. **💰 ANÁLISE DE CUSTOS E RESULTADOS** 
    4. **🚀 RECOMENDAÇÕES PRÁTICAS**
    5. **📈 PRÓXIMOS PASSOS**
    
    Seja prático, baseado em dados e evite jargões excessivos.
    """
    
    try:
        if modelo_texto:
            response = modelo_texto.generate_content(prompt)
            return response.text
        else:
            return "⚠️ Gemini não configurado. Configure a API key para análises com IA."
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
    
    try:
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
            else:
                analise.update({'tipo_detalhado': 'Numérica (vazia)'})
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
            else:
                analise.update({'tipo_detalhado': 'Data (vazia)'})
        else:
            analise.update({'tipo_detalhado': 'Outro'})
            
        return analise
        
    except Exception as e:
        # Retorna uma análise básica em caso de erro
        return {
            'nome': coluna,
            'tipo': 'Erro',
            'tipo_detalhado': f'Erro na análise: {str(e)[:50]}...',
            'total': len(df),
            'nao_nulos': 0,
            'nulos': len(df),
            'percentual_nulos': 100,
            'valores_unicos': 0
        }

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
tab1, tab2, tab4, tab5, tab6 = st.tabs([
    "📋 Visão Geral das Colunas", 
    "📈 Análise Numérica", 
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

# =============================================================================
# TAB 6: NOVA ABA - PERFORMANCE DE CAMPANHAS (VERSÃO SIMPLIFICADA)
# =============================================================================

with tab6:
    st.header("🎯 Performance de Campanhas")
    st.markdown("Análise detalhada de campanhas de marketing")
    
    # Usar os dados já carregados
    df = st.session_state.df_completo
    
    if df.empty:
        st.warning("📭 Nenhum dado carregado. Use o botão 'Carregar TODOS os Dados' na sidebar.")
        st.stop()
    
    # Verificar colunas mínimas
    if 'campaign' not in df.columns:
        st.error("❌ Coluna 'campaign' não encontrada nos dados.")
        st.info("ℹ️ Esta análise requer uma coluna chamada 'campaign' para identificar as campanhas.")
        st.stop()
    
    # Identificar colunas numéricas disponíveis
    numeric_cols = st.session_state.colunas_numericas
    if not numeric_cols:
        st.warning("⚠️ Nenhuma coluna numérica identificada para análise.")
        st.stop()
    
    # Sidebar para configuração
    with st.sidebar.expander("⚙️ Configuração da Análise", expanded=True):
        # Listar campanhas disponíveis
        campaigns = sorted(df['campaign'].dropna().unique())
        
        if not campaigns:
            st.error("Nenhuma campanha encontrada na coluna 'campaign'")
            st.stop()
        
        selected_campaign = st.selectbox(
            "Selecione a campanha:",
            options=campaigns,
            index=0,
            help="Escolha uma campanha para análise detalhada"
        )
        
        # Selecionar métricas para análise
        available_metrics = [col for col in numeric_cols if col in df.columns]
        
        metric_options = []
        for metric in available_metrics:
            # Traduzir nomes comuns
            if metric == 'spend':
                metric_options.append(("💰 Investimento (spend)", metric))
            elif metric == 'impressions':
                metric_options.append(("👁️ Impressões", metric))
            elif metric == 'clicks':
                metric_options.append(("🖱️ Cliques", metric))
            elif 'conversion' in metric.lower():
                metric_options.append((f"🔄 {metric}", metric))
            elif 'revenue' in metric.lower():
                metric_options.append((f"💸 {metric}", metric))
            elif 'cpc' in metric.lower():
                metric_options.append((f"🎯 {metric}", metric))
            elif 'ctr' in metric.lower():
                metric_options.append((f"📊 {metric}", metric))
            elif 'roas' in metric.lower():
                metric_options.append((f"📈 {metric}", metric))
            else:
                metric_options.append((metric, metric))
        
        selected_metrics = st.multiselect(
            "Métricas para análise:",
            options=[m[0] for m in metric_options],
            default=[m[0] for m in metric_options[:3]],
            help="Selecione as métricas que deseja analisar"
        )
        
        # Converter nomes exibidos para nomes reais das colunas
        selected_metric_names = []
        for display_name in selected_metrics:
            for opt_display, opt_real in metric_options:
                if opt_display == display_name:
                    selected_metric_names.append(opt_real)
                    break
    
    # Filtrar dados da campanha selecionada
    campaign_data = df[df['campaign'] == selected_campaign].copy()
    
    if campaign_data.empty:
        st.error(f"❌ Nenhum dado encontrado para a campanha '{selected_campaign}'")
        st.stop()
    
    # Ordenar por data se disponível
    if 'date' in campaign_data.columns:
        campaign_data = campaign_data.sort_values('date')
    
    # ====================
    # DASHBOARD DE MÉTRICAS
    # ====================
    
    st.subheader(f"📊 Dashboard - {selected_campaign}")
    
    # Mostrar período se tiver data
    if 'date' in campaign_data.columns:
        start_date = campaign_data['date'].min()
        end_date = campaign_data['date'].max()
        days_active = (end_date - start_date).days + 1
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("📅 Início", start_date.strftime('%d/%m/%Y'))
        with col_info2:
            st.metric("📅 Término", end_date.strftime('%d/%m/%Y'))
        with col_info3:
            st.metric("⏱️ Dias ativa", days_active)
    
    # Métricas principais em cards
    st.subheader("📈 Métricas Principais")
    
    # Mostrar até 6 métricas em linha
    cols = st.columns(min(6, len(selected_metric_names)))
    
    for idx, metric in enumerate(selected_metric_names[:6]):
        with cols[idx % len(cols)]:
            if metric in campaign_data.columns:
                total_value = campaign_data[metric].sum()
                avg_value = campaign_data[metric].mean()
                
                # Formatar valores
                if total_value >= 1000000:
                    display_total = f"{total_value/1000000:.1f}M"
                elif total_value >= 1000:
                    display_total = f"{total_value/1000:.1f}K"
                else:
                    display_total = f"{total_value:.0f}"
                
                # Nome amigável
                metric_name = next((m[0] for m in metric_options if m[1] == metric), metric)
                
                st.metric(
                    metric_name,
                    display_total,
                    f"Média: {avg_value:.2f}"
                )
    
    # ====================
    # GRÁFICOS
    # ====================
    
    st.subheader("📊 Visualizações")
    
    tab_charts, tab_table = st.tabs(["Gráficos", "Dados"])
    
    with tab_charts:
        # Gráfico de linhas para métricas ao longo do tempo
        if 'date' in campaign_data.columns and selected_metric_names:
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            
            for idx, metric in enumerate(selected_metric_names[:5]):  # Limitar a 5 métricas
                if metric in campaign_data.columns:
                    metric_name = next((m[0] for m in metric_options if m[1] == metric), metric)
                    
                    # Normalizar para melhor visualização
                    if campaign_data[metric].max() > 0:
                        normalized = campaign_data[metric] / campaign_data[metric].max()
                    else:
                        normalized = campaign_data[metric]
                    
                    fig.add_trace(go.Scatter(
                        x=campaign_data['date'],
                        y=normalized,
                        name=metric_name,
                        mode='lines+markers',
                        line=dict(color=colors[idx % len(colors)], width=2),
                        yaxis='y' if idx == 0 else f'y{idx+1}'
                    ))
            
            fig.update_layout(
                title=f"Evolução das Métricas - {selected_campaign}",
                xaxis_title="Data",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de barras para métricas totais
        if selected_metric_names:
            fig2 = go.Figure()
            
            totals = []
            metric_names_display = []
            
            for metric in selected_metric_names[:8]:  # Limitar a 8 métricas
                if metric in campaign_data.columns:
                    totals.append(campaign_data[metric].sum())
                    metric_names_display.append(next((m[0] for m in metric_options if m[1] == metric), metric))
            
            if totals:
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
    
    with tab_table:
        # Tabela com dados brutos
        st.subheader("📋 Dados da Campanha")
        
        # Selecionar colunas para mostrar
        display_cols = ['date'] if 'date' in campaign_data.columns else []
        display_cols.extend(selected_metric_names)
        
        # Manter apenas colunas que existem
        display_cols = [col for col in display_cols if col in campaign_data.columns]
        
        if display_cols:
            st.dataframe(
                campaign_data[display_cols].sort_values(
                    'date' if 'date' in display_cols else display_cols[0],
                    ascending=False
                ),
                use_container_width=True,
                height=400
            )
            
            # Estatísticas descritivas
            st.subheader("📊 Estatísticas Descritivas")
            
            stats_df = campaign_data[selected_metric_names].describe().T
            st.dataframe(stats_df, use_container_width=True)
    
    # ====================
    # ANÁLISE COM IA (OPCIONAL)
    # ====================
    
    if gemini_api_key and modelo_texto:
        st.subheader("🤖 Análise com IA")
        
        if st.button("🔍 Gerar Insights com IA", type="secondary"):
            with st.spinner("Analisando dados com IA..."):
                try:
                    # Preparar dados para a IA
                    summary_stats = {}
                    for metric in selected_metric_names[:5]:  # Limitar a 5 métricas
                        if metric in campaign_data.columns:
                            summary_stats[metric] = {
                                'total': campaign_data[metric].sum(),
                                'media': campaign_data[metric].mean(),
                                'min': campaign_data[metric].min(),
                                'max': campaign_data[metric].max(),
                                'tendencia': 'crescente' if len(campaign_data) > 1 and campaign_data[metric].iloc[-1] > campaign_data[metric].iloc[0] else 'decrescente'
                            }
                    
                    prompt = f"""
                    Analise os dados desta campanha de marketing:
                    
                    Campanha: {selected_campaign}
                    Período: {f"{start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}" if 'date' in campaign_data.columns else "Período não especificado"}
                    Total de registros: {len(campaign_data)}
                    
                    Métricas analisadas:
                    {json.dumps(summary_stats, indent=2)}
                    
                    Forneça:
                    1. Um resumo executivo (2-3 parágrafos)
                    2. Pontos fortes identificados
                    3. Pontos de atenção
                    4. 3 recomendações para melhorar
                    
                    Seja conciso e prático.
                    """
                    
                    response = modelo_texto.generate_content(prompt)
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Erro na análise com IA: {str(e)}")
    else:
        st.info("ℹ️ Configure a chave da API Gemini para usar análise com IA")
    
    # ====================
    # DOWNLOAD DE DADOS
    # ====================
    
    st.subheader("📥 Exportar Dados")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # CSV da campanha
        csv_data = campaign_data.to_csv(index=False)
        st.download_button(
            label="💾 Baixar Dados da Campanha (CSV)",
            data=csv_data,
            file_name=f"campanha_{selected_campaign}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col_dl2:
        # Relatório simples
        report_text = f"""
        RELATÓRIO DE CAMPANHA - {selected_campaign}
        Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        
        PERÍODO: {f"{start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}" if 'date' in campaign_data.columns else "Não disponível"}
        DIAS ATIVA: {days_active if 'date' in campaign_data.columns else "N/A"}
        
        MÉTRICAS:
        """
        
        for metric in selected_metric_names[:10]:
            if metric in campaign_data.columns:
                total = campaign_data[metric].sum()
                avg = campaign_data[metric].mean()
                report_text += f"\n- {metric}: Total={total:,.2f}, Média={avg:,.2f}"
        
        st.download_button(
            label="📄 Baixar Relatório (TXT)",
            data=report_text,
            file_name=f"relatorio_{selected_campaign}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

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
