import os
import pandas as pd
import streamlit as st
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


def init_gemini():
    if not GEMINI_AVAILABLE:
        st.sidebar.warning("⚠️ Biblioteca Gemini não disponível")
        return None

    gemini_api_key = os.getenv("GEM_API_KEY")

    if not gemini_api_key:
        for key_name in ["GEN_API_KEY", "GEN_API_KEY2", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
            key_value = os.getenv(key_name)
            if key_value:
                gemini_api_key = key_value
                break

    if not gemini_api_key and hasattr(st, 'secrets'):
        for key_name in ["GEM_API_KEY", "GEN_API_KEY", "GEN_API_KEY2", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
            if key_name in st.secrets:
                gemini_api_key = st.secrets[key_name]
                break

    if not gemini_api_key:
        st.sidebar.info("ℹ️ Gemini não configurado")
        return None

    try:
        genai.configure(api_key=gemini_api_key)
        modelo = genai.GenerativeModel("gemini-2.5-flash")
        st.sidebar.success("✅ Gemini configurado!")
        return modelo
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro Gemini: {str(e)[:50]}")
        return None


def generate_gemini_analysis(modelo_texto, df_filtered, analysis_type="overall", user_instructions=""):
    if not modelo_texto:
        return "⚠️ Gemini não configurado."

    if df_filtered.empty:
        return "❌ Nenhum dado disponível."

    try:
        num_records = len(df_filtered)
        has_campaigns = 'campaign' in df_filtered.columns
        has_date = 'date' in df_filtered.columns

        date_info = "N/A"
        if has_date and not df_filtered['date'].isna().all():
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

        general_info = f"""
        ## 📊 CONTEXTO GERAL:
        - **Total de registros:** {num_records:,}
        - **Período:** {date_info}
        - **Colunas disponíveis:** {len(df_filtered.columns)}
        - **Campanhas:** {df_filtered['campaign'].nunique() if has_campaigns else 'N/A'}
        """

        campaign_analysis = ""
        if has_campaigns:
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

            metricas_com_dados = []
            for metric in important_metrics[:8]:
                if metric in df_filtered.columns:
                    try:
                        metric_data = pd.to_numeric(df_filtered[metric], errors='coerce').dropna()
                        total = metric_data.sum()
                        # só inclui métricas que têm valores reais (não todos zero)
                        if len(metric_data) > 0 and total != 0:
                            metricas_com_dados.append((metric, total, metric_data.mean()))
                    except:
                        continue

            if metricas_com_dados:
                metric_analysis = "## 📈 MÉTRICAS COM DADOS:\n"
                for metric, total, avg in metricas_com_dados:
                    metric_analysis += f"\n**{metric}**: Total {total:,.2f} | Média {avg:,.2f}\n"

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

        try:
            sample_df = df_filtered.head(20).copy()
            for col in sample_df.columns:
                sample_df[col] = sample_df[col].astype(str)
            sample_data = sample_df.to_string()
        except:
            sample_data = "Erro ao gerar amostra"

        focus_map = {
            "overall": "ANÁLISE GERAL DE PERFORMANCE",
            "trends": "ANÁLISE DE TENDÊNCIAS",
            "efficiency": "ANÁLISE DE EFICIÊNCIA",
        }
        focus_text = focus_map.get(analysis_type, "ANÁLISE COMPLETA")

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

        Analise os dados acima e crie um relatório executivo em português.
        Inclua **apenas as seções que tiverem dados reais** — não invente valores nem
        mencione métricas ausentes. Se uma seção não tiver dados suficientes, omita-a.

        Seções possíveis (use somente as pertinentes):
        - **📈 RESUMO EXECUTIVO** — sempre incluir
        - **🎯 CAMPANHAS** — somente se houver dados de campanha
        - **💰 FINANCEIRO** — somente se houver spend/revenue/roas com valores não-zero
        - **📊 MÉTRICAS** — apenas as métricas presentes na seção "MÉTRICAS COM DADOS" acima
        - **🔍 INSIGHTS** — 3 a 5 conclusões baseadas estritamente nos números fornecidos
        - **🚀 RECOMENDAÇÕES** — ações práticas derivadas do que foi analisado
        - **📅 PRÓXIMOS PASSOS** — somente se houver base suficiente nos dados

        {f"Foco: {user_instructions}" if user_instructions else ""}
        Seja direto e específico com os números reais.
        """

        with st.spinner("🤖 Gemini está analisando..."):
            response = modelo_texto.generate_content(prompt)
            return response.text

    except Exception as e:
        return f"❌ Erro: {str(e)[:200]}"


def generate_slides_description(modelo_texto, gemini_analysis_report, user_instructions=""):
    """Gera descrição do que colocar em cada slide baseada no relatório Gemini"""
    if not modelo_texto:
        return "⚠️ Gemini não configurado."

    if not gemini_analysis_report or gemini_analysis_report.startswith("❌") or gemini_analysis_report.startswith("⚠️"):
        return "❌ Nenhuma análise disponível para criar slides."

    try:
        prompt = f"""
        # 🎬 ESTRUTURA DE APRESENTAÇÃO — RELATÓRIO DE PERFORMANCE

        ## ANÁLISE BASE:
        {gemini_analysis_report}

        ## DIRETRIZES ADICIONAIS:
        {user_instructions if user_instructions else "Apresentação executiva de performance de campanhas digitais."}

        ## TAREFA:
        Com base na análise acima, crie o roteiro completo de uma apresentação de performance.
        Para cada slide descreva exatamente o que colocar — título, conteúdo, dados específicos e tipo de visualização.
        Use apenas os dados presentes na análise. Não invente números.

        Formato de cada slide:

        ---
        **SLIDE [N] — [TÍTULO DO SLIDE]**
        - **Tipo:** [capa / agenda / dados / gráfico / insights / recomendações]
        - **Conteúdo principal:** [o que vai no corpo do slide — bullets, números, comparativos]
        - **Visualização sugerida:** [gráfico de barras / pizza / linha / tabela / ícones / texto / nenhuma]
        - **Dado em destaque:** [o número ou frase mais importante deste slide]
        - **Nota ao apresentador:** [contexto ou observação para quem vai apresentar]
        ---

        Estrutura mínima esperada:
        1. Capa (cliente, período, tema)
        2. Agenda
        3. Resumo executivo (1 slide)
        4. Investimento por plataforma (se houver dados)
        5. Performance — métricas disponíveis (1 slide por bloco de métricas relacionadas)
        6. Comparativo / variações
        7. Insights principais
        8. Recomendações
        9. Próximos passos

        Inclua apenas os slides que tiverem dados para sustentar. Omita seções sem informação.
        """

        with st.spinner("🤖 Gerando descrição dos slides..."):
            response = modelo_texto.generate_content(prompt)
            return response.text

    except Exception as e:
        return f"❌ Erro ao gerar slides: {str(e)[:200]}"


def generate_yoy_analysis(modelo_texto, yoy_data, context=""):
    if not modelo_texto:
        return "⚠️ Gemini não configurado."

    try:
        prompt = f"""
        # 📊 ANÁLISE YoY (Year-over-Year) DE PERFORMANCE

        ## 📋 DADOS FORNECIDOS (texto livre — extraia o que encontrar):
        {yoy_data}

        ## 📝 CONTEXTO:
        {context if context else "Não informado."}

        ## 🎯 TAREFA:

        Analise os dados acima em português. Trabalhe apenas com as métricas e períodos
        que estiverem presentes — não invente dados ausentes nem mencione métricas que
        não foram fornecidas.

        Estruture a resposta com as seções que fizerem sentido dado o que foi informado:

        - **📊 RESUMO EXECUTIVO** — principais movimentos do período
        - **💰 INVESTIMENTO** — apenas se houver dados de investimento/budget
        - **📈 PERFORMANCE** — apenas as métricas presentes nos dados
        - **🔍 INSIGHTS** — 3 a 5 conclusões baseadas estritamente nos números fornecidos
        - **🎯 RECOMENDAÇÕES** — ações práticas derivadas dos dados disponíveis

        Se algum dado estiver incompleto ou ambíguo, aponte isso ao invés de assumir valores.
        """

        with st.spinner("🤖 Gemini está analisando os dados YoY..."):
            response = modelo_texto.generate_content(prompt)
            return response.text

    except Exception as e:
        return f"❌ Erro na análise YoY: {str(e)[:200]}"


def review_briefing(modelo_texto, pasted_text):
    """
    Lê o briefing e devolve:
    - um resumo do que foi entendido
    - perguntas necessárias antes de gerar a análise
    """
    if not modelo_texto:
        return "⚠️ Gemini não configurado."

    try:
        prompt = f"""
        Você é um analista de mídia digital lendo um briefing antes de gerar um relatório.

        ## BRIEFING RECEBIDO:
        {pasted_text}

        ## TAREFA:
        1. Faça um **resumo em 2-3 linhas** do que você entendeu: cliente, período, métricas presentes.
        2. Liste as **perguntas necessárias** para gerar uma análise completa e precisa.
           - Inclua apenas perguntas cujas respostas não estão no briefing.
           - Priorize: objetivo da campanha, métricas ausentes relevantes, contexto de negócio.
           - Máximo 5 perguntas. Se tudo estiver claro, diga isso explicitamente.

        Formato esperado:

        **✅ O que entendi:**
        [resumo]

        **❓ Perguntas antes de gerar:**
        1. [pergunta]
        2. [pergunta]
        ...

        Se o briefing for suficiente para análise, finalize com:
        **Pronto para gerar** — nenhuma pergunta adicional necessária.
        """

        with st.spinner("🔍 Lendo briefing..."):
            response = modelo_texto.generate_content(prompt)
            return response.text

    except Exception as e:
        return f"❌ Erro: {str(e)[:200]}"


def analyze_pasted_data(modelo_texto, pasted_text, context=""):
    """Analisa dados colados com Gemini — aceita qualquer formato de entrada"""
    if not modelo_texto:
        return "⚠️ Gemini não configurado.", None

    try:
        prompt = f"""
        # 📊 ANÁLISE DE BRIEFING / DADOS DE PERFORMANCE

        ## 📋 DADOS FORNECIDOS:
        {pasted_text}

        ## 📝 FOCO / INSTRUÇÕES:
        {context if context else "Análise geral — extraia o que for relevante."}

        ## 🎯 TAREFA:

        Analise o conteúdo acima em português. O input pode ser texto corrido, tabela,
        e-mail, números soltos ou qualquer combinação — interprete o que encontrar.

        Trabalhe **apenas com o que estiver presente**. Não invente dados ausentes.

        Estruture a resposta com as seções que tiverem informação suficiente:

        - **📊 RESUMO** — o que são esses dados e o que revelam de imediato
        - **📈 MÉTRICAS IDENTIFICADAS** — tabela compacta com os números encontrados e suas variações (se houver comparativo)
        - **🔍 INSIGHTS** — 3 a 5 conclusões diretas baseadas nos números
        - **🎯 RECOMENDAÇÕES** — ações práticas derivadas do que foi analisado
        - **❓ DADOS AUSENTES** — se alguma informação importante não foi fornecida, liste aqui

        Seja direto e específico. Use os números exatos do input.
        """

        with st.spinner("🤖 Analisando..."):
            response = modelo_texto.generate_content(prompt)

        return response.text, None

    except Exception as e:
        return f"❌ Erro na análise: {str(e)[:200]}", None
