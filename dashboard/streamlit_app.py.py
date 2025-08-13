# Dashboard Interativo - Employee Attrition
# Streamlit App para análise e predições

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class AttritionDashboard:
    """Dashboard interativo para análise de Employee Attrition"""
    
    def __init__(self):
        self.load_data()
        self.setup_sidebar()
    
    @st.cache_data
    def load_data(_self):
        """Carrega dados e modelo (com cache)"""
        try:
            # Carregar dados processados
            df_original = pd.read_csv('IBM_Fn-UseC_-HR-Employee-Attrition.csv')
            
            # Simular dados para demonstração se não encontrar arquivos processados
            _self.df = df_original.copy()
            _self.df['Attrition_Binary'] = (_self.df['Attrition'] == 'Yes').astype(int)
            _self.df['Risk_Score'] = np.random.beta(2, 5, len(_self.df))  # Simular scores de risco
            
            # Adicionar predições simuladas para demonstração
            _self.df['Predicted_Attrition'] = (_self.df['Risk_Score'] > 0.3).astype(int)
            
            return True
            
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            return False
    
    def setup_sidebar(self):
        """Configura sidebar com filtros"""
        st.sidebar.header("🎛️ Filtros e Configurações")
        
        # Filtros
        departments = ['Todos'] + list(self.df['Department'].unique())
        self.selected_dept = st.sidebar.selectbox("Departamento", departments)
        
        age_range = st.sidebar.slider(
            "Faixa Etária", 
            int(self.df['Age'].min()), 
            int(self.df['Age'].max()), 
            (int(self.df['Age'].min()), int(self.df['Age'].max()))
        )
        
        salary_range = st.sidebar.slider(
            "Faixa Salarial ($)", 
            int(self.df['MonthlyIncome'].min()), 
            int(self.df['MonthlyIncome'].max()), 
            (int(self.df['MonthlyIncome'].min()), int(self.df['MonthlyIncome'].max()))
        )
        
        # Aplicar filtros
        filtered_df = self.df.copy()
        
        if self.selected_dept != 'Todos':
            filtered_df = filtered_df[filtered_df['Department'] == self.selected_dept]
        
        filtered_df = filtered_df[
            (filtered_df['Age'] >= age_range[0]) & 
            (filtered_df['Age'] <= age_range[1])
        ]
        
        filtered_df = filtered_df[
            (filtered_df['MonthlyIncome'] >= salary_range[0]) & 
            (filtered_df['MonthlyIncome'] <= salary_range[1])
        ]
        
        self.filtered_df = filtered_df
        
        # Métricas do filtro
        st.sidebar.markdown("---")
        st.sidebar.markdown("📊 **Dados Filtrados**")
        st.sidebar.metric("Total de Funcionários", len(filtered_df))
        st.sidebar.metric("Taxa de Attrition", f"{filtered_df['Attrition_Binary'].mean():.1%}")
    
    def create_overview_tab(self):
        """Cria tab de visão geral"""
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        total_employees = len(self.filtered_df)
        attrition_rate = self.filtered_df['Attrition_Binary'].mean()
        high_risk_employees = (self.filtered_df['Risk_Score'] > 0.7).sum()
        avg_tenure = self.filtered_df['YearsAtCompany'].mean()
        
        with col1:
            st.metric(
                "👥 Total de Funcionários", 
                f"{total_employees:,}",
                delta=f"{len(self.filtered_df) - len(self.df):+,}" if len(self.filtered_df) != len(self.df) else None
            )
        
        with col2:
            st.metric(
                "📊 Taxa de Attrition", 
                f"{attrition_rate:.1%}",
                delta=f"{attrition_rate - self.df['Attrition_Binary'].mean():+.1%}" if len(self.filtered_df) != len(self.df) else None
            )
        
        with col3:
            st.metric(
                "🚨 Alto Risco", 
                f"{high_risk_employees:,}",
                delta=f"{high_risk_employees / total_employees:.1%} do total"
            )
        
        with col4:
            st.metric(
                "⏱️ Permanência Média", 
                f"{avg_tenure:.1f} anos",
                delta=f"{avg_tenure - self.df['YearsAtCompany'].mean():+.1f}" if len(self.filtered_df) != len(self.df) else None
            )
        
        st.markdown("---")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de Attrition por Departamento
            dept_attrition = (self.filtered_df.groupby('Department')['Attrition_Binary']
                            .agg(['count', 'mean']).reset_index())
            dept_attrition.columns = ['Department', 'Total', 'Attrition_Rate']
            
            fig_dept = px.bar(
                dept_attrition, 
                x='Department', 
                y='Attrition_Rate',
                title="📊 Taxa de Attrition por Departamento",
                color='Attrition_Rate',
                color_continuous_scale='RdYlBu_r'
            )
            fig_dept.update_layout(height=400)
            st.plotly_chart(fig_dept, use_container_width=True)
        
        with col2:
            # Distribuição de Risk Score
            fig_risk = px.histogram(
                self.filtered_df, 
                x='Risk_Score',
                title="📈 Distribuição do Score de Risco",
                nbins=30,
                color_discrete_sequence=['#4ECDC4']
            )
            fig_risk.add_vline(x=0.7, line_dash="dash", line_color="red", 
                             annotation_text="Alto Risco")
            fig_risk.add_vline(x=0.3, line_dash="dash", line_color="orange", 
                             annotation_text="Médio Risco")
            fig_risk.update_layout(height=400)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Análise de Correlações
        st.subheader("🔗 Análise de Fatores de Risco")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Overtime vs Attrition
            overtime_data = (self.filtered_df.groupby('OverTime')['Attrition_Binary']
                           .mean().reset_index())
            
            fig_overtime = px.bar(
                overtime_data,
                x='OverTime',
                y='Attrition_Binary',
                title="⏰ Impacto do Overtime",
                color='Attrition_Binary',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_overtime, use_container_width=True)
        
        with col2:
            # Satisfação vs Attrition
            satisfaction_bins = pd.cut(self.filtered_df['JobSatisfaction'], 
                                     bins=4, labels=['Baixa', 'Média', 'Boa', 'Excelente'])
            sat_data = (self.filtered_df.groupby(satisfaction_bins)['Attrition_Binary']
                       .mean().reset_index())
            
            fig_sat = px.bar(
                sat_data,
                x='JobSatisfaction',
                y='Attrition_Binary',
                title="😊 Satisfação vs Attrition",
                color='Attrition_Binary',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_sat, use_container_width=True)
        
        with col3:
            # Idade vs Attrition
            age_bins = pd.cut(self.filtered_df['Age'], 
                            bins=5, labels=['<30', '30-35', '35-45', '45-55', '55+'])
            age_data = (self.filtered_df.groupby(age_bins)['Attrition_Binary']
                       .mean().reset_index())
            
            fig_age = px.bar(
                age_data,
                x='Age',
                y='Attrition_Binary',
                title="👥 Faixa Etária vs Attrition",
                color='Attrition_Binary',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_age, use_container_width=True)
    
    def create_prediction_tab(self):
        """Cria tab de predições"""
        
        st.subheader("🔮 Preditor de Attrition Individual")
        
        # Formulário para predição individual
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Idade", 18, 65, 30)
                monthly_income = st.number_input("Salário Mensal ($)", 1000, 20000, 5000)
                distance_from_home = st.number_input("Distância de Casa (km)", 1, 50, 10)
                years_at_company = st.number_input("Anos na Empresa", 0, 40, 5)
            
            with col2:
                department = st.selectbox("Departamento", self.df['Department'].unique())
                job_role = st.selectbox("Cargo", self.df['JobRole'].unique())
                marital_status = st.selectbox("Estado Civil", self.df['MaritalStatus'].unique())
                gender = st.selectbox("Gênero", ['Male', 'Female'])
            
            with col3:
                overtime = st.selectbox("Overtime", ['No', 'Yes'])
                job_satisfaction = st.slider("Satisfação no Trabalho", 1, 4, 3)
                work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
                environment_satisfaction = st.slider("Satisfação com Ambiente", 1, 4, 3)
            
            predict_button = st.form_submit_button("🎯 Calcular Risco de Attrition")
        
        if predict_button:
            # Simular predição (em produção, usaria modelo treinado)
            risk_factors = [
                overtime == 'Yes',
                job_satisfaction <= 2,
                work_life_balance <= 2,
                distance_from_home > 20,
                age < 30,
                years_at_company < 2
            ]
            
            risk_score = sum(risk_factors) / len(risk_factors)
            risk_percentage = risk_score * 100
            
            # Exibir resultado
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if risk_score > 0.7:
                    st.error(f"🚨 **ALTO RISCO**: {risk_percentage:.1f}%")
                    risk_level = "Alto"
                    color = "red"
                elif risk_score > 0.3:
                    st.warning(f"⚠️ **MÉDIO RISCO**: {risk_percentage:.1f}%")
                    risk_level = "Médio"
                    color = "orange"
                else:
                    st.success(f"✅ **BAIXO RISCO**: {risk_percentage:.1f}%")
                    risk_level = "Baixo"
                    color = "green"
            
            with col2:
                # Gráfico de gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_percentage,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Score de Risco"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # Lista de funcionários de alto risco
        st.subheader("🚨 Funcionários de Alto Risco")
        
        high_risk_df = self.filtered_df[self.filtered_df['Risk_Score'] > 0.7].copy()
        
        if len(high_risk_df) > 0:
            # Preparar dados para exibição
            display_cols = ['Age', 'Department', 'JobRole', 'MonthlyIncome', 
                          'YearsAtCompany', 'OverTime', 'JobSatisfaction', 'Risk_Score']
            
            high_risk_display = high_risk_df[display_cols].copy()
            high_risk_display['Risk_Score'] = high_risk_display['Risk_Score'].apply(lambda x: f"{x:.1%}")
            high_risk_display = high_risk_display.sort_values('Risk_Score', ascending=False)
            
            st.dataframe(
                high_risk_display.head(20),
                use_container_width=True,
                height=400
            )
            
            # Download dos dados
            csv = high_risk_df.to_csv(index=False)
            st.download_button(
                "📥 Download Lista Completa",
                csv,
                "funcionarios_alto_risco.csv",
                "text/csv"
            )
        else:
            st.info("✅ Nenhum funcionário de alto risco encontrado com os filtros atuais!")
    
    def create_analytics_tab(self):
        """Cria tab de analytics avançados"""
        
        # Análise de Segmentos
        st.subheader("📊 Análise de Segmentos")
        
        # Segmentação por risco
        risk_segments = pd.cut(
            self.filtered_df['Risk_Score'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Baixo Risco', 'Médio Risco', 'Alto Risco']
        )
        
        segment_analysis = (self.filtered_df
                          .groupby(risk_segments)
                          .agg({
                              'Age': 'mean',
                              'MonthlyIncome': 'mean',
                              'YearsAtCompany': 'mean',
                              'JobSatisfaction': 'mean',
                              'Attrition_Binary': 'mean'
                          })
                          .round(2))
        
        segment_analysis.columns = ['Idade Média', 'Salário Médio', 'Anos na Empresa', 
                                  'Satisfação Média', 'Taxa Real de Attrition']
        
        st.dataframe(segment_analysis, use_container_width=True)
        
        # Gráfico de segmentos
        col1, col2 = st.columns(2)
        
        with col1:
            segment_counts = risk_segments.value_counts()
            fig_segments = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="🎯 Distribuição de Segmentos de Risco",
                color_discrete_sequence=['#4ECDC4', '#FFD93D', '#FF6B6B']
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col2:
            # Comparação de métricas por segmento
            fig_metrics = px.bar(
                segment_analysis.reset_index(),
                x='Risk_Score',
                y='Taxa Real de Attrition',
                title="📈 Taxa de Attrition Real por Segmento",
                color='Taxa Real de Attrition',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Análise Temporal
        st.subheader("⏰ Análise Temporal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Attrition por anos na empresa
            years_analysis = (self.filtered_df
                            .groupby('YearsAtCompany')['Attrition_Binary']
                            .mean()
                            .reset_index())
            
            fig_years = px.line(
                years_analysis,
                x='YearsAtCompany',
                y='Attrition_Binary',
                title="📅 Taxa de Attrition por Anos na Empresa",
                markers=True
            )
            fig_years.update_traces(line_color='#FF6B6B', line_width=3)
            st.plotly_chart(fig_years, use_container_width=True)
        
        with col2:
            # Distribuição de tenure
            fig_tenure = px.histogram(
                self.filtered_df,
                x='YearsAtCompany',
                color='Attrition',
                title="🏢 Distribuição de Permanência na Empresa",
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig_tenure, use_container_width=True)
        
        # Matriz de Correlação Interativa
        st.subheader("🔗 Matriz de Correlação Interativa")
        
        numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 
                       'EnvironmentSatisfaction', 'WorkLifeBalance', 'Risk_Score', 'Attrition_Binary']
        
        correlation_matrix = self.filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="🎯 Matriz de Correlação das Principais Variáveis",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    def create_insights_tab(self):
        """Cria tab de insights e recomendações"""
        
        st.subheader("💡 Insights Principais")
        
        # Calcular insights baseados nos dados filtrados
        total_employees = len(self.filtered_df)
        attrition_rate = self.filtered_df['Attrition_Binary'].mean()
        high_risk_count = (self.filtered_df['Risk_Score'] > 0.7).sum()
        
        # Insights cards
        insights = [
            {
                "title": "📈 Taxa de Attrition Atual",
                "value": f"{attrition_rate:.1%}",
                "insight": f"Com {total_employees} funcionários analisados, a taxa atual está {'acima' if attrition_rate > 0.15 else 'dentro'} da média do mercado (15%)",
                "recommendation": "Implementar programa de retenção focado nos fatores de maior impacto" if attrition_rate > 0.15 else "Manter práticas atuais e monitorar continuamente"
            },
            {
                "title": "🚨 Funcionários em Risco",
                "value": f"{high_risk_count}",
                "insight": f"Representam {high_risk_count/total_employees:.1%} do total de funcionários",
                "recommendation": "Ação imediata necessária: reuniões individuais, revisão salarial e planos de desenvolvimento"
            },
            {
                "title": "💰 Impacto Financeiro",
                "value": f"${high_risk_count * 20000:,}",
                "insight": "Custo estimado de substituição se todos os funcionários de alto risco saírem",
                "recommendation": "Investimento em programa de retenção pode gerar ROI de 300-500%"
            }
        ]
        
        for insight in insights:
            with st.expander(f"{insight['title']}: {insight['value']}", expanded=True):
                st.markdown(f"**💭 Insight:** {insight['insight']}")
                st.markdown(f"**🎯 Recomendação:** {insight['recommendation']}")
        
        st.markdown("---")
        
        # Análise de Fatores Críticos
        st.subheader("🎯 Fatores Críticos Identificados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚠️ Principais Fatores de Risco")
            
            # Calcular correlações com attrition
            risk_factors = [
                ("Overtime", self.filtered_df.groupby('OverTime')['Attrition_Binary'].mean().max()),
                ("Baixa Satisfação", (self.filtered_df[self.filtered_df['JobSatisfaction'] <= 2]['Attrition_Binary'].mean())),
                ("Distância Casa", (self.filtered_df[self.filtered_df['DistanceFromHome'] > 20]['Attrition_Binary'].mean() if 'DistanceFromHome' in self.filtered_df.columns else 0.3)),
                ("Pouca Experiência", (self.filtered_df[self.filtered_df['YearsAtCompany'] < 2]['Attrition_Binary'].mean())),
                ("Funcionários Jovens", (self.filtered_df[self.filtered_df['Age'] < 30]['Attrition_Binary'].mean()))
            ]
            
            risk_factors_df = pd.DataFrame(risk_factors, columns=['Fator', 'Taxa_Attrition'])
            risk_factors_df = risk_factors_df.sort_values('Taxa_Attrition', ascending=False)
            
            for _, row in risk_factors_df.head(5).iterrows():
                st.metric(
                    row['Fator'],
                    f"{row['Taxa_Attrition']:.1%}",
                    delta=f"{row['Taxa_Attrition'] - attrition_rate:+.1%} vs média"
                )
        
        with col2:
            st.markdown("### ✅ Fatores Protetivos")
            
            protective_factors = [
                ("Alta Satisfação", (self.filtered_df[self.filtered_df['JobSatisfaction'] >= 4]['Attrition_Binary'].mean())),
                ("Sem Overtime", self.filtered_df.groupby('OverTime')['Attrition_Binary'].mean().min()),
                ("Longa Permanência", (self.filtered_df[self.filtered_df['YearsAtCompany'] > 10]['Attrition_Binary'].mean())),
                ("Funcionários Sêniores", (self.filtered_df[self.filtered_df['Age'] > 45]['Attrition_Binary'].mean())),
                ("Altos Salários", (self.filtered_df[self.filtered_df['MonthlyIncome'] > self.filtered_df['MonthlyIncome'].quantile(0.75)]['Attrition_Binary'].mean()))
            ]
            
            protective_df = pd.DataFrame(protective_factors, columns=['Fator', 'Taxa_Attrition'])
            protective_df = protective_df.sort_values('Taxa_Attrition')
            
            for _, row in protective_df.head(5).iterrows():
                st.metric(
                    row['Fator'],
                    f"{row['Taxa_Attrition']:.1%}",
                    delta=f"{row['Taxa_Attrition'] - attrition_rate:+.1%} vs média",
                    delta_color="inverse"
                )
        
        st.markdown("---")
        
        # Plano de Ação
        st.subheader("📋 Plano de Ação Recomendado")
        
        action_plan = {
            "🚨 Ação Imediata (1-2 semanas)": [
                "Identificar e contatar todos os funcionários de alto risco",
                "Agendar reuniões individuais com gestores diretos",
                "Revisar casos de overtime excessivo",
                "Implementar pesquisa de pulso para funcionários em risco"
            ],
            "⚠️ Ação de Médio Prazo (1-3 meses)": [
                "Desenvolver programa de mentoria para funcionários jovens",
                "Revisar políticas de work-life balance",
                "Implementar programa de reconhecimento e feedback",
                "Analisar competitividade salarial por cargo/nível"
            ],
            "📈 Ação de Longo Prazo (3-12 meses)": [
                "Desenvolver trilhas de carreira mais claras",
                "Implementar programa de wellbeing corporativo",
                "Criar sistema de monitoramento contínuo de satisfação",
                "Estabelecer KPIs de retenção por departamento"
            ]
        }
        
        for timeframe, actions in action_plan.items():
            with st.expander(timeframe, expanded=False):
                for action in actions:
                    st.markdown(f"• {action}")
        
        # ROI Calculator
        st.markdown("---")
        st.subheader("💹 Calculadora de ROI")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            retention_rate = st.slider("Taxa de Retenção Esperada", 0.1, 0.9, 0.5)
            replacement_cost = st.number_input("Custo de Substituição ($)", 5000, 50000, 20000)
        
        with col2:
            program_cost_per_employee = st.number_input("Custo do Programa por Funcionário ($)", 500, 5000, 2000)
            
        with col3:
            # Calcular ROI
            employees_retained = high_risk_count * retention_rate
            savings = employees_retained * replacement_cost
            program_costs = high_risk_count * program_cost_per_employee
            net_savings = savings - program_costs
            roi_percentage = (net_savings / program_costs * 100) if program_costs > 0 else 0
            
            st.metric("Funcionários Retidos", f"{employees_retained:.0f}")
            st.metric("ROI Estimado", f"{roi_percentage:.0f}%")
            st.metric("Economia Líquida", f"${net_savings:,.0f}")
    
    def run_dashboard(self):
        """Executa o dashboard principal"""
        
        # Header
        st.markdown('<h1 class="main-header">👥 Employee Attrition Analytics</h1>', 
                   unsafe_allow_html=True)
        
        # Verificar se dados foram carregados
        if not hasattr(self, 'df'):
            st.error("❌ Erro ao carregar dados. Verifique se o arquivo CSV está disponível.")
            return
        
        # Tabs principais
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visão Geral", 
            "🔮 Predições", 
            "📈 Analytics Avançados", 
            "💡 Insights & Ações"
        ])
        
        with tab1:
            self.create_overview_tab()
        
        with tab2:
            self.create_prediction_tab()
        
        with tab3:
            self.create_analytics_tab()
        
        with tab4:
            self.create_insights_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #888;'>
                🚀 <b>Employee Attrition Analytics Dashboard</b><br>
                Desenvolvido com Streamlit • Dados simulados para demonstração<br>
                📧 Contato: seu.email@empresa.com
            </div>
            """, 
            unsafe_allow_html=True
        )

# ========================================
# EXECUÇÃO PRINCIPAL
# ========================================

if __name__ == "__main__":
    # Inicializar e executar dashboard
    dashboard = AttritionDashboard()
    dashboard.run_dashboard()
    
# ========================================
# INSTRUÇÕES DE EXECUÇÃO
# ========================================

"""
Para executar este dashboard:

1. Instalar dependências:
   pip install streamlit plotly pandas numpy scikit-learn

2. Executar o dashboard:
   streamlit run dashboard.py

3. Abrir no navegador:
   http://localhost:8501

4. Para deploy em produção:
   - Streamlit Cloud (grátis)
   - Heroku
   - AWS/Azure/GCP

Features do Dashboard:
✅ Visualizações interativas
✅ Filtros dinâmicos
✅ Predição individual
✅ Lista de funcionários de risco
✅ Analytics avançados
✅ Insights acionáveis
✅ Calculadora de ROI
✅ Download de relatórios
✅ Interface responsiva
✅ Design profissional
"""