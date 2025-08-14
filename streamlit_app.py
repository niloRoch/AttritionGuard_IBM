# 🔥 AttritionGuard - Employee Analytics Dashboard
# Streamlit App para análise e predições de rotatividade de funcionários
# Versão 2.0 - Completamente renovada

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import io

warnings.filterwarnings('ignore')

# ========================================
# CONFIGURAÇÃO DA PÁGINA
# ========================================

st.set_page_config(
    page_title="🔥 AttritionGuard - Employee Analytics",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CSS MODERNO E RESPONSIVO
# ========================================

def load_css():
    """Carrega CSS personalizado moderno"""
    st.markdown("""
    <style>
    /* Importar fontes modernas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Reset e configurações globais */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 2rem 0;
        padding: 1rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.4)); }
        to { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.6)); }
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 3rem;
    }
    
    /* Cards de métricas modernos */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    /* Status de risco */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.3);
        text-align: center;
        font-weight: 600;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(255, 167, 38, 0.3);
        text-align: center;
        font-weight: 600;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(102, 187, 106, 0.3);
        text-align: center;
        font-weight: 600;
    }
    
    /* Caixas de insight */
    .insight-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1);
        font-weight: 500;
    }
    
    /* Tabs personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* Sidebar melhorada */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Botões modernos */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 4rem;
        font-weight: 500;
    }
    
    /* Animações sutis */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        animation: countUp 1s ease-out;
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .metric-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
    }
    
    /* Loading spinner personalizado */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

# ========================================
# CLASSE PRINCIPAL RENOVADA
# ========================================

class AttritionDashboard:
    """Dashboard renovado para análise de Employee Attrition"""
    
    def __init__(self):
        """Inicializa o dashboard"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.df = None
            st.session_state.model = None
            st.session_state.model_metrics = {}
            st.session_state.feature_importance = None
            
        self.load_data()
    
    def generate_synthetic_data(self, n_samples=1470):
        """Gera dados sintéticos realísticos para demonstração"""
        np.random.seed(42)
        
        # Dados demográficos
        age = np.random.randint(18, 65, n_samples)
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
        marital_status = np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.32, 0.46, 0.22])
        education = np.random.randint(1, 6, n_samples)
        
        # Dados organizacionais
        departments = ['Research & Development', 'Sales', 'Human Resources']
        department = np.random.choice(departments, n_samples, p=[0.65, 0.31, 0.04])
        
        job_roles = ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                    'Manufacturing Director', 'Healthcare Representative', 'Manager',
                    'Sales Representative', 'Research Director', 'Human Resources']
        job_role = np.random.choice(job_roles, n_samples)
        
        job_level = np.random.randint(1, 6, n_samples)
        years_at_company = np.random.randint(0, 40, n_samples)
        years_in_current_role = np.minimum(years_at_company, np.random.randint(0, 18, n_samples))
        total_working_years = np.maximum(years_at_company, np.random.randint(0, 40, n_samples))
        
        # Dados financeiros
        base_salary = 2000 + job_level * 2500 + total_working_years * 120
        monthly_income = np.maximum(base_salary + np.random.normal(0, 1500, n_samples), 1009)
        
        # Dados de satisfação
        job_satisfaction = np.random.randint(1, 5, n_samples)
        environment_satisfaction = np.random.randint(1, 5, n_samples)
        work_life_balance = np.random.randint(1, 5, n_samples)
        
        # Dados operacionais
        distance_from_home = np.random.randint(1, 30, n_samples)
        overtime = np.random.choice(['No', 'Yes'], n_samples, p=[0.72, 0.28])
        business_travel = np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], 
                                         n_samples, p=[0.71, 0.19, 0.10])
        
        # Criar Attrition baseado em fatores realísticos
        attrition_prob = np.zeros(n_samples)
        
        # Fatores de risco
        attrition_prob += (age < 30) * 0.20
        attrition_prob += (job_satisfaction <= 2) * 0.25
        attrition_prob += (environment_satisfaction <= 2) * 0.15
        attrition_prob += (work_life_balance <= 2) * 0.15
        attrition_prob += (years_at_company <= 2) * 0.30
        attrition_prob += (overtime == 'Yes') * 0.25
        attrition_prob += (business_travel == 'Travel_Frequently') * 0.15
        attrition_prob += (distance_from_home > 20) * 0.12
        
        # Fatores protetivos
        attrition_prob -= (job_level >= 4) * 0.15
        attrition_prob -= (monthly_income > np.percentile(monthly_income, 75)) * 0.20
        attrition_prob -= (years_at_company > 10) * 0.25
        
        # Normalizar probabilidades
        attrition_prob = np.clip(attrition_prob, 0.02, 0.55)
        
        # Gerar attrition
        attrition = np.array(['Yes' if np.random.random() < prob else 'No' for prob in attrition_prob])
        
        # Criar DataFrame
        df = pd.DataFrame({
            'EmployeeNumber': range(1, n_samples + 1),
            'Age': age,
            'Gender': gender,
            'MaritalStatus': marital_status,
            'Education': education,
            'Department': department,
            'JobRole': job_role,
            'JobLevel': job_level,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'TotalWorkingYears': total_working_years,
            'MonthlyIncome': monthly_income.astype(int),
            'JobSatisfaction': job_satisfaction,
            'EnvironmentSatisfaction': environment_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'DistanceFromHome': distance_from_home,
            'OverTime': overtime,
            'BusinessTravel': business_travel,
            'Attrition': attrition
        })
        
        return df
    
    def prepare_data(self, df):
        """Prepara e enriquece os dados"""
        # Criar variável binária
        df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)
        
        # Calcular score de risco
        risk_score = np.zeros(len(df))
        
        # Aplicar fatores de risco
        risk_score += (df['OverTime'] == 'Yes') * 0.25
        risk_score += (df['JobSatisfaction'] <= 2) * 0.20
        risk_score += (df['Age'] < 30) * 0.15
        risk_score += (df['YearsAtCompany'] < 2) * 0.20
        risk_score += (df['DistanceFromHome'] > 20) * 0.10
        risk_score += (df['WorkLifeBalance'] <= 2) * 0.10
        
        # Normalizar
        df['Risk_Score'] = np.clip(risk_score, 0, 1)
        
        # Categorias de risco
        df['Risk_Category'] = pd.cut(df['Risk_Score'], 
                                   bins=[0, 0.3, 0.7, 1.0], 
                                   labels=['Baixo Risco', 'Médio Risco', 'Alto Risco'])
        
        # Features adicionais
        df['Income_Age_Ratio'] = df['MonthlyIncome'] / np.maximum(df['Age'], 1)
        df['Company_Experience_Ratio'] = df['YearsAtCompany'] / np.maximum(df['TotalWorkingYears'], 1)
        
        # Faixas etárias
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 100], labels=['Jovem (<30)', 'Médio (30-45)', 'Sênior (>45)'])
        
        # Faixas salariais
        df['Salary_Group'] = pd.qcut(df['MonthlyIncome'], q=4, labels=['Q1 (Baixo)', 'Q2 (Médio-Baixo)', 'Q3 (Médio-Alto)', 'Q4 (Alto)'])
        
        return df
    
    @st.cache_data
    def load_data(_self):
        """Carrega ou gera dados"""
        try:
            # Tentar carregar arquivo CSV
            uploaded_file = st.file_uploader("📁 Carregar arquivo CSV (Opcional)", type=['csv'])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success("✅ Dados carregados com sucesso!")
            else:
                # Gerar dados sintéticos
                df = _self.generate_synthetic_data()
                st.info("ℹ️ Usando dados sintéticos para demonstração")
            
            # Preparar dados
            df = _self.prepare_data(df)
            
            # Salvar no session state
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            return df
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {str(e)}")
            return None
    
    def train_model(self):
        """Treina modelo de ML"""
        if st.session_state.df is None:
            return
        
        try:
            df = st.session_state.df
            
            # Preparar features
            features = ['Age', 'JobLevel', 'YearsAtCompany', 'MonthlyIncome', 
                       'JobSatisfaction', 'WorkLifeBalance', 'DistanceFromHome']
            
            # One-hot encoding para categóricas
            df_encoded = pd.get_dummies(df[['OverTime', 'BusinessTravel', 'Department', 'Gender']], prefix_sep='_')
            
            # Combinar features
            X = pd.concat([df[features], df_encoded], axis=1)
            y = df['Attrition_Binary']
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Treinar modelo
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # Predições
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métricas
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            # AUC-ROC
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                metrics['auc_roc'] = auc(fpr, tpr)
            else:
                metrics['auc_roc'] = 0.5
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Salvar no session state
            st.session_state.model = model
            st.session_state.model_metrics = metrics
            st.session_state.feature_importance = feature_importance
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred_proba = y_pred_proba
            
        except Exception as e:
            st.error(f"❌ Erro ao treinar modelo: {str(e)}")
    
    def setup_sidebar(self):
        """Configura sidebar com filtros"""
        st.sidebar.markdown("## 🎛️ Filtros Interativos")
        st.sidebar.markdown("---")
        
        if not st.session_state.data_loaded or st.session_state.df is None:
            st.sidebar.error("❌ Dados não carregados")
            return None
        
        df = st.session_state.df
        filtered_df = df.copy()
        
        # Filtro por departamento
        departments = ['Todos'] + sorted(df['Department'].unique().tolist())
        selected_dept = st.sidebar.selectbox("🏢 Departamento", departments, key="dept_filter")
        if selected_dept != 'Todos':
            filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
        
        # Filtro por idade
        age_range = st.sidebar.slider(
            "👥 Faixa Etária", 
            int(df['Age'].min()), 
            int(df['Age'].max()), 
            (int(df['Age'].min()), int(df['Age'].max())),
            key="age_filter"
        )
        filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
        
        # Filtro por salário
        salary_range = st.sidebar.slider(
            "💰 Faixa Salarial ($)", 
            int(df['MonthlyIncome'].min()), 
            int(df['MonthlyIncome'].max()), 
            (int(df['MonthlyIncome'].min()), int(df['MonthlyIncome'].max())),
            step=500,
            key="salary_filter"
        )
        filtered_df = filtered_df[(filtered_df['MonthlyIncome'] >= salary_range[0]) & (filtered_df['MonthlyIncome'] <= salary_range[1])]
        
        # Filtro por categoria de risco
        risk_categories = ['Todos'] + list(df['Risk_Category'].cat.categories)
        selected_risk = st.sidebar.selectbox("⚠️ Categoria de Risco", risk_categories, key="risk_filter")
        if selected_risk != 'Todos':
            filtered_df = filtered_df[filtered_df['Risk_Category'] == selected_risk]
        
        # Estatísticas dos filtros
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Estatísticas Filtradas")
        
        if len(filtered_df) > 0:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("👥 Total", f"{len(filtered_df):,}")
            with col2:
                attrition_rate = filtered_df['Attrition_Binary'].mean()
                st.metric("📊 Attrition", f"{attrition_rate:.1%}")
            
            high_risk = (filtered_df['Risk_Score'] > 0.7).sum()
            st.sidebar.metric("🚨 Alto Risco", f"{high_risk:,}")
        
        # Data de atualização
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"🕒 **Atualizado em:** {datetime.datetime.now().strftime('%d/%m/%Y às %H:%M')}")
        
        return filtered_df
    
    def create_overview_tab(self, df):
        """Cria aba de visão geral"""
        st.header("📊 Visão Geral - Employee Attrition")
        
        if df is None or len(df) == 0:
            st.warning("⚠️ Nenhum dado disponível para exibir")
            return
        
        # KPIs principais
        col1, col2, col3, col4 = st.columns(4)
        
        total_employees = len(df)
        attrition_rate = df['Attrition_Binary'].mean()
        high_risk_employees = (df['Risk_Score'] > 0.7).sum()
        avg_tenure = df['YearsAtCompany'].mean()
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>👥 Total Funcionários</h3>
                <div class="metric-value">{:,}</div>
            </div>
            """.format(total_employees), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Taxa de Attrition</h3>
                <div class="metric-value">{:.1%}</div>
            </div>
            """.format(attrition_rate), unsafe_allow_html=True)
        
        with col3:
            risk_class = "risk-high" if high_risk_employees/total_employees > 0.2 else "risk-medium" if high_risk_employees/total_employees > 0.1 else "risk-low"
            st.markdown(f"""
            <div class="{risk_class}">
                <h3>🚨 Alto Risco</h3>
                <div class="metric-value">{high_risk_employees:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>⏱️ Permanência Média</h3>
                <div class="metric-value">{:.1f} anos</div>
            </div>
            """.format(avg_tenure), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            # Attrition por departamento
            dept_stats = df.groupby('Department')['Attrition_Binary'].agg(['count', 'mean']).reset_index()
            dept_stats.columns = ['Department', 'Total', 'Attrition_Rate']
            
            fig_dept = px.bar(
                dept_stats, 
                x='Department', 
                y='Attrition_Rate',
                title="📊 Taxa de Attrition por Departamento",
                color='Attrition_Rate',
                color_continuous_scale='RdYlBu_r',
                text='Attrition_Rate'
            )
            fig_dept.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_dept.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_dept, use_container_width=True)
        
        with col2:
            # Distribuição do score de risco
            fig_risk = px.histogram(
                df, 
                x='Risk_Score',
                title="📈 Distribuição do Score de Risco",
                nbins=30,
                color_discrete_sequence=['#667eea']
            )
            fig_risk.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Alto Risco")
            fig_risk.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Médio Risco")
            fig_risk.update_layout(height=400)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Análise de fatores de risco
        st.subheader("🎯 Análise de Fatores de Risco")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Impacto do overtime
            overtime_stats = df.groupby('OverTime')['Attrition_Binary'].mean()
            fig_overtime = px.bar(
                x=overtime_stats.index,
                y=overtime_stats.values,
                title="⏰ Impacto do Overtime",
                color=overtime_stats.values,
                color_continuous_scale='RdYlBu_r'
            )
            fig_overtime.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            fig_overtime.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_overtime, use_container_width=True)
        
        with col2:
            # Satisfação vs attrition
            satisfaction_stats = df.groupby('JobSatisfaction')['Attrition_Binary'].mean()
            fig_sat = px.line(
                x=satisfaction_stats.index,
                y=satisfaction_stats.values,
                title="😊 Satisfação vs Attrition",
                markers=True
            )
            fig_sat.update_traces(line_color='#667eea', line_width=3)
            fig_sat.update_layout(height=300)
            st.plotly_chart(fig_sat, use_container_width=True)
        
        with col3:
            # Idade vs attrition
            age_stats = df.groupby('Age_Group')['Attrition_Binary'].mean()
            fig_age = px.bar(
                x=age_stats.index,
                y=age_stats.values,
                title="👥 Idade vs Attrition",
                color=age_stats.values,
                color_continuous_scale='RdYlBu_r'
            )
            fig_age.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            fig_age.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Insights principais
        st.subheader("💡 Insights Principais")
        
        insights = []
        
        # Insight sobre overtime
        if 'Yes' in overtime_stats.index and 'No' in overtime_stats.index:
            impact = overtime_stats['Yes'] - overtime_stats['No']
            insights.append(f"🔥 Funcionários com overtime têm {impact:.1%} mais probabilidade de sair")
        
        # Insight sobre idade
        young_attrition = df[df['Age'] < 30]['Attrition_Binary'].mean()
        senior_attrition = df[df['Age'] > 45]['Attrition_Binary'].mean()
        insights.append(f"👶 Funcionários jovens (<30) têm {young_attrition:.
