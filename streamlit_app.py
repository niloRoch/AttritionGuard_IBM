# Dashboard Interativo - Employee Attrition Analytics
# Streamlit App para análise e predições de rotatividade de funcionários

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import warnings
import datetime
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURAÇÃO DA PÁGINA
# ========================================

st.set_page_config(
    page_title="AttritionGuard - Employee Analytics",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CSS CUSTOMIZADO
# ========================================

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
        padding: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        text-align: center;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    .highlight-metric {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# CLASSE PRINCIPAL DO DASHBOARD
# ========================================

class AttritionDashboard:
    """Dashboard interativo para análise de Employee Attrition"""
    
    def __init__(self):
        self.df = None
        self.df_filtered = None
        self.model = None
        self.feature_importance = None
        self.model_metrics = {}
        self.load_data()
    
    @st.cache_data
    def load_data(_self):
        """Carrega e prepara os dados"""
        try:
            # Tentar carregar dados de diferentes locais possíveis
            possible_paths = [
                'IBM_Fn-UseC_-HR-Employee-Attrition.csv',
                'data/raw/IBM_Fn-UseC_-HR-Employee-Attrition.csv',
                'data/IBM_Fn-UseC_-HR-Employee-Attrition.csv',
                'WA_Fn-UseC_-HR-Employee-Attrition.csv',
                'data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv',
                'HR_Analytics.csv'
            ]
            
            df = None
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    st.success(f"✅ Dados carregados de: {path}")
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    st.warning(f"Erro ao ler {path}: {str(e)}")
                    continue
            
            if df is None:
                # Gerar dados sintéticos para demonstração
                df = _self.generate_synthetic_data()
                st.info("ℹ️ Usando dados sintéticos para demonstração")
            
            # Preparar dados
            df = _self.prepare_data(df)
            _self.df = df
            
            # Treinar modelo
            _self.train_models()
            
        except Exception as e:
            st.error(f"❌ Erro crítico ao carregar dados: {str(e)}")
            # Fallback para dados sintéticos em caso de erro
            _self.df = _self.generate_synthetic_data()
            _self.df = _self.prepare_data(_self.df)
            _self.train_models()
    
    def generate_synthetic_data(self):
        """Gera dados sintéticos realísticos para demonstração"""
        np.random.seed(42)
        n_samples = 1470  # Mesmo tamanho do dataset original da IBM
        
        data = {}
        
        # Variáveis demográficas
        data['Age'] = np.random.randint(18, 65, n_samples)
        data['Gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
        data['MaritalStatus'] = np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.32, 0.46, 0.22])
        data['Education'] = np.random.randint(1, 6, n_samples)
        data['EducationField'] = np.random.choice([
            'Life Sciences', 'Other', 'Medical', 'Marketing', 
            'Technical Degree', 'Human Resources'
        ], n_samples, p=[0.41, 0.16, 0.15, 0.16, 0.08, 0.04])
        
        # Variáveis organizacionais
        data['Department'] = np.random.choice([
            'Research & Development', 'Sales', 'Human Resources'
        ], n_samples, p=[0.65, 0.31, 0.04])
        
        data['JobRole'] = np.random.choice([
            'Sales Executive', 'Research Scientist', 'Laboratory Technician',
            'Manufacturing Director', 'Healthcare Representative', 'Manager',
            'Sales Representative', 'Research Director', 'Human Resources'
        ], n_samples)
        
        data['JobLevel'] = np.random.randint(1, 6, n_samples)
        data['YearsAtCompany'] = np.random.randint(0, 40, n_samples)
        data['YearsInCurrentRole'] = np.minimum(
            data['YearsAtCompany'], 
            np.random.randint(0, 18, n_samples)
        )
        data['YearsSinceLastPromotion'] = np.minimum(
            data['YearsAtCompany'],
            np.random.randint(0, 15, n_samples)
        )
        data['YearsWithCurrManager'] = np.minimum(
            data['YearsAtCompany'],
            np.random.randint(0, 17, n_samples)
        )
        data['TotalWorkingYears'] = np.maximum(
            data['YearsAtCompany'],
            np.random.randint(0, 40, n_samples)
        )
        data['NumCompaniesWorked'] = np.random.randint(0, 10, n_samples)
        
        # Variáveis financeiras
        base_salary = 2000 + data['JobLevel'] * 2000 + data['TotalWorkingYears'] * 100
        data['MonthlyIncome'] = base_salary + np.random.normal(0, 1000, n_samples)
        data['MonthlyIncome'] = np.maximum(data['MonthlyIncome'], 1009)
        
        data['HourlyRate'] = np.random.randint(30, 101, n_samples)
        data['DailyRate'] = np.random.randint(102, 1499, n_samples)
        data['MonthlyRate'] = np.random.randint(2094, 26999, n_samples)
        data['PercentSalaryHike'] = np.random.randint(11, 25, n_samples)
        data['StockOptionLevel'] = np.random.randint(0, 4, n_samples)
        
        # Variáveis de satisfação
        data['JobSatisfaction'] = np.random.randint(1, 5, n_samples)
        data['EnvironmentSatisfaction'] = np.random.randint(1, 5, n_samples)
        data['RelationshipSatisfaction'] = np.random.randint(1, 5, n_samples)
        data['WorkLifeBalance'] = np.random.randint(1, 5, n_samples)
        data['JobInvolvement'] = np.random.randint(1, 5, n_samples)
        data['PerformanceRating'] = np.random.choice([3, 4], n_samples, p=[0.85, 0.15])
        
        # Variáveis operacionais
        data['DistanceFromHome'] = np.random.randint(1, 30, n_samples)
        data['BusinessTravel'] = np.random.choice([
            'Travel_Rarely', 'Travel_Frequently', 'Non-Travel'
        ], n_samples, p=[0.71, 0.19, 0.10])
        data['OverTime'] = np.random.choice(['No', 'Yes'], n_samples, p=[0.72, 0.28])
        data['TrainingTimesLastYear'] = np.random.randint(0, 7, n_samples)
        
        # Variáveis constantes
        data['Over18'] = 'Y'
        data['EmployeeCount'] = 1
        data['StandardHours'] = 80
        data['EmployeeNumber'] = range(1, n_samples + 1)
        
        # Criar Attrition baseado em fatores realísticos
        attrition_prob = np.zeros(n_samples)
        
        # Fatores de risco
        attrition_prob += (data['Age'] < 30) * 0.15
        attrition_prob += (data['JobSatisfaction'] <= 2) * 0.20
        attrition_prob += (data['EnvironmentSatisfaction'] <= 2) * 0.15
        attrition_prob += (data['WorkLifeBalance'] <= 2) * 0.15
        attrition_prob += (data['YearsAtCompany'] <= 2) * 0.25
        attrition_prob += (np.array(data['OverTime']) == 'Yes') * 0.20
        attrition_prob += (np.array(data['BusinessTravel']) == 'Travel_Frequently') * 0.10
        attrition_prob += (data['DistanceFromHome'] > 20) * 0.10
        attrition_prob += (np.array(data['MaritalStatus']) == 'Single') * 0.05
        attrition_prob += (data['NumCompaniesWorked'] >= 4) * 0.10
        
        # Fatores protetivos
        attrition_prob -= (data['JobLevel'] >= 4) * 0.10
        attrition_prob -= (data['MonthlyIncome'] > np.percentile(data['MonthlyIncome'], 75)) * 0.15
        attrition_prob -= (data['YearsAtCompany'] > 10) * 0.20
        attrition_prob -= (data['StockOptionLevel'] >= 1) * 0.05
        
        # Normalizar probabilidades
        attrition_prob = np.clip(attrition_prob, 0.05, 0.50)
        
        # Gerar attrition
        data['Attrition'] = np.array(['Yes' if np.random.random() < prob else 'No' 
                                    for prob in attrition_prob])
        
        return pd.DataFrame(data)
    
    def prepare_data(self, df):
        """Prepara os dados para análise"""
        try:
            # Criar variável binária para Attrition
            df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)
            
            # Criar score de risco
            risk_score = np.zeros(len(df))
            
            # Fatores de risco com verificação de existência das colunas
            if 'OverTime' in df.columns:
                risk_score += (df['OverTime'] == 'Yes') * 0.25
            if 'JobSatisfaction' in df.columns:
                risk_score += (df['JobSatisfaction'] <= 2) * 0.20
            if 'Age' in df.columns:
                risk_score += (df['Age'] < 30) * 0.15
            if 'YearsAtCompany' in df.columns:
                risk_score += (df['YearsAtCompany'] < 2) * 0.20
            if 'DistanceFromHome' in df.columns:
                risk_score += (df['DistanceFromHome'] > 20) * 0.10
            if 'WorkLifeBalance' in df.columns:
                risk_score += (df['WorkLifeBalance'] <= 2) * 0.10
            
            # Normalizar entre 0 e 1
            df['Risk_Score'] = np.clip(risk_score, 0, 1)
            
            # Criar categorias de risco
            df['Risk_Category'] = pd.cut(
                df['Risk_Score'], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['Baixo Risco', 'Médio Risco', 'Alto Risco']
            )
            
            # Features adicionais
            if 'MonthlyIncome' in df.columns and 'Age' in df.columns:
                df['Income_Age_Ratio'] = df['MonthlyIncome'] / np.maximum(df['Age'], 1)
            
            if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
                df['Company_Experience_Ratio'] = df['YearsAtCompany'] / np.maximum(df['TotalWorkingYears'], 1)
            
            return df
            
        except Exception as e:
            st.error(f"Erro ao preparar dados: {str(e)}")
            return df
    
    def train_models(self):
        """Treina modelo para predição de attrition"""
        if self.df is None:
            return
        
        try:
            # Selecionar features disponíveis
            numeric_features = []
            for col in ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction',
                       'DistanceFromHome', 'EnvironmentSatisfaction', 'WorkLifeBalance',
                       'YearsInCurrentRole', 'YearsSinceLastPromotion', 'TotalWorkingYears',
                       'TrainingTimesLastYear', 'JobInvolvement', 'JobLevel']:
                if col in self.df.columns:
                    numeric_features.append(col)
            
            categorical_features = []
            for col in ['OverTime', 'BusinessTravel', 'Department', 'JobRole', 
                       'MaritalStatus', 'Gender']:
                if col in self.df.columns:
                    categorical_features.append(col)
            
            # Preparar features
            X_numeric = self.df[numeric_features].fillna(0)
            
            # One-hot encoding para categóricas
            X_categorical = pd.DataFrame()
            for col in categorical_features:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                X_categorical = pd.concat([X_categorical, dummies], axis=1)
            
            # Combinar features
            if len(X_categorical.columns) > 0:
                X = pd.concat([X_numeric, X_categorical], axis=1)
            else:
                X = X_numeric
            
            y = self.df['Attrition_Binary']
            
            # Verificar se temos dados suficientes
            if len(X) < 10 or X.shape[1] == 0:
                st.warning("Dados insuficientes para treinar modelo")
                return
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Treinar Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.model.fit(X_train, y_train)
            
            # Feature importance
            if len(X.columns) > 0:
                importances = self.model.feature_importances_
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(15)
            
            # Métricas
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            self.model_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            # AUC-ROC
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                self.model_metrics['auc_roc'] = auc(fpr, tpr)
            else:
                self.model_metrics['auc_roc'] = 0.5
            
            # Salvar dados de teste
            self.X_test = X_test
            self.y_test = y_test
            self.y_pred_proba = y_pred_proba
            
        except Exception as e:
            st.warning(f"Erro ao treinar modelo: {str(e)}")
            self.model = None
    
    def setup_sidebar(self):
        """Configura sidebar com filtros"""
        st.sidebar.markdown("## 🎛️ Filtros e Configurações")


        # Botão para voltar à home
        if st.sidebar.button("🏠 Voltar para Home"):
            st.session_state["page"] = "home"
            st.experimental_rerun()
            
        # Verificar se dados estão carregados
        if self.df is None or len(self.df) == 0:
            st.sidebar.error("❌ Dados não carregados")
            return None
        
        # Filtros com verificação de colunas
        filtered_df = self.df.copy()
        
        # Filtro por departamento
        if 'Department' in self.df.columns:
            departments = ['Todos'] + sorted(self.df['Department'].unique().tolist())
            selected_dept = st.sidebar.selectbox("🏢 Departamento", departments)
            if selected_dept != 'Todos':
                filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
        
        # Filtro por idade
        if 'Age' in self.df.columns:
            age_range = st.sidebar.slider(
                "👥 Faixa Etária", 
                int(self.df['Age'].min()), 
                int(self.df['Age'].max()), 
                (int(self.df['Age'].min()), int(self.df['Age'].max()))
            )
            filtered_df = filtered_df[
                (filtered_df['Age'] >= age_range[0]) & 
                (filtered_df['Age'] <= age_range[1])
            ]
        
        # Filtro por salário
        if 'MonthlyIncome' in self.df.columns:
            salary_range = st.sidebar.slider(
                "💰 Faixa Salarial ($)", 
                int(self.df['MonthlyIncome'].min()), 
                int(self.df['MonthlyIncome'].max()), 
                (int(self.df['MonthlyIncome'].min()), int(self.df['MonthlyIncome'].max())),
                step=500
            )
            filtered_df = filtered_df[
                (filtered_df['MonthlyIncome'] >= salary_range[0]) & 
                (filtered_df['MonthlyIncome'] <= salary_range[1])
            ]
        
        # Filtro por função
        if 'JobRole' in self.df.columns:
            job_roles = ['Todos'] + sorted(self.df['JobRole'].unique().tolist())
            selected_role = st.sidebar.selectbox("👔 Função", job_roles)
            if selected_role != 'Todos':
                filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]
        
        # Filtro por categoria de risco
        if 'Risk_Category' in filtered_df.columns:
            risk_categories = ['Todos'] + list(filtered_df['Risk_Category'].unique())
            selected_risk = st.sidebar.selectbox("⚠️ Categoria de Risco", risk_categories)
            if selected_risk != 'Todos':
                filtered_df = filtered_df[filtered_df['Risk_Category'] == selected_risk]
        
        # Métricas do filtro
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Dados Filtrados")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total", f"{len(filtered_df):,}")
        with col2:
            if len(filtered_df) > 0:
                attrition_rate = filtered_df['Attrition_Binary'].mean()
                st.metric("Attrition", f"{attrition_rate:.1%}")
        
        if 'Risk_Score' in filtered_df.columns and len(filtered_df) > 0:
            high_risk = (filtered_df['Risk_Score'] > 0.7).sum()
            st.sidebar.metric("🚨 Alto Risco", f"{high_risk:,}")
        
        # Data de atualização
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"📅 **Última atualização:** {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        self.df_filtered = filtered_df
        return filtered_df
    
    def create_overview_tab(self, filtered_df):
        """Cria tab de visão geral"""
        st.header("📊 Visão Geral - Employee Attrition")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        total_employees = len(filtered_df)
        attrition_rate = filtered_df['Attrition_Binary'].mean() if len(filtered_df) > 0 else 0
        high_risk_employees = (filtered_df['Risk_Score'] > 0.7).sum() if 'Risk_Score' in filtered_df.columns else 0
        avg_tenure = filtered_df['YearsAtCompany'].mean() if 'YearsAtCompany' in filtered_df.columns else 0
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("👥 Total Funcionários", f"{total_employees:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Taxa de Attrition", f"{attrition_rate:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            risk_class = "risk-high" if high_risk_employees / max(total_employees, 1) > 0.2 else "risk-medium" if high_risk_employees / max(total_employees, 1) > 0.1 else "risk-low"
            st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
            st.metric("🚨 Alto Risco", f"{high_risk_employees:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("⏱️ Permanência Média", f"{avg_tenure:.1f} anos")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Department' in filtered_df.columns and len(filtered_df) > 0:
                dept_stats = (filtered_df.groupby('Department')['Attrition_Binary']
                            .agg(['count', 'mean']).reset_index())
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
            else:
                st.info("📊 Dados de departamento não disponíveis")
        
        with col2:
            if 'Risk_Score' in filtered_df.columns and len(filtered_df) > 0:
                fig_risk = px.histogram(
                    filtered_df, 
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
            else:
                st.info("📈 Score de risco não calculado")
        
        # Análise de fatores de risco
        st.subheader("🎯 Principais Fatores de Risco")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'OverTime' in filtered_df.columns and len(filtered_df) > 0:
                overtime_stats = filtered_df.groupby('OverTime')['Attrition_Binary'].mean()
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
            if 'JobSatisfaction' in filtered_df.columns and len(filtered_df) > 0:
                satisfaction_labels = ['Baixa (1-2)', 'Média (3)', 'Alta (4-5)']
                sat_bins = pd.cut(filtered_df['JobSatisfaction'], 
                                bins=[0, 2.5, 3.5, 5], labels=satisfaction_labels)
                sat_stats = filtered_df.groupby(sat_bins)['Attrition_Binary'].mean()
                
                fig_sat = px.bar(
                    x=sat_stats.index,
                    y=sat_stats.values,
                    title="😊 Satisfação vs Attrition",
                    color=sat_stats.values,
                    color_continuous_scale='RdYlBu_r'
                )
                fig_sat.update_traces(texttemplate='%{y:.1%}', textposition='outside')
                fig_sat.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_sat, use_container_width=True)
        
        with col3:
            if 'Age' in filtered_df.columns and len(filtered_df) > 0:
                age_labels = ['Jovem (<30)', 'Médio (30-45)', 'Sênior (>45)']
                age_bins = pd.cut(filtered_df['Age'], 
                                bins=[0, 30, 45, 100], labels=age_labels)
                age_stats = filtered_df.groupby(age_bins)['Attrition_Binary'].mean()
                
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
        if 'OverTime' in filtered_df.columns and len(filtered_df) > 0:
            overtime_impact = filtered_df.groupby('OverTime')['Attrition_Binary'].mean()
            if 'Yes' in overtime_impact.index and 'No' in overtime_impact.index:
                impact = overtime_impact['Yes'] - overtime_impact['No']
                insights.append(f"🔥 Funcionários com overtime têm {impact:.1%} mais probabilidade de sair")
        
        if 'JobSatisfaction' in filtered_df.columns and len(filtered_df) > 0:
            low_sat = filtered_df[filtered_df['JobSatisfaction'] <= 2]['Attrition_Binary'].mean()
            high_sat = filtered_df[filtered_df['JobSatisfaction'] >= 4]['Attrition_Binary'].mean()
            insights.append(f"😔 Baixa satisfação aumenta attrition em {(low_sat - high_sat):.1%}")
        
        if 'Age' in filtered_df.columns and len(filtered_df) > 0:
            young_attrition = filtered_df[filtered_df['Age'] < 30]['Attrition_Binary'].mean()
            senior_attrition = filtered_df[filtered_df['Age'] > 45]['Attrition_Binary'].mean()
            insights.append(f"👶 Funcionários jovens (<30) têm taxa de attrition {young_attrition:.1%} vs {senior_attrition:.1%} dos sêniores")
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    def create_prediction_tab(self, filtered_df):
        """Cria tab de predições"""
        st.header("🔮 Predições de Attrition")
        
        # Verificar se modelo está disponível
        if self.model is None:
            st.warning("⚠️ Modelo não disponível. Treinando modelo...")
            self.train_models()
            if self.model is None:
                st.error("❌ Não foi possível treinar o modelo")
                return
        
        # Métricas do modelo
        st.subheader("📊 Performance do Modelo")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Acurácia", f"{self.model_metrics.get('accuracy', 0):.1%}")
        with col2:
            st.metric("🔍 Precisão", f"{self.model_metrics.get('precision', 0):.1%}")
        with col3:
            st.metric("📈 Recall", f"{self.model_metrics.get('recall', 0):.1%}")
        with col4:
            st.metric("📊 AUC-ROC", f"{self.model_metrics.get('auc_roc', 0):.3f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance
            if self.feature_importance is not None and len(self.feature_importance) > 0:
                st.subheader("🎯 Importância das Features")
                fig_importance = px.bar(
                    self.feature_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Features Mais Importantes",
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig_importance.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("Feature importance não disponível")
        
        with col2:
            # ROC Curve
            if hasattr(self, 'y_test') and hasattr(self, 'y_pred_proba'):
                st.subheader("📈 Curva ROC")
                fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC (AUC = {self.model_metrics.get("auc_roc", 0):.3f})',
                    line=dict(color='#4ECDC4', width=3)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Linha de Base',
                    line=dict(color='red', dash='dash')
                ))
                fig_roc.update_layout(
                    title="Curva ROC",
                    xaxis_title="Taxa de Falsos Positivos",
                    yaxis_title="Taxa de Verdadeiros Positivos",
                    height=400
                )
                st.plotly_chart(fig_roc, use_container_width=True)
        
        st.markdown("---")
        
        # Lista de funcionários de alto risco
        st.subheader("🚨 Funcionários de Alto Risco")
        
        if 'Risk_Score' in filtered_df.columns:
            high_risk_df = filtered_df[filtered_df['Risk_Score'] > 0.7].copy()
            
            if len(high_risk_df) > 0:
                # Selecionar colunas para exibir
                display_columns = ['EmployeeNumber']
                for col in ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany', 'Risk_Score']:
                    if col in high_risk_df.columns:
                        display_columns.append(col)
                
                # Ordenar por risk score
                high_risk_df = high_risk_df.sort_values('Risk_Score', ascending=False)
                
                # Formatação
                if 'MonthlyIncome' in high_risk_df.columns:
                    high_risk_df['MonthlyIncome'] = high_risk_df['MonthlyIncome'].apply(lambda x: f"${x:,.0f}")
                if 'Risk_Score' in high_risk_df.columns:
                    high_risk_df['Risk_Score'] = high_risk_df['Risk_Score'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    high_risk_df[display_columns].head(20),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Botão de download
                csv = high_risk_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Lista Completa",
                    data=csv,
                    file_name="funcionarios_alto_risco.csv",
                    mime="text/csv"
                )
            else:
                st.info("🎉 Nenhum funcionário classificado como alto risco!")
        
        st.markdown("---")
        
        # Preditor individual
        st.subheader("🎯 Preditor Individual")
        
        with st.expander("📝 Inserir dados do funcionário"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Idade", min_value=18, max_value=65, value=30)
                monthly_income = st.number_input("Salário Mensal ($)", min_value=1000, max_value=50000, value=5000)
                years_company = st.number_input("Anos na Empresa", min_value=0, max_value=40, value=3)
            
            with col2:
                job_satisfaction = st.selectbox("Satisfação no Trabalho", [1, 2, 3, 4], index=2)
                work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4], index=2)
                distance_home = st.number_input("Distância de Casa (km)", min_value=1, max_value=50, value=10)
            
            with col3:
                overtime = st.selectbox("Faz Overtime?", ["No", "Yes"])
                business_travel = st.selectbox("Viagem de Negócios", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
                job_level = st.selectbox("Nível do Cargo", [1, 2, 3, 4, 5], index=2)
            
            if st.button("🔮 Calcular Risco de Attrition"):
                # Calcular risk score baseado nos inputs
                risk_score = 0
                risk_score += 0.25 if overtime == "Yes" else 0
                risk_score += 0.20 if job_satisfaction <= 2 else 0
                risk_score += 0.15 if age < 30 else 0
                risk_score += 0.20 if years_company < 2 else 0
                risk_score += 0.10 if distance_home > 20 else 0
                risk_score += 0.10 if work_life_balance <= 2 else 0
                
                # Fatores protetivos
                risk_score -= 0.10 if job_level >= 4 else 0
                risk_score -= 0.15 if monthly_income > 8000 else 0
                risk_score -= 0.20 if years_company > 10 else 0
                
                risk_score = max(0, min(1, risk_score))
                
                # Exibir resultado
                col1, col2, col3 = st.columns(3)
                with col2:
                    if risk_score > 0.7:
                        st.markdown(f'<div class="risk-high"><h3>🚨 ALTO RISCO</h3><h2>{risk_score:.1%}</h2></div>', unsafe_allow_html=True)
                        st.markdown("**Recomendações:**")
                        st.markdown("- 1:1 imediato com gestor")
                        st.markdown("- Revisar carga de trabalho")
                        st.markdown("- Plano de retenção personalizado")
                    elif risk_score > 0.3:
                        st.markdown(f'<div class="risk-medium"><h3>⚠️ MÉDIO RISCO</h3><h2>{risk_score:.1%}</h2></div>', unsafe_allow_html=True)
                        st.markdown("**Recomendações:**")
                        st.markdown("- Monitoramento mensal")
                        st.markdown("- Feedback regular")
                        st.markdown("- Oportunidades de desenvolvimento")
                    else:
                        st.markdown(f'<div class="risk-low"><h3>✅ BAIXO RISCO</h3><h2>{risk_score:.1%}</h2></div>', unsafe_allow_html=True)
                        st.markdown("**Recomendações:**")
                        st.markdown("- Manter engajamento atual")
                        st.markdown("- Reconhecimento e feedback")
                        st.markdown("- Oportunidades de crescimento")
    
    def create_analytics_tab(self, filtered_df):
        """Cria tab de analytics avançados"""
        st.header("📈 Analytics Avançados")
        
        # Correlações
        st.subheader("🔗 Matrix de Correlações")
        
        # Selecionar apenas colunas numéricas
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Matrix de Correlações",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Dados insuficientes para análise de correlações")
        
        st.markdown("---")
        
        # Análises por segmento
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Attrition por Tempo de Empresa")
            if 'YearsAtCompany' in filtered_df.columns and len(filtered_df) > 0:
                tenure_bins = [0, 1, 2, 5, 10, 40]
                tenure_labels = ['<1 ano', '1-2 anos', '2-5 anos', '5-10 anos', '>10 anos']
                filtered_df['Tenure_Group'] = pd.cut(filtered_df['YearsAtCompany'], 
                                                   bins=tenure_bins, labels=tenure_labels)
                
                tenure_stats = filtered_df.groupby('Tenure_Group')['Attrition_Binary'].agg(['count', 'mean']).reset_index()
                tenure_stats.columns = ['Tenure_Group', 'Count', 'Attrition_Rate']
                
                fig_tenure = px.line(
                    tenure_stats,
                    x='Tenure_Group',
                    y='Attrition_Rate',
                    title="Taxa de Attrition por Tempo na Empresa",
                    markers=True,
                    line_shape='spline'
                )
                fig_tenure.update_traces(line_color='#FF6B6B', line_width=3)
                fig_tenure.update_layout(height=350)
                st.plotly_chart(fig_tenure, use_container_width=True)
        
        with col2:
            st.subheader("💰 Distribuição Salarial")
            if 'MonthlyIncome' in filtered_df.columns and len(filtered_df) > 0:
                fig_salary = px.box(
                    filtered_df,
                    x='Attrition',
                    y='MonthlyIncome',
                    title="Distribuição Salarial por Attrition",
                    color='Attrition',
                    color_discrete_map={'No': '#4ECDC4', 'Yes': '#FF6B6B'}
                )
                fig_salary.update_layout(height=350)
                st.plotly_chart(fig_salary, use_container_width=True)
        
        # Análise multidimensional
        st.subheader("🎯 Análise Multidimensional")
        
        if all(col in filtered_df.columns for col in ['Age', 'MonthlyIncome', 'YearsAtCompany', 'Risk_Score']):
            fig_3d = px.scatter_3d(
                filtered_df,
                x='Age',
                y='MonthlyIncome',
                z='YearsAtCompany',
                color='Risk_Score',
                size='Risk_Score',
                title="Análise 3D: Idade vs Salário vs Tempo na Empresa",
                color_continuous_scale='RdYlBu_r',
                height=500
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # Análise de segmentação
        st.subheader("🔍 Segmentação de Funcionários")
        
        if 'Risk_Category' in filtered_df.columns:
            segment_analysis = filtered_df.groupby('Risk_Category').agg({
                'Age': 'mean',
                'MonthlyIncome': 'mean',
                'YearsAtCompany': 'mean',
                'JobSatisfaction': 'mean',
                'Attrition_Binary': ['count', 'mean']
            }).round(2)
            
            segment_analysis.columns = ['Idade Média', 'Salário Médio', 'Tempo Médio (anos)', 
                                      'Satisfação Média', 'Total Funcionários', 'Taxa Attrition']
            
            st.dataframe(segment_analysis, use_container_width=True)
    
    def create_insights_tab(self, filtered_df):
        """Cria tab de insights e recomendações"""
        st.header("💡 Insights & Recomendações")
        
        # Insights baseados em dados
        st.subheader("🔍 Principais Descobertas")
        
        insights = []
        
        # Análise de overtime
        if 'OverTime' in filtered_df.columns and len(filtered_df) > 0:
            overtime_analysis = filtered_df.groupby('OverTime')['Attrition_Binary'].agg(['count', 'mean'])
            if 'Yes' in overtime_analysis.index and 'No' in overtime_analysis.index:
                overtime_impact = overtime_analysis.loc['Yes', 'mean'] - overtime_analysis.loc['No', 'mean']
                insights.append({
                    'title': '⏰ Impacto do Overtime',
                    'finding': f'Funcionários com overtime têm {overtime_impact:.1%} mais probabilidade de sair',
                    'action': 'Implementar políticas de controle de horas extras e monitoramento de burnout'
                })
        
        # Análise de satisfação
        if 'JobSatisfaction' in filtered_df.columns and len(filtered_df) > 0:
            low_satisfaction = filtered_df[filtered_df['JobSatisfaction'] <= 2]['Attrition_Binary'].mean()
            high_satisfaction = filtered_df[filtered_df['JobSatisfaction'] >= 4]['Attrition_Binary'].mean()
            satisfaction_impact = low_satisfaction - high_satisfaction
            insights.append({
                'title': '😊 Satisfação no Trabalho',
                'finding': f'Funcionários com baixa satisfação têm {satisfaction_impact:.1%} mais chance de sair',
                'action': 'Implementar pesquisas regulares de satisfação e programas de melhoria do ambiente de trabalho'
            })
        
        # Análise de idade
        if 'Age' in filtered_df.columns and len(filtered_df) > 0:
            young_attrition = filtered_df[filtered_df['Age'] < 30]['Attrition_Binary'].mean()
            senior_attrition = filtered_df[filtered_df['Age'] > 45]['Attrition_Binary'].mean()
            insights.append({
                'title': '👥 Fator Idade',
                'finding': f'Funcionários jovens (<30) têm {young_attrition:.1%} de attrition vs {senior_attrition:.1%} dos sêniores',
                'action': 'Desenvolver programas de mentoria e planos de carreira para jovens profissionais'
            })
        
        # Exibir insights
        for insight in insights:
            with st.expander(f"📊 {insight['title']}"):
                st.markdown(f"**Descoberta:** {insight['finding']}")
                st.markdown(f"**Recomendação:** {insight['action']}")
        
        st.markdown("---")
        
        # Recomendações estratégicas
        st.subheader("🎯 Recomendações Estratégicas")
        
        recommendations = [
            {
                'category': '🚨 Ação Imediata',
                'items': [
                    'Identificar e conversar com funcionários de alto risco (score > 70%)',
                    'Revisar políticas de overtime e carga de trabalho',
                    'Implementar programa de feedback regular com gestores'
                ]
            },
            {
                'category': '📈 Médio Prazo (3-6 meses)',
                'items': [
                    'Desenvolver programa de retenção de talentos jovens',
                    'Implementar pesquisas trimestrais de satisfação',
                    'Criar planos de desenvolvimento individual para funcionários-chave',
                    'Revisar estrutura salarial e benefícios'
                ]
            },
            {
                'category': '🎓 Longo Prazo (6-12 meses)',
                'items': [
                    'Estabelecer programa de mentoria inter-geracional',
                    'Criar trilhas de carreira claras por departamento',
                    'Implementar política de work-life balance mais robusta',
                    'Desenvolver sistema de reconhecimento e recompensas'
                ]
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"{rec['category']}"):
                for item in rec['items']:
                    st.markdown(f"• {item}")
        
        st.markdown("---")
        
        # Calculadora de ROI
        st.subheader("💰 Calculadora de ROI - Programa de Retenção")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Parâmetros do Negócio**")
            num_employees = st.number_input("Número total de funcionários", 
                                          min_value=1, value=len(filtered_df))
            avg_salary = st.number_input("Salário médio anual ($)", 
                                       min_value=10000, value=60000)
            current_attrition = st.slider("Taxa atual de attrition (%)", 
                                        0.0, 50.0, filtered_df['Attrition_Binary'].mean() * 100)
            
        with col2:
            st.markdown("**🎯 Programa de Retenção**")
            program_cost = st.number_input("Custo do programa anual ($)", 
                                         min_value=1000, value=50000)
            target_reduction = st.slider("Redução esperada de attrition (%)", 
                                       0.0, 50.0, 25.0)
            replacement_cost_pct = st.slider("Custo de substituição (% do salário)", 
                                           50.0, 200.0, 100.0)
        
        # Cálculos
        current_attrition_rate = current_attrition / 100
        new_attrition_rate = current_attrition_rate * (1 - target_reduction / 100)
        
        current_losses = num_employees * current_attrition_rate
        new_losses = num_employees * new_attrition_rate
        employees_retained = current_losses - new_losses
        
        replacement_cost = avg_salary * (replacement_cost_pct / 100)
        savings = employees_retained * replacement_cost
        roi = ((savings - program_cost) / program_cost) * 100
        
        # Resultados
        st.markdown("---")
        st.subheader("📊 Resultados da Análise de ROI")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Economia Anual", f"${savings:,.0f}")
        
        with col2:
            st.metric("👥 Funcionários Retidos", f"{employees_retained:.0f}")
        
        with col3:
            color = "normal" if roi > 0 else "inverse"
            st.metric("📈 ROI", f"{roi:.0f}%", delta_color=color)
        
        # Análise de sensibilidade
        if roi > 0:
            st.success(f"🎉 O programa é viável! Para cada $1 investido, você economiza ${(savings/program_cost):.2f}")
        else:
            st.warning("⚠️ O programa pode não ser viável com os parâmetros atuais. Considere ajustar os custos ou metas.")
        
        # Gráfico de sensibilidade
        st.subheader("📈 Análise de Sensibilidade")
        
        reduction_scenarios = np.arange(5, 51, 5)
        roi_scenarios = []
        
        for reduction in reduction_scenarios:
            scenario_new_rate = current_attrition_rate * (1 - reduction / 100)
            scenario_retained = num_employees * (current_attrition_rate - scenario_new_rate)
            scenario_savings = scenario_retained * replacement_cost
            scenario_roi = ((scenario_savings - program_cost) / program_cost) * 100
            roi_scenarios.append(scenario_roi)
        
        fig_sensitivity = px.line(
            x=reduction_scenarios,
            y=roi_scenarios,
            title="ROI vs Redução de Attrition",
            labels={'x': 'Redução de Attrition (%)', 'y': 'ROI (%)'}
        )
        fig_sensitivity.add_hline(y=0, line_dash="dash", line_color="red", 
                                annotation_text="Break-even")
        fig_sensitivity.update_traces(line_color='#4ECDC4', line_width=3)
        st.plotly_chart(fig_sensitivity, use_container_width=True)
    
    def run(self):
        """Executa o dashboard"""
        # Cabeçalho principal
        st.markdown('<div class="main-header">🔥 AttritionGuard - Employee Analytics</div>', 
                   unsafe_allow_html=True)
        
        # Subtitle
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem; color: #666; font-size: 1.2rem;'>
        Análise Preditiva e Insights Acionáveis para Retenção de Talentos
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Setup sidebar e obter dados filtrados
        filtered_df = self.setup_sidebar()
        
        # Verificar se há dados
        if filtered_df is None or len(filtered_df) == 0:
            st.error("⚠️ Nenhum dado disponível com os filtros aplicados.")
            return
        
        # Criar tabs principais
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visão Geral", 
            "🔮 Predições", 
            "📈 Analytics", 
            "💡 Insights"
        ])
        
        with tab1:
            self.create_overview_tab(filtered_df)
        
        with tab2:
            self.create_prediction_tab(filtered_df)
        
        with tab3:
            self.create_analytics_tab(filtered_df)
        
        with tab4:
            self.create_insights_tab(filtered_df)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div class="footer">
            <p>🔥 <strong>AttritionGuard</strong> - Employee Analytics Dashboard</p>
            <p>Desenvolvido com ❤️ usando Streamlit | 
            📊 Dados: IBM HR Analytics Dataset | 
            🤖 ML: Random Forest Classifier</p>
            <p><em>Para sugestões e melhorias, entre em contato!</em></p>
        </div>
        """, unsafe_allow_html=True)

# ========================================
# EXECUÇÃO PRINCIPAL
# ========================================

if __name__ == "__main__":
    # Inicializar e executar dashboard
    try:
        dashboard = AttritionDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Erro crítico na aplicação: {str(e)}")
        st.info("🔄 Recarregue a página ou verifique os dados de entrada.")

