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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURAÇÃO DA PÁGINA
# ========================================

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="💥",
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
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #FF6B6B;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .insight-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4ECDC4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .success-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
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
        self.model = None
        self.feature_importance = None
        self.scalers = {}
        self.label_encoders = {}
        if self.load_data():
            st.success("✅ Dashboard inicializado com sucesso!")
        else:
            st.error("❌ Erro na inicialização do dashboard")
    
    @st.cache_data
    def load_data(_self):
        """Carrega e prepara os dados"""
        try:
            # Primeiro tentar carregar dados reais de diferentes locais possíveis
            possible_paths = [
                'IBM_Fn-UseC_-HR-Employee-Attrition.csv',
                'data/raw/IBM_Fn-UseC_-HR-Employee-Attrition.csv',
                'data/IBM_Fn-UseC_-HR-Employee-Attrition.csv',
                'WA_Fn-UseC_-HR-Employee-Attrition.csv',
                'data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv'
            ]
            
            df = None
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    st.success(f"✅ Dados carregados de: {path}")
                    break
                except:
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
            
            return True
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {str(e)}")
            return False
    
    def generate_synthetic_data(self):
        """Gera dados sintéticos realísticos para demonstração"""
        np.random.seed(42)
        n_samples = 1470  # Mesmo tamanho do dataset original
        
        # Definir probabilidades condicionais realísticas
        data = {}
        
        # Variáveis básicas
        data['Age'] = np.random.randint(18, 65, n_samples)
        data['Gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
        data['MaritalStatus'] = np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.32, 0.46, 0.22])
        data['Education'] = np.random.randint(1, 6, n_samples)
        data['EducationField'] = np.random.choice([
            'Life Sciences', 'Other', 'Medical', 'Marketing', 
            'Technical Degree', 'Human Resources'
        ], n_samples, p=[0.41, 0.16, 0.15, 0.16, 0.08, 0.04])
        
        # Variáveis de trabalho
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
        
        # Variáveis financeiras (correlacionadas com nível e experiência)
        base_salary = 2000 + data['JobLevel'] * 2000 + data['TotalWorkingYears'] * 100
        data['MonthlyIncome'] = base_salary + np.random.normal(0, 1000, n_samples)
        data['MonthlyIncome'] = np.maximum(data['MonthlyIncome'], 1009)  # Salário mínimo
        
        data['HourlyRate'] = np.random.randint(30, 101, n_samples)
        data['DailyRate'] = np.random.randint(102, 1499, n_samples)
        data['MonthlyRate'] = np.random.randint(2094, 26999, n_samples)
        data['PercentSalaryHike'] = np.random.randint(11, 25, n_samples)
        data['StockOptionLevel'] = np.random.randint(0, 4, n_samples)
        
        # Variáveis de satisfação e ambiente
        data['JobSatisfaction'] = np.random.randint(1, 5, n_samples)
        data['EnvironmentSatisfaction'] = np.random.randint(1, 5, n_samples)
        data['RelationshipSatisfaction'] = np.random.randint(1, 5, n_samples)
        data['WorkLifeBalance'] = np.random.randint(1, 5, n_samples)
        data['JobInvolvement'] = np.random.randint(1, 5, n_samples)
        data['PerformanceRating'] = np.random.choice([3, 4], n_samples, p=[0.85, 0.15])
        
        # Variáveis geográficas e logísticas
        data['DistanceFromHome'] = np.random.randint(1, 30, n_samples)
        data['BusinessTravel'] = np.random.choice([
            'Travel_Rarely', 'Travel_Frequently', 'Non-Travel'
        ], n_samples, p=[0.71, 0.19, 0.10])
        data['OverTime'] = np.random.choice(['No', 'Yes'], n_samples, p=[0.72, 0.28])
        
        # Treinamento
        data['TrainingTimesLastYear'] = np.random.randint(0, 7, n_samples)
        
        # Variáveis constantes ou quase constantes
        data['Over18'] = 'Y'
        data['EmployeeCount'] = 1
        data['StandardHours'] = 80
        data['EmployeeNumber'] = range(1, n_samples + 1)
        
        # Criar Attrition baseado em fatores realísticos
        attrition_prob = np.zeros(n_samples)
        
        # Fatores que aumentam probabilidade de attrition
        attrition_prob += (data['Age'] < 30) * 0.15  # Jovens têm mais tendência
        attrition_prob += (data['JobSatisfaction'] <= 2) * 0.20  # Baixa satisfação
        attrition_prob += (data['EnvironmentSatisfaction'] <= 2) * 0.15
        attrition_prob += (data['WorkLifeBalance'] <= 2) * 0.15
        attrition_prob += (data['YearsAtCompany'] <= 2) * 0.25  # Novatos
        attrition_prob += (data['OverTime'] == 'Yes') * 0.20  # Overtime
        attrition_prob += (data['BusinessTravel'] == 'Travel_Frequently') * 0.10
        attrition_prob += (data['DistanceFromHome'] > 20) * 0.10
        attrition_prob += (data['MaritalStatus'] == 'Single') * 0.05
        attrition_prob += (data['NumCompaniesWorked'] >= 4) * 0.10
        
        # Fatores que diminuem probabilidade
        attrition_prob -= (data['JobLevel'] >= 4) * 0.10  # Níveis seniores
        attrition_prob -= (data['MonthlyIncome'] > np.percentile(data['MonthlyIncome'], 75)) * 0.15
        attrition_prob -= (data['YearsAtCompany'] > 10) * 0.20
        attrition_prob -= (data['StockOptionLevel'] >= 1) * 0.05
        
        # Normalizar probabilidades
        attrition_prob = np.clip(attrition_prob, 0.05, 0.50)
        
        # Gerar attrition baseado nas probabilidades
        data['Attrition'] = np.array(['Yes' if np.random.random() < prob else 'No' 
                                    for prob in attrition_prob])
        
        return pd.DataFrame(data)
    
    def prepare_data(self, df):
        """Prepara os dados para análise"""
        # Criar variável binária para Attrition
        df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)
        
        # Criar score de risco baseado em múltiplos fatores
        risk_score = np.zeros(len(df))
        
        # Fatores de risco com pesos baseados em literatura
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
        
        # Adicionar ruído aleatório pequeno
        risk_score += np.random.normal(0, 0.03, len(df))
        
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
            df['Income_Age_Ratio'] = df['MonthlyIncome'] / df['Age']
        
        if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
            df['Company_Experience_Ratio'] = df['YearsAtCompany'] / np.maximum(df['TotalWorkingYears'], 1)
        
        return df
    
    def train_models(self):
        """Treina múltiplos modelos para predição de attrition"""
        if self.df is None:
            return
        
        try:
            # Selecionar features para modelagem
            numeric_features = [
                'Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction',
                'DistanceFromHome', 'EnvironmentSatisfaction', 'WorkLifeBalance',
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'TotalWorkingYears',
                'TrainingTimesLastYear', 'JobInvolvement', 'JobLevel'
            ]
            
            categorical_features = [
                'OverTime', 'BusinessTravel', 'Department', 'JobRole', 
                'MaritalStatus', 'Gender'
            ]
            
            # Filtrar features que existem no dataset
            available_numeric = [f for f in numeric_features if f in self.df.columns]
            available_categorical = [f for f in categorical_features if f in self.df.columns]
            
            # Preparar features numéricas
            X_numeric = self.df[available_numeric].fillna(0)
            
            # Preparar features categóricas
            X_categorical = pd.DataFrame()
            for col in available_categorical:
                if col in self.df.columns:
                    # One-hot encoding para variáveis categóricas
                    dummies = pd.get_dummies(self.df[col], prefix=col)
                    X_categorical = pd.concat([X_categorical, dummies], axis=1)
            
            # Combinar features
            X = pd.concat([X_numeric, X_categorical], axis=1)
            y = self.df['Attrition_Binary']
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Treinar Random Forest (modelo principal)
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.model.fit(X_train, y_train)
            
            # Feature importance
            feature_names = X.columns
            importances = self.model.feature_importances_
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(15)
            
            # Métricas do modelo
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Salvar métricas para uso posterior
            self.model_metrics = {
                'accuracy': (y_pred == y_test).mean(),
                'precision': ((y_pred == 1) & (y_test == 1)).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0,
                'recall': ((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0
            }
            
            # Calcular AUC-ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            self.model_metrics['auc_roc'] = auc(fpr, tpr)
            
            # Salvar dados de teste para análises
            self.X_test = X_test
            self.y_test = y_test
            self.y_pred_proba = y_pred_proba
            
        except Exception as e:
            st.warning(f"⚠️ Erro ao treinar modelo: {str(e)}")
            self.model = None
    
    def create_main_interface(self):
        """Cria a interface principal do dashboard"""
        
        # Cabeçalho principal
        st.markdown('<div class="main-header">🔥 Employee Attrition Analytics</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Setup sidebar e obter dados filtrados
        filtered_df = self.setup_sidebar()
        
        # Verificar se há dados para exibir
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
        st.markdown(
            '<div class="footer">💥 Employee Attrition Analytics Dashboard | '
            'Desenvolvido com ❤️ usando Streamlit</div>',
            unsafe_allow_html=True
        )
    
    def setup_sidebar(self):
        """Configura sidebar com filtros"""
        if self.df is None:
            st.sidebar.error("❌ Dados não carregados")
            return None
        
        st.sidebar.header("🎛️ Filtros e Configurações")
        
        # Filtros básicos
        if 'Department' in self.df.columns:
            departments = ['Todos'] + sorted(self.df['Department'].unique().tolist())
            selected_dept = st.sidebar.selectbox("🏢 Departamento", departments)
        else:
            selected_dept = 'Todos'
        
        if 'Age' in self.df.columns:
            age_range = st.sidebar.slider(
                "👥 Faixa Etária", 
                int(self.df['Age'].min()), 
                int(self.df['Age'].max()), 
                (int(self.df['Age'].min()), int(self.df['Age'].max()))
            )
        else:
            age_range = (18, 65)
        
        if 'MonthlyIncome' in self.df.columns:
            salary_range = st.sidebar.slider(
                "💰 Faixa Salarial ($)", 
                int(self.df['MonthlyIncome'].min()), 
                int(self.df['MonthlyIncome'].max()), 
                (int(self.df['MonthlyIncome'].min()), int(self.df['MonthlyIncome'].max())),
                step=500
            )
        else:
            salary_range = (1000, 20000)
        
        # Filtros adicionais
        if 'JobRole' in self.df.columns:
            job_roles = ['Todos'] + sorted(self.df['JobRole'].unique().tolist())
            selected_role = st.sidebar.selectbox("👔 Função", job_roles)
        else:
            selected_role = 'Todos'
        
        # Aplicar filtros
        filtered_df = self.df.copy()
        
        if selected_dept != 'Todos' and 'Department' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
        
        if 'Age' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['Age'] >= age_range[0]) & 
                (filtered_df['Age'] <= age_range[1])
            ]
        
        if 'MonthlyIncome' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['MonthlyIncome'] >= salary_range[0]) & 
                (filtered_df['MonthlyIncome'] <= salary_range[1])
            ]
        
        if selected_role != 'Todos' and 'JobRole' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]
        
        # Métricas do filtro
        st.sidebar.markdown("---")
        st.sidebar.markdown("📊 **Dados Filtrados**")
        st.sidebar.metric("Total de Funcionários", f"{len(filtered_df):,}")
        
        if len(filtered_df) > 0:
            attrition_rate = filtered_df['Attrition_Binary'].mean()
            st.sidebar.metric("Taxa de Attrition", f"{attrition_rate:.1%}")
            
            if 'Risk_Score' in filtered_df.columns:
                high_risk = (filtered_df['Risk_Score'] > 0.7).sum()
                st.sidebar.metric("Funcionários Alto Risco", f"{high_risk:,}")
        
        return filtered_df
    
    def create_overview_tab(self, filtered_df):
        """Cria tab de visão geral"""
        
        # Métricas principais
        st.subheader("📊 Métricas Principais")
        col1, col2, col3, col4 = st.columns(4)
        
        total_employees = len(filtered_df)
        attrition_rate = filtered_df['Attrition_Binary'].mean()
        high_risk_employees = (filtered_df['Risk_Score'] > 0.7).sum() if 'Risk_Score' in filtered_df.columns else 0
        avg_tenure = filtered_df['YearsAtCompany'].mean() if 'YearsAtCompany' in filtered_df.columns else 0
        
        with col1:
            st.metric(
                "👥 Total de Funcionários", 
                f"{total_employees:,}"
            )
        
        with col2:
            st.metric(
                "📊 Taxa de Attrition", 
                f"{attrition_rate:.1%}"
            )
        
        with col3:
            st.metric(
                "🚨 Alto Risco", 
                f"{high_risk_employees:,}",
                delta=f"{high_risk_employees / max(total_employees, 1):.1%} do total"
            )
        
        with col4:
            st.metric(
                "⏱️ Permanência Média", 
                f"{avg_tenure:.1f} anos"
            )
        
        st.markdown("---")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de Attrition por Departamento
            if 'Department' in filtered_df.columns:
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
                st.info("Dados de departamento não disponíveis")
        
        with col2:
            # Distribuição do Risk Score
            if 'Risk_Score' in filtered_df.columns:
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
                st.info("Score de risco não calculado")
        
        # Análise de fatores de risco
        st.subheader("🎯 Principais Fatores de Risco")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'OverTime' in filtered_df.columns:
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
            if 'JobSatisfaction' in filtered_df.columns:
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
        
        with col
