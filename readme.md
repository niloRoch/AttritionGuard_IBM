# 💥 Employee Attrition Analytics
## Análise Preditiva de Rotatividade de Funcionários

<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=FF6B6B&background=00000000&center=true&vCenter=true&width=700&lines=Employee+Attrition+Analytics;Análise+Preditiva+%7C+Machine+Learning+%7C+HR+Tech;Transforming+Data+into+Insights)](https://git.io/typing-svg)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**🔗 [🚀 Acesse o Dashboard Online](https://attritionguardibm.streamlit.app/)** 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://attritionguardibm.streamlit.app/)

</div>

---

## 🎯 **Objetivo do Projeto**

Este projeto desenvolve uma **solução completa de analytics** para predição e análise de rotatividade de funcionários (employee attrition), utilizando o dataset da IBM HR Analytics. O objetivo é fornecer **insights acionáveis** para RH e gestores, permitindo ações proativas de retenção de talentos através de:

- 🔮 **Predição de Risco Individual** com score de probabilidade
- 📊 **Dashboard Interativo** com visualizações em tempo real  
- 💡 **Insights Estratégicos** baseados em dados
- 💰 **Calculadora de ROI** para justificar investimentos

---

## 🏆 **Principais Resultados**

<div align="center">

| Métrica | Valor | Benchmark | Status |
|---------|-------|-----------|--------|
| **AUC-ROC** | 0.87 | > 0.85 | ✅ |
| **Precisão** | 0.82 | > 0.80 | ✅ |
| **Recall** | 0.79 | > 0.75 | ✅ |
| **F1-Score** | 0.80 | > 0.78 | ✅ |

### **💰 Impacto de Negócio**
- **16%** de taxa de attrition identificada
- **$450K** economia potencial anual
- **ROI de 350%** em programas de retenção
- **85%** dos funcionários de alto risco identificados corretamente

</div>

---

## 📊 **Dashboard Interativo**

<div align="center">

### **Principais Funcionalidades:**
- 📈 **Visão Geral**: KPIs, distribuições e correlações
- 🔮 **Predições**: Score de risco individual e lista de alto risco
- 📊 **Analytics**: Segmentação, análise temporal e correlações
- 💡 **Insights**: Recomendações acionáveis e calculadora de ROI

[![Demo](https://img.shields.io/badge/🚀_Demo_Online-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://attritionguardibm.streamlit.app/)

*Dashboard disponível 24/7 na Streamlit Cloud*

</div>

---

## 🚀 **Stack Tecnológica**

<div align="center">

### **📊 Análise de Dados**
![Python](https://img.shields.io/badge/Python-000000?style=for-the-badge&logo=python&logoColor=blue)
![Pandas](https://img.shields.io/badge/Pandas-000000?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-000000?style=for-the-badge&logo=numpy&logoColor=blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-000000?style=for-the-badge&logo=python&logoColor=blue)

### **🤖 Machine Learning**
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-000000?style=for-the-badge&logo=scikit-learn&logoColor=orange)
![XGBoost](https://img.shields.io/badge/XGBoost-000000?style=for-the-badge&logo=xgboost&logoColor=red)
![SHAP](https://img.shields.io/badge/SHAP-000000?style=for-the-badge&logo=python&logoColor=white)

### **🖥️ Interface e Deploy**
![Streamlit](https://img.shields.io/badge/Streamlit-000000?style=for-the-badge&logo=streamlit&logoColor=red)
![Plotly](https://img.shields.io/badge/Plotly-000000?style=for-the-badge&logo=plotly&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-000000?style=for-the-badge&logo=html5&logoColor=orange)
![CSS3](https://img.shields.io/badge/CSS3-000000?style=for-the-badge&logo=css3&logoColor=blue)

</div>

---

## 📁 **Estrutura do Projeto**

<details>
<summary>🌟 <strong>Arquitetura Completa</strong> (clique para expandir)</summary>

```
employee-attrition-analytics/
├── 📊 data/
│   ├── raw/                          # Dados originais
│   ├── processed/                    # Dados processados
│   └── external/                     # Dados externos
├── 📓 notebooks/
│   ├── 01_exploratory_analysis.ipynb # Análise exploratória
│   ├── 02_data_preprocessing.ipynb   # Pré-processamento
│   ├── 03_feature_engineering.ipynb # Engenharia de features
│   ├── 04_modeling.ipynb            # Modelagem ML
│   └── 05_model_interpretation.ipynb # Interpretabilidade
├── 🐍 src/
│   ├── data/
│   │   ├── preprocessing.py          # Limpeza e transformação
│   │   └── feature_engineering.py   # Criação de features
│   ├── models/
│   │   ├── train_model.py           # Treinamento de modelos
│   │   ├── predict_model.py         # Predições
│   │   └── evaluate_model.py        # Avaliação
│   └── visualization/
│       └── visualize.py             # Funções de visualização
├── 🎛️ dashboard/
│   ├── streamlit_app.py             # Dashboard principal
│   ├── components/                   # Componentes do dashboard
│   └── assets/                      # Arquivos estáticos
├── 📋 reports/
│   ├── executive_summary.pdf        # Resumo executivo
│   ├── technical_report.pdf         # Relatório técnico
│   └── presentation.pptx            # Apresentação
├── 🧪 tests/
│   ├── test_preprocessing.py        # Testes unitários
│   └── test_models.py              # Testes de modelos
├── 📦 models/
│   ├── best_model.pkl              # Modelo final treinado
│   └── preprocessor.pkl            # Pré-processador
├── ⚙️ config/
│   └── config.yaml                 # Configurações
├── 📄 requirements.txt             # Dependências Python
├── 🐳 Dockerfile                  # Container Docker
├── 🚀 docker-compose.yml          # Orquestração
└── 📖 README.md                   # Este arquivo
```

</details>

---

## 🔧 **Instalação e Configuração**

### **Pré-requisitos**
- Python 3.8 ou superior
- Git
- (Opcional) Docker para containerização

### **1️⃣ Clone o Repositório**
```bash
git clone https://github.com/seu-usuario/employee-attrition-analytics.git
cd employee-attrition-analytics
```

### **2️⃣ Ambiente Virtual**
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate

# Ativar (Linux/Mac)
source venv/bin/activate
```

### **3️⃣ Instalar Dependências**
```bash
pip install -r requirements.txt
```

### **4️⃣ Configurar Dados**
```bash
# Baixar dataset (se não incluído)
mkdir data/raw
# Colocar IBM_Fn-UseC_-HR-Employee-Attrition.csv em data/raw/
```

---

## 🎮 **Como Usar**

### **📊 Dashboard Online (Recomendado)**
```bash
# Acesse diretamente no navegador:
https://attritionguardibm.streamlit.app/
```

### **🎛️ Dashboard Local**
```bash
# Executar dashboard localmente
streamlit run dashboard/streamlit_app.py
```

### **🔮 Predições Individuais**
```python
from src.models.predict_model import AttritionPredictor

# Carregar modelo treinado
predictor = AttritionPredictor()
predictor.load_model('models/best_model.pkl')

# Fazer predição
employee_data = {
    'Age': 30,
    'MonthlyIncome': 5000,
    'OverTime': 'Yes',
    'JobSatisfaction': 2
}

risk_score = predictor.predict_risk(employee_data)
print(f"Score de risco: {risk_score:.2%}")
```

---

## 📊 **Principais Insights Descobertos**

<details>
<summary>🚨 <strong>Fatores de Alto Risco</strong> (clique para expandir)</summary>

### **Top 5 Fatores de Risco**
1. **Overtime** → 31% de attrition vs 10% sem overtime
2. **Baixa Satisfação** → 25% de attrition (satisfação ≤ 2)
3. **Funcionários Jovens** → 22% de attrition (<30 anos)
4. **Distância de Casa** → 18% de attrition (>20km)
5. **Pouca Experiência** → 35% de attrition (<2 anos na empresa)

</details>

<details>
<summary>✅ <strong>Fatores Protetivos</strong> (clique para expandir)</summary>

### **Top 5 Fatores de Retenção**
1. **Alta Satisfação** → 3% de attrition (satisfação = 4)
2. **Sem Overtime** → 10% de attrition
3. **Longa Permanência** → 5% de attrition (>10 anos)
4. **Funcionários Sêniores** → 7% de attrition (>45 anos)
5. **Altos Salários** → 8% de attrition (top quartil)

</details>

<details>
<summary>💡 <strong>Padrões Identificados</strong> (clique para expandir)</summary>

- 📈 **Curva de Attrition**: Pico nos primeiros 2 anos, depois estabiliza
- 🏢 **Por Departamento**: Sales (20%) > R&D (14%) > HR (12%)
- 💰 **Impacto Salarial**: Cada $1000 de aumento reduz risco em 2%
- ⚖️ **Work-Life Balance**: 40% menos risco com bom equilíbrio

</details>

---

## 🎯 **Features do Projeto**

### **📊 Análise de Dados**
- ✅ EDA completa com visualizações interativas
- ✅ Identificação de 15+ fatores de risco
- ✅ Segmentação de funcionários por risco
- ✅ Análise temporal de permanência

### **🤖 Machine Learning**
- ✅ Comparação de 7 algoritmos diferentes
- ✅ Otimização automática de hiperparâmetros
- ✅ Validação cruzada robusta
- ✅ Interpretabilidade com SHAP

### **🎛️ Dashboard Interativo**
- ✅ 4 tabs principais com 15+ visualizações
- ✅ Filtros dinâmicos por departamento, idade, salário
- ✅ Preditor individual de risco
- ✅ Lista de funcionários de alto risco
- ✅ Calculadora de ROI
- ✅ Download de relatórios

### **💼 Business Intelligence**
- ✅ KPIs de RH em tempo real
- ✅ Insights acionáveis por segmento
- ✅ Recomendações estratégicas
- ✅ Análise de impacto financeiro

--- 

## 🤝 **Contribuindo**

Contribuições são muito bem-vindas! Por favor, leia o [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes sobre nosso código de conduta e processo de envio de pull requests.

<details>
<summary>📝 <strong>Como Contribuir</strong> (clique para expandir)</summary>

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### **🛠 Reportar Bugs**
Use as [GitHub Issues](https://github.com/seu-usuario/employee-attrition-analytics/issues) para reportar bugs ou solicitar features.

</details>

---

<div align="center">

## 📞 **Contato**

[![Website](https://img.shields.io/badge/Website-4c1d95?style=for-the-badge&logo=firefox&logoColor=a855f7)](https://www.nilorocha.tech)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nilo-rocha-/)
[![Email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:nilo.roch4@gmail.com)

---

## 📈 **Analytics do Projeto**

![GitHub stars](https://img.shields.io/github/stars/seu-usuario/employee-attrition-analytics?style=social)
![GitHub forks](https://img.shields.io/github/forks/seu-usuario/employee-attrition-analytics?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/seu-usuario/employee-attrition-analytics?style=social)

---

#
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">


![Footer](https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=100&section=footer&text=Thanks%20for%20exploring%20the%20insights&fontSize=16&fontColor=ffffff&animation=twinkling)

</div>

---

## 📄 **Changelog**

### **v1.0.0** (2024-06)
- 🎉 Release inicial
- ✅ Pipeline completo de ML
- ✅ Dashboard interativo
- ✅ Documentação completa

### **v0.9.0** (2024-04)
- 🧪 Versão beta
- ✅ Modelos básicos funcionais
- ✅ EDA completa
- ⏳ Dashboard em desenvolvimento

### **v0.5.0** (2024-03)
- 🚀 MVP funcional
- ✅ Pré-processamento
- ✅ Primeira versão dos modelos
- ⏳ Visualizações básicas
