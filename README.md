# ✈️ Previsão de Atrasos em Voos com Machine Learning

Este projeto tem como objetivo construir um modelo de regressão capaz de prever o tempo de atraso de voos com base em diversas variáveis relacionadas ao voo, companhia aérea, tipo de aeronave, datas e feriados. O modelo final é salvo e pode ser utilizado para inferência em novos dados.

## 📁 Dataset

O dataset utilizado foi obtido do seguinte repositório público no GitHub:

[flights.csv](https://raw.githubusercontent.com/TheGabrielVieira/ia-prevendo-atrasos-de-voos/refs/heads/main/data/flights.csv)

---

## 📌 Etapas do Projeto

### 1. 📊 Análise Exploratória dos Dados (EDA)
- Leitura da base de dados com `pandas`
- Estatísticas descritivas e avaliação do tipo das variáveis
- Gráficos de:
  - Atraso médio por companhia aérea
  - Número de voos por companhia aérea
  - Atrasos por tipo de voo (Schengen ou não)
  - Dias de feriado
  - Tipo de aeronave
  - Distribuições e boxplots de horários de chegada/partida e atraso

### 2. 🛠️ Engenharia de Atributos (Feature Engineering)
- Criação de nova coluna `date` a partir de `year` e `day`
- Criação das colunas:
  - `is_weekend` (indicando fim de semana)
  - `day_name` (nome do dia da semana)

### 3. 🔢 Codificação de Variáveis (Feature Encoding)
- Transformação de variáveis booleanas para binárias
- One-hot encoding para variáveis categóricas: `airline`, `aircraft_type`, `origin`, `day_name`

### 4. 🧹 Limpeza dos Dados
- Remoção de colunas não informativas ou redundantes
- Preparação do `df_clean` para modelagem

### 5. 🤖 Modelagem Preditiva
#### Modelos:
- `DummyRegressor`: baseline para comparação
- `RandomForestRegressor`: modelo principal

#### Avaliação:
- Métricas de avaliação: RMSE, MAE, R²
- Visualizações de erro de predição e resíduos com `Yellowbrick`

### 6. 📈 Validação Cruzada
- Aplicação de validação cruzada (5-fold)
- Avaliação das métricas médias e desvio padrão

### 7. 🔍 Seleção de Atributos
- Avaliação da importância dos atributos via `RandomForest`
- Testes com diferentes quantidades de variáveis preditoras

### 8. ⚙️ Otimização de Hiperparâmetros
- Utilização de `GridSearchCV` para encontrar os melhores parâmetros
- Utilização de `KFold` como validador cruzado

### 9. 💾 Salvamento e Teste do Modelo
- Salvamento do modelo final como `modelo_producao.pkl` usando `pickle`
- Teste do modelo com uma nova amostra de dados

---

## 🧪 Tecnologias e Bibliotecas Utilizadas

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yellowbrick
- pickle

---

## ▶️ Como Executar o Projeto na Sua Máquina

### 1. Clone o repositório

```bash
git clone https://github.com/TheGabrielVieira/ia-prevendo-atrasos-de-voos.git
cd ia-prevendo-atrasos-de-voos
```

### 2. Crie e ative um ambiente virtual (opcional, mas recomendado)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o notebook Jupyter

```bash
jupyter notebook
```

Abra o notebook `.ipynb` no navegador para seguir o passo a passo.

---

## 📈 Exemplo de uso do modelo salvo

```python
import pickle

# Carregar o modelo treinado
with open('modelo_producao.pkl', 'rb') as file:
    model = pickle.load(file)

# Nova amostra com 13 features selecionadas
nova_amostra = [0.0, 10.8941, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
predicao = model.predict([nova_amostra])[0]
print(f"Previsão de atraso (min): {predicao:.2f}")
```

---

## 📄 Licença

Este projeto é livre para fins acadêmicos e de aprendizado.

---

## 👤 Autor

**Gabriel Vieira**  
GitHub: [@TheGabrielVieira](https://github.com/TheGabrielVieira)
