# Tech Challenge 4 - Previsão de Preços de Ações da Coca-Cola Company (KO)

**Aluno:** Mateus Bressan

**Video Explicativo:** https://www.youtube.com/watch?v=i3u7iGFIHCY

## Descrição do Projeto e Objetivo Principal

Este projeto consiste em um sistema de **previsão do valor de fechamento de ações** utilizando técnicas de **Deep Learning**. O objetivo principal é desenvolver uma **pipeline completa de Machine Learning**, desde a coleta e pré-processamento de dados até a criação de um modelo preditivo (LSTM) e seu deploy em uma **API RESTful**.

O foco está na previsão do valor de fechamento das ações da **Coca-Cola Company (KO)**.

---

## Tecnologias Utilizadas

- **Linguagem:** Python
- **Framework Web:** Flask
- **Machine Learning/Deep Learning:** Keras (com TensorFlow como backend)
- **Coleta de Dados Financeiros:** yfinance
- **Serialização de Modelos:** joblib
- **Manipulação de Dados:** Pandas, NumPy
- **Visualização de Dados:** Matplotlib
- **Pré-processamento de Dados:** Scikit-learn (MinMaxScaler, train_test_split)
- **Contêinerização:** Docker (com Gunicorn para servir a aplicação)
- **Monitoramento de Recursos:** psutil
- **Logging:** Módulo logging padrão do Python
- **Requisições HTTP (internas):** Requests
- **Variáveis de Ambiente:** python-dotenv

---

## Como Configurar e Rodar o Projeto Localmente

### 1. Pré-requisitos

- Python 3.9+
- Docker
- Git

### 2. Clonar o Repositório

```bash
git clone https://github.com/mbressan/techchallenge4.git
cd techchallenge4
```

### 3. Configuração do Ambiente

#### Instalar Dependências

```bash
pip install -r requirements.txt
```

Principais dependências:
- Flask
- tensorflow
- keras
- yfinance
- gunicorn
- psutil
- requests
- python-dotenv
- pandas
- numpy
- matplotlib
- scikit-learn
- joblib


### 4. Como Iniciar a API (com Docker)

#### Construir a Imagem Docker

```bash
docker build -t previsao-acoes-ko .
```

#### Executar o Contêiner Docker

```bash
docker run -p 5000:5000 previsao-acoes-ko
```

### 5. Como Treinar o Modelo

Com a API rodando (via Docker), acesse o endpoint de treinamento em seu navegador:

```
http://localhost:5000/treinarmodelo
```

Aguarde a mensagem **"Dados coletados e modelo treinado!"**. O modelo e o scaler serão salvos na pasta `Models/`.

### 6. Como Testar a Previsão via API (Postman/cURL)

- **URL:** `http://localhost:5000/prever`
- **Método:** POST
- **Headers:** `Content-Type: application/json`
- **Body (raw, JSON):**

```json
{
    "historical_prices": [
        71.49, 71.15, 71.78, 71.77, 70.04, 69.68, 69.85, 70.29, 70.22, 70.69,
        70.07, 70.29, 70.13, 69.78, 69.39, 69.61, 69.31, 69.42, 69.50, 69.57,
        68.43, 68.55, 68.13, 67.45, 68.12, 67.82, 68.04, 67.83, 68.12, 67.81
    ]
}
```

- **Resposta Esperada:**

```json
{
    "previsao": 70.45
}
```

> O valor retornado estará formatado com duas casas decimais.

---

## Monitoramento

O projeto inclui **monitoramento básico de performance** para produção.

- **Métricas Coletadas:** Tempo de resposta da API, utilização de CPU e memória por processo.
- **Local dos Logs:** Todas as métricas são registradas no arquivo `app_monitor.log` na raiz do projeto.
- **Visualização:**
  - Local: `tail -f app_monitor.log`
  - Docker: `docker logs -f [nome_do_container]`
  - Nuvem: Console da plataforma de deploy


---

## Informações Relevantes Adicionais

- **Modelo:** O modelo LSTM foi treinado para prever o preço de fechamento com base nos últimos 30 dias de dados de fechamento.
- **Retreinamento:** Para manter o modelo atualizado, recomenda-se retreiná-lo periodicamente (ex: semanalmente ou mensalmente) através do endpoint `/treinarmodelo`.
- **Dados Históricos:** A coleta de dados usa yfinance para garantir que o modelo esteja sempre se baseando em informações de mercado atualizadas.

---



