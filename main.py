# Importações
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model # Adicionado load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, jsonify, render_template, request # Adicionado 'request'
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import base64
from io import BytesIO
import pickle

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Constante para time_steps ---
# Definir o número de passos de tempo que o modelo espera.
# Isso deve ser o mesmo valor usado durante o treinamento do modelo.
TIME_STEPS = 30 #

# Classe para Coleta de Dados
class ColetorDeDados:
    """Classe para coleta de dados históricos de ações."""

    def __init__(self, ticker, data_inicio, data_fim):
        self.ticker = ticker
        self.data_inicio = data_inicio
        self.data_fim = data_fim

    def coletar_dados_historicos(self):
        """Coleta dados históricos de fechamento de ações usando a biblioteca yfinance."""
        dados = yf.download(self.ticker, start=self.data_inicio, end=self.data_fim)
        dados = dados[['Close']]
        # Renomear a coluna 'Close' para 'fechamento' para consistência com o restante do código
        dados.columns = ['fechamento']
        # Resetar o índice para que a data seja uma coluna, se necessário para outras funções
        dados = dados.reset_index()
        dados = dados.rename(columns={'Date': 'data_pregao'})
        return dados

    # Manter esta função como estava, pois ela é chamada pelo dashboard
    def coletar_dados_tempo_real(self):
        dados = yf.download(self.ticker, start=self.data_inicio, end=self.data_fim)
        dados = dados[['Close']]
        # Renomear a coluna 'Close' para 'fechamento'
        dados.columns = ['fechamento']
        # Resetar o índice para que a data seja uma coluna
        dados = dados.reset_index()
        dados = dados.rename(columns={'Date': 'data_pregao'})
        return dados

# Função das janelas temporais
def cria_janelas_temporais(data, time_steps=TIME_STEPS): # Usando a constante TIME_STEPS
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, 0])
        y.append(data[i+time_steps, 0])
    return np.array(X), np.array(y)


def pre_processar_dados(dados):
    # Se os dados não tiverem a coluna 'fechamento', tentar usar 'Close'
    if 'fechamento' not in dados.columns and 'Close' in dados.columns:
        dados = dados.rename(columns={'Close': 'fechamento'})
    
    # Assegurar que 'fechamento' é um DataFrame para o scaler
    dados_fechamento = dados[['fechamento']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(dados_fechamento)

    X, Y = cria_janelas_temporais(df_scaled)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Divisão dos dados de treino e teste
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    return X_train, X_test, Y_train, Y_test, scaler

# Funções de Modelagem
def treinar_modelo(X_treino, y_treino):
    """Treina o modelo LSTM."""
    # Garante que X_train.shape[1] esteja disponível para input_shape
    # Esta função é chamada *após* pre_processar_dados, então X_train.shape[1] estará definido
    model = Sequential()
    model.add(LSTM( units=100,
                    activation='relu',
                    input_shape=(X_treino.shape[1], 1))) # Usando X_treino
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Treinamento
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(X_treino,
            y_treino,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0)

    return model


def avaliar_modelo(modelo, X_test, Y_test, scaler):
    """Avalia o modelo."""

    Y_pred = modelo.predict(X_test, verbose=0)
    # Certifique-se de que Y_test é um array 2D para inverse_transform
    Y_test_rescaled = scaler.inverse_transform(np.column_stack((np.zeros_like(Y_test), Y_test)))[:, 1]
    Y_pred_rescaled = scaler.inverse_transform(np.column_stack((np.zeros_like(Y_pred), Y_pred)))[:, 1]

    loss = modelo.evaluate(X_test, Y_test, verbose=0)
    mae = mean_absolute_error(Y_test_rescaled, Y_pred_rescaled)
    rmse = np.sqrt(root_mean_squared_error(Y_test_rescaled, Y_pred_rescaled))
    mape = mean_absolute_percentage_error(Y_test_rescaled, Y_pred_rescaled)*100

    resultados = {
        "loss": loss,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }

    print(f'Loss: {loss}')
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {(mape):.2f}%")

    return resultados

# --- Novas Funções para Gráficos e Previsão do Dashboard ---
# Estas funções foram adaptadas para o dashboard que busca a previsão de amanhã
# e exibe um gráfico que pode incluir essa previsão.

def ler_dados_historicos_para_dashboard(ticker="KO", days_to_fetch=365):
    """
    Lê os dados históricos necessários para o gráfico do dashboard.
    Por padrão, busca o último ano para KO (exemplo).
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_fetch)
    coletor = ColetorDeDados(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    dados = coletor.coletar_dados_historicos()
    return dados

def gerar_grafico_historico(ticker="GOOG", previsao=None, data_previsao=None):
    """Gera o gráfico de histórico de preços com a previsão, se disponível."""
    dados = ler_dados_historicos_para_dashboard(ticker=ticker)
    if dados is None or dados.empty:
        return None  # Retorna None em caso de erro ou dados vazios

    plt.figure(figsize=(10, 5))
    plt.plot(dados['data_pregao'], dados['fechamento'], label='Preço de Fechamento', color='blue')

    # Adicionar o ponto da previsão, se disponível
    if previsao and data_previsao:
        plt.scatter(data_previsao, previsao, color='red', label=f'Previsão para {data_previsao.strftime("%Y-%m-%d")}', zorder=5)
        plt.annotate(f'${previsao:.2f}', (data_previsao, previsao), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.title(f'Histórico de Preços de {ticker.upper()} com Previsão')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Ajusta o layout para evitar sobreposição

    # Salvar o gráfico em um buffer na memória
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return plot_url

# API Flask

app = Flask(__name__)

# --- Rota /prever_dashboard para a lógica interna do dashboard ---
# Esta rota busca os 30 dias mais recentes da KO e faz a previsão para amanhã.
# É uma função auxiliar para o dashboard, não a principal /prever da API.
@app.route('/prever_dashboard', methods=['GET'])
def prever_para_dashboard():
    """
    Realiza a previsão para o dashboard, buscando os últimos 30 dias da KO.
    """
    ticker = "KO"
    # Buscar os últimos TIME_STEPS dias de dados para a previsão
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TIME_STEPS + 5) # Pegar um pouco mais de dados para garantir TIME_STEPS dias válidos após dropna
    coletor = ColetorDeDados(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    dados_recentes = coletor.coletar_dados_historicos()

    if dados_recentes.empty or len(dados_recentes) < TIME_STEPS:
        return jsonify({"mensagem": f"Não há dados suficientes ({len(dados_recentes)} dias) para gerar a previsão. Mínimo {TIME_STEPS} dias."}), 400

    # Selecionar apenas a coluna 'fechamento' e pegar os últimos TIME_STEPS valores
    # A função cria_janelas_temporais espera um array 2D
    last_n_days_data = dados_recentes['fechamento'].values[-TIME_STEPS:].reshape(-1, 1)

    try:
        # Carregar modelo e scaler
        # Certifique-se de que o caminho para os arquivos .keras e .pkl está correto
        modelo = load_model('Models/modelKO.keras')
        scaler = joblib.load('Models/scalerKO.pkl')

        # Pré-processar os dados
        # O scaler foi fitado em uma única coluna 'fechamento', então precisamos de um array 2D
        input_scaled = scaler.transform(last_n_days_data)

        # A entrada para predict deve ser (1, TIME_STEPS, 1)
        input_for_prediction = input_scaled.reshape(1, TIME_STEPS, 1)

        # Realizar a previsão
        previsao_escalada = modelo.predict(input_for_prediction)

        # Inverter a escala da previsão para a escala original
        # A inverse_transform espera um array 2D com a mesma forma do input original
        # Como o scaler foi fitado apenas na coluna 'fechamento', precisamos 'simular' as outras colunas para inverse_transform
        # Criamos um array de zeros e colocamos a previsão na coluna correta (a que foi usada no fit do scaler)
        # Assumimos que o scaler foi fitado apenas na coluna de fechamento (índice 0)
        dummy_input = np.zeros((1, scaler.n_features_in_))
        dummy_input[0, 0] = previsao_escalada[0, 0] # Coloca a previsão na primeira feature (fechamento)
        previsao_original = scaler.inverse_transform(dummy_input)[0, 0]

        data_previsao = datetime.now() + timedelta(days=1)

        return jsonify({
            "previsao": float(previsao_original),
            "data_previsao": data_previsao.strftime("%Y-%m-%d")
        }), 200

    except Exception as e:
        return jsonify({"mensagem": f"Erro ao prever: {str(e)}"}), 500

# --- Rota /prever da API RESTful (recebe dados do usuário) ---
@app.route('/prever', methods=['POST'])
def prever_acao():
    """
    Realiza a previsão do valor da ação com base nos dados históricos de preços fornecidos pelo usuário.
    Espera um JSON com a chave 'historical_prices' contendo uma lista de floats.
    Exemplo de corpo da requisição:
    {
        "historical_prices": [100.5, 101.2, 102.0, ..., 105.3] (últimos TIME_STEPS dias)
    }
    """
    data = request.get_json()

    if not data or 'historical_prices' not in data:
        return jsonify({"mensagem": "Dados inválidos. Espera-se um JSON com 'historical_prices'."}), 400

    historical_prices = data['historical_prices']

    # Validar a quantidade de dados fornecidos
    if len(historical_prices) < TIME_STEPS:
        return jsonify({"mensagem": f"Dados insuficientes. É necessário fornecer pelo menos {TIME_STEPS} preços históricos."}), 400

    # Usar apenas os últimos TIME_STEPS preços
    # O modelo foi treinado com uma sequência de 30 dias de preços de fechamento
    input_data = np.array(historical_prices[-TIME_STEPS:]).reshape(-1, 1)

    try:
        # Carregar modelo e scaler
        modelo = load_model('Models/modelKO.keras')
        scaler = joblib.load('Models/scalerKO.pkl')

        # Pré-processar os dados de entrada usando o mesmo scaler
        input_scaled = scaler.transform(input_data)

        # Reformatar os dados para a entrada do modelo LSTM (amostras, time_steps, features)
        input_for_prediction = input_scaled.reshape(1, TIME_STEPS, 1)

        # Realizar a previsão
        previsao_escalada = modelo.predict(input_for_prediction)

        # Inverter a escala da previsão para o valor original
        # O scaler foi fitado apenas na coluna 'fechamento', então precisamos 'simular' a mesma forma para inverse_transform
        dummy_input = np.zeros((1, scaler.n_features_in_))
        dummy_input[0, 0] = previsao_escalada[0, 0] # Coloca a previsão na primeira feature (fechamento)
        previsao_original = scaler.inverse_transform(dummy_input)[0, 0]


        return jsonify({"previsao": float(f"{previsao_original:.2f}")}), 200

    except FileNotFoundError:
        return jsonify({"mensagem": "Modelo ou scaler não encontrados. Por favor, treine o modelo primeiro."}), 500
    except Exception as e:
        return jsonify({"mensagem": f"Erro ao prever: {str(e)}"}), 500

@app.route('/treinarmodelo', methods=['GET'])
def coletar_dados_treinar():
    print("Iniciando o treinamento do modelo...")

    # Coletar e processar dados históricos até o dia anterior
    ticker = "KO"
    data_inicio = "2010-01-01"
    data_fim = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Coletando dados de {ticker} de {data_inicio} a {data_fim}...")

    coletor = ColetorDeDados(ticker, data_inicio, data_fim)
    dados = coletor.coletar_dados_historicos()

    if dados.empty:
        return jsonify({"mensagem": "Não foram encontrados dados históricos para treinamento."}), 400

    # Explorar e pré-processar dados
    # A função pre_processar_dados já lida com a coluna 'Close' ou 'fechamento'
    X_treino, X_teste, y_treino, y_teste, scaler = pre_processar_dados(dados)

    # Treinar e avaliar o modelo
    modelo = treinar_modelo(X_treino, y_treino)
    resultados_avaliacao = avaliar_modelo(modelo, X_teste, y_teste, scaler)

    os.makedirs("Models", exist_ok=True)
    modelo.save(os.path.join("Models", "modelKO.keras")) # Usar os.path.join para compatibilidade de SO
    joblib.dump(scaler, os.path.join("Models", "scalerKO.pkl")) # Usar os.path.join

    return jsonify({"mensagem": "Dados coletados e modelo treinado!", "resultados_avaliacao": resultados_avaliacao}), 200


@app.route("/")
def dashboard():
    """Rota principal do dashboard."""
    # Chamando a nova rota interna para a previsão do dashboard
    previsao_data = requests.get(request.url_root + 'prever_dashboard').json()

    mensagem_previsao = "Não foi possível obter a previsão."
    previsao_valor = None
    data_previsao_obj = None

    if previsao_data and "previsao" in previsao_data:
        previsao_valor = previsao_data['previsao']
        data_previsao_str = previsao_data.get('data_previsao', '')
        if data_previsao_str:
            data_previsao_obj = datetime.strptime(data_previsao_str, "%Y-%m-%d")
            mensagem_previsao = f"Previsão para {data_previsao_str}: ${previsao_valor:.2f}"
        else:
            mensagem_previsao = f"Previsão: ${previsao_valor:.2f}"
    elif previsao_data and "mensagem" in previsao_data:
        mensagem_previsao = previsao_data['mensagem']

    grafico_historico = gerar_grafico_historico(
        ticker="KO", # Alterado para KO para consistência
        previsao=previsao_valor,
        data_previsao=data_previsao_obj
    )

    return render_template(
        "dashboard.html",
        grafico_historico=grafico_historico,
        mensagem_previsao=mensagem_previsao
    )

# Exemplo de Uso (para teste)
if __name__ == '__main__':
    # É uma boa prática não ter a chamada da rota de treinamento aqui no if __name__
    # pois ela seria executada toda vez que a API fosse iniciada.
    # O treinamento deve ser um processo separado, disparado manualmente ou via endpoint.
    # Ex: Para treinar: curl http://localhost:5000/treinarmodelo

    # Iniciar a API
    # Importar requests para usar na rota dashboard
    import requests
    app.run(debug=False, use_reloader=False)