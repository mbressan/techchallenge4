# Usa uma imagem base oficial do Python
# Recomendado usar uma versão específica para garantir consistência
FROM python:3.11.13-slim-bullseye

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de requisitos para o diretório de trabalho
# Isso é feito primeiro para aproveitar o cache do Docker,
# caso as dependências não mudem com frequência.
COPY requirements.txt .

# Instala as dependências Python
# O --no-cache-dir é para evitar o armazenamento de cache e reduzir o tamanho da imagem
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação para o diretório de trabalho
COPY . .

# Exponha a porta em que o Flask rodará (padrão é 5000)
EXPOSE 5000

# Define a variável de ambiente para que o Flask rode em modo de produção
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Comando para iniciar a aplicação Flask quando o contêiner for executado
# Usando gunicorn para um servidor de produção mais robusto
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]