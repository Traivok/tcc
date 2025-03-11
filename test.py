import datetime
import random

import pandas as pd
import streamlit as st

# Função para simular os treinos


def simular_treinos(intensidades, treinos_qtd, inicio, fim):
    treinos = []
    treinos_feitos = 0
    data_atual = inicio
    while data_atual <= fim:
        dia_da_semana = data_atual.weekday()
        intensidade = intensidades[dia_da_semana]
        qtd_max = treinos_qtd[dia_da_semana]
        qtd = 0

        for _ in range(qtd_max):
            rng = random.random()
            if rng < intensidade:
                qtd += 1
                treinos_feitos += 1

        record_training_session(treinos, treinos_feitos, data_atual, qtd)

        # Próximo dia
        data_atual += datetime.timedelta(days=1)

    # Retornar um DataFrame com os resultados
    return pd.DataFrame(treinos), treinos_feitos


def record_training_session(treinos, treinos_feitos, data_atual, qtd):
    if qtd > 0:  # Se houver treino, conta o treino
        treinos_feitos += qtd
        treinos.append({
            'data': data_atual,
            'dia_da_semana': data_atual.strftime('%A'),
            'treino_feito': True,
            'qtd': qtd
        })
    else:
        treinos.append({
            'data': data_atual,
            'dia_da_semana': data_atual.strftime('%A'),
            'treino_feito': False,
            'qtd': 0
        })


# Interface Streamlit
st.title("Simulador de Treinos")

# Entrada de dados para intensidades
st.sidebar.header("Defina as probabilidades de treino")
intensidade_segunda = st.sidebar.slider("Segunda-feira", 0, 100, 80) / 100
intensidade_terca = st.sidebar.slider("Terça-feira", 0, 100, 60) / 100
intensidade_quarta = st.sidebar.slider("Quarta-feira", 0, 100, 80) / 100
intensidade_quinta = st.sidebar.slider("Quinta-feira", 0, 100, 50) / 100
intensidade_sexta = st.sidebar.slider("Sexta-feira", 0, 100, 50) / 100
intensidade_sabado = st.sidebar.slider("Sábado", 0, 100, 40) / 100
intensidade_domingo = st.sidebar.slider("Domingo", 0, 100, 0) / 100

st.sidebar.header("Defina os Treinos/Dia")
qtd_segunda = st.sidebar.slider("Segunda-feira", 0, 2, 2)
qtd_terca = st.sidebar.slider("Terça-feira", 0, 2, 1)
qtd_quarta = st.sidebar.slider("Quarta-feira", 0, 2, 2)
qtd_quinta = st.sidebar.slider("Quinta-feira", 0, 2, 1)
qtd_sexta = st.sidebar.slider("Sexta-feira", 0, 2, 1)
qtd_sabado = st.sidebar.slider("Sábado", 0, 2, 1)
qtd_domingo = st.sidebar.slider("Domingo", 0, 2, 0)

# Intensidades por dia da semana
intensidades = {
    0: intensidade_segunda,  # Segunda-feira
    1: intensidade_terca,    # Terça-feira
    2: intensidade_quarta,   # Quarta-feira
    3: intensidade_quinta,   # Quinta-feira
    4: intensidade_sexta,    # Sexta-feira
    5: intensidade_sabado,   # Sábado
    6: intensidade_domingo,  # Domingo
}

treinos = {
    0: qtd_segunda,  # Segunda-feira
    1: qtd_terca,    # Terça-feira
    2: qtd_quarta,   # Quarta-feira
    3: qtd_quinta,   # Quinta-feira
    4: qtd_sexta,    # Sexta-feira
    5: qtd_sabado,   # Sábado
    6: qtd_domingo,  # Domingo
}

# Data de início e fim
inicio = datetime.date(2025, 1, 6)
fim = datetime.date(2025, 12, 31)

# Simular os treinos
df_treinos, total_treinos = simular_treinos(intensidades, treinos, inicio, fim)

# Mostrar a simulação
st.write("Simulação de Treinos")
st.dataframe(df_treinos)

# Exibir o total de treinos realizados
st.write(f"Total de treinos realizados no ano: {total_treinos}")
