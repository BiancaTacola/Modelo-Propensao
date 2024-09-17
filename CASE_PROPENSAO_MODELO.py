from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# (Por problemas de importação, preferi utilizar a Keras nas linhas acima,para acessar o tensorflow)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Importação para métricas de desempenho
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Importação para Curva Roc e AUC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Verificando a importância das variáveis
from sklearn.ensemble import RandomForestClassifier


# Carregar a base de dados original com a coluna 'Alta'
df = pd.read_csv(
    'C:\\Bianca\\Bianca\\Empregabilidade\\2024\\CASE PROPENSÃO PYTHON\\CASE PROPENSÃO PYTHON\\dados_clientes.csv', encoding='utf-8-sig')

# Converter variáveis categóricas em variáveis numéricas
label_encoder_uf = LabelEncoder()
label_encoder_setor = LabelEncoder()
df['UF'] = label_encoder_uf.fit_transform(df['UF'])
df['SETOR CNAE'] = label_encoder_setor.fit_transform(df['SETOR CNAE'])

# Dividir os dados em treino e teste
X = df.drop('Alta', axis=1)
y = df['Alta']

# Obter uma lista de colunas contendo dados categóricos
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

# Obter uma lista de colunas contendo dados numéricos
numerical_cols = [col for col in X.columns if X[col].dtype != 'object']

# Aplicar normalização aos dados numéricos usando MinMaxScaler
scaler = MinMaxScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Aplicar one-hot encoding aos dados categóricos usando OneHotEncoder
encoder = OneHotEncoder()
ct = ColumnTransformer(
    transformers=[("encoder", encoder, categorical_cols)], remainder='passthrough')
X = ct.fit_transform(X)

# Dividir os dados em conjuntos de treino e teste com 70%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Balancear as classes usando SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Verificar se X_train_balanced é uma matriz esparsa antes de converter
if not isinstance(X_train_balanced, np.ndarray):
    X_train_balanced = X_train_balanced.toarray()

# Verificar se X_test é uma matriz esparsa antes de converter
if not isinstance(X_test, np.ndarray):
    X_test = X_test.toarray()

# Converter pandas dataframe para numpy array
y_train_balanced = y_train_balanced.values
y_test = y_test.values

######################### model ################################
# Construa o modelo
model = Sequential()
model.add(Dense(units=216, activation="relu"))
model.add(Dense(units=162, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=108, activation="relu"))
model.add(Dense(units=54, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

######################################## Testando Melhorias####################
# Ajustando hiperparâmetros: número de camadas e unidades
# Testando com uma taxa de aprendizado diferente
optimizer = Adam(learning_rate=0.001)

# Compilando o modelo com a nova configuração
model.compile(loss="binary_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])

# Treinamento do modelo
# model.fit(X_train_balanced, y_train_balanced, epochs=50,batch_size=32, validation_data=(X_test, y_test))

#########################################################################

# Compile o modelo
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Defina o Early Stopping
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=20, restore_best_weights=True)

# Treine o modelo com Early Stopping
history = model.fit(X_train_balanced, y_train_balanced, epochs=1000,
                    batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Fazer previsões no conjunto de teste
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

################################################################################################
# Calcular as métricas de desempenho (Novas linhas de codigo)*******
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Acurácia: {accuracy:.4f}')
print(f'Precisão: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Obter as probabilidades de predição do modelo
# Flatten para usar no cálculo da ROC
y_pred_prob = model.predict(X_test).ravel()

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calcular a AUC
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='Curva ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Exibir a AUC
print(f"AUC: {roc_auc:.4f}")
# ##############################################################################################

# Carregar a nova base de dados sem a coluna 'Alta'
df_nova_base = pd.read_csv(
    'C:\\Bianca\\Bianca\\Empregabilidade\\2024\\CASE PROPENSÃO PYTHON\\CASE PROPENSÃO PYTHON\\dados_clientes_sem_alta.csv', encoding='utf-8-sig')

# Manter uma cópia das colunas originais de UF e SETOR CNAE para restaurar os valores originais depois das previsões
df_nova_base_original_uf_setor = df_nova_base[['UF', 'SETOR CNAE']].copy()

# Converter variáveis categóricas em variáveis numéricas na nova base de dados
df_nova_base['UF'] = label_encoder_uf.transform(df_nova_base['UF'])
df_nova_base['SETOR CNAE'] = label_encoder_setor.transform(
    df_nova_base['SETOR CNAE'])

# Padronizar os dados da nova base de dados
df_nova_base_scaled = scaler.transform(df_nova_base)

# Fazer previsões na nova base de dados e obter as probabilidades
clientes_propensos_prob = model.predict(df_nova_base_scaled)

# Adicionar a coluna 'Propenso' ao DataFrame da nova base de dados
df_nova_base['Propenso'] = clientes_propensos_prob

# Ordenar o DataFrame pela coluna 'Propenso' em ordem decrescente (formato decimal)
df_nova_base = df_nova_base.sort_values(by='Propenso', ascending=False)

# Adicionar a coluna 'Rank' ao DataFrame da nova base de dados (1 a 1000)
df_nova_base['Rank'] = range(1, len(df_nova_base) + 1)

# Ajustar a coluna 'Propenso' para o formato 0,00%
df_nova_base['Propenso'] = df_nova_base['Propenso'].apply(lambda x: f'{x:.2%}')

# Restaurar os valores originais das colunas UF e SETOR CNAE no DataFrame final
df_nova_base['UF'] = df_nova_base_original_uf_setor['UF']
df_nova_base['SETOR CNAE'] = df_nova_base_original_uf_setor['SETOR CNAE']

# Salvar o DataFrame formatado em um novo arquivo CSV
df_nova_base.to_csv('C:\\Bianca\\Bianca\\Empregabilidade\\2024\\CASE PROPENSÃO PYTHON\\CASE PROPENSÃO PYTHON\\clientes_propensos_com_rank_formatado_v1.csv',
                    index=False, encoding='utf-8-sig')
