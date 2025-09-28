# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# %% [code]
# Шаг 2: Загрузка данных и создание временной оси по накопленному итогу

# Загрузка данных ГВС
df_gvs = pd.read_excel('Посуточная ведомость ОДПУ ГВС.xlsx')

# Удаляем строки с заголовками
df_gvs = df_gvs[~df_gvs['Дата'].astype(str).str.contains(r'^\s*Дата\s*$', na=False)].copy()

# Выбираем нужные столбцы
df_gvs = df_gvs[['Дата', 'Время суток, ч', 'Потребление за период, м3', 'Т1 гвс, оС', 'Т2 гвс, оС']]
df_gvs.columns = ['date_str', 'time_str', 'Consumption_GVS', 'Temp_GVS_Supply', 'Temp_GVS_Return']

# Загрузка данных ХВС
df_hvs = pd.read_excel('Посуточная_ведомость_водосчетчика_ХВС_ИТП.xlsx')

# Удаляем строки с заголовками
df_hvs = df_hvs[~df_hvs['Дата'].astype(str).str.contains(r'^\s*Дата\s*$', na=False)].copy()

# Выбираем нужные столбцы
df_hvs = df_hvs[['Дата', 'Время суток, ч', 'Потребление накопленным итогом, м3', 'Потребление за период, м3']]
df_hvs.columns = ['date_str', 'time_str', 'Cumulative_HVS', 'Consumption_HVS']

# !!!! КЛЮЧЕВОЙ ШАГ: Сортируем ХВС по накопленному итогу (это наша временная ось) и создаем индекс
df_hvs = df_hvs.sort_values('Cumulative_HVS').reset_index(drop=True)
df_hvs['time_index'] = df_hvs.index  # Номер строки = номер часа (0, 1, 2, 3...)

# !!!! Теперь создаем такой же индекс для ГВС: сортируем по дате и времени, чтобы порядок совпал
df_gvs = df_gvs.sort_values(['date_str', 'time_str']).reset_index(drop=True)
df_gvs['time_index'] = df_gvs.index  # Номер строки = номер часа

# !!!! Объединяем по time_index — это гарантирует 1:1 соответствие
df = df_gvs.merge(df_hvs[['time_index', 'Consumption_HVS']], on='time_index', how='inner')

# Создаем производные признаки
df['Delta_GVS_HVS'] = df['Consumption_GVS'] - df['Consumption_HVS']
df['Temp_Delta'] = df['Temp_GVS_Supply'] - df['Temp_GVS_Return']

# Извлекаем час из 'time_str' (например, '0-1' → 0, '19-20' → 19)
df['Hour'] = df['time_str'].str.split('-').str[0].astype(int)

# Добавляем признаки дня недели (для сезонности)
# Поскольку мы не знаем реальные даты, но знаем порядок, мы можем вычислить день недели по номеру дня
# Предположим, что первый день — 01.04.2025 (суббота)
# 01.04.2025 — суббота → dayofweek = 5
# Считаем: 01.04.2025 — это time_index = 0 → dayofweek = 5
# Тогда: dayofweek = (5 + time_index) % 7
df['DayOfWeek'] = (5 + df['time_index']) % 7
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# Удаляем строки с NaN (если есть)
df = df.dropna().reset_index(drop=True)

print("✅ Данные успешно объединены по временному индексу!")
print(f"Размер данных: {df.shape}")
print("\nПервые 5 строк:")
print(df[['time_index', 'date_str', 'time_str', 'Consumption_GVS', 'Consumption_HVS', 'Delta_GVS_HVS', 'Temp_GVS_Supply', 'Hour', 'DayOfWeek']].head())

# %% [code]
# Шаг 3: Подготовка данных для моделей

feature_columns = ['Consumption_GVS', 'Consumption_HVS', 'Temp_GVS_Supply', 'Temp_GVS_Return', 'Delta_GVS_HVS', 'Temp_Delta', 'Hour', 'DayOfWeek', 'IsWeekend']
target_column = 'Consumption_GVS'

# ✅ СКАЛЕР ДЛЯ NARX-LSTM — ОБУЧАЕМ НА 9 ПРИЗНАКАХ
scaler_narx = MinMaxScaler()
scaled_data_narx = scaler_narx.fit_transform(df[feature_columns])

# ✅ СКАЛЕР ДЛЯ LSTM-AE — ОБУЧАЕМ НА 8 ПРИЗНАКАХ (без Consumption_GVS)
ae_feature_columns = ['Consumption_HVS', 'Temp_GVS_Supply', 'Temp_GVS_Return', 'Delta_GVS_HVS', 'Temp_Delta', 'Hour', 'DayOfWeek', 'IsWeekend']
scaler_ae = MinMaxScaler()  # НОВЫЙ, НЕЗАВИСИМЫЙ скалер!
scaled_data_ae = scaler_ae.fit_transform(df[ae_feature_columns])

# Функция для создания последовательностей (окон)
def create_sequences(data, target, window_size=24):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])  # Последние `window_size` часов
        y.append(target[i])             # Целевое значение на следующий час
    return np.array(X), np.array(y)

WINDOW_SIZE = 24

# Создание последовательностей для NARX-LSTM
X_narx, y_narx = create_sequences(scaled_data_narx, df[target_column].values, WINDOW_SIZE)

# Разделение на train/val/test (80%/10%/10%)
split_train = int(0.8 * len(X_narx))
split_val = int(0.9 * len(X_narx))

X_train_narx, X_val_narx, X_test_narx = X_narx[:split_train], X_narx[split_train:split_val], X_narx[split_val:]
y_train_narx, y_val_narx, y_test_narx = y_narx[:split_train], y_narx[split_train:split_val], y_narx[split_val:]

print(f"\nNARX-LSTM: X_train={X_train_narx.shape}, y_train={y_train_narx.shape}")

# Создание последовательностей для LSTM-AE
X_ae, y_ae = create_sequences(scaled_data_ae, scaled_data_ae, WINDOW_SIZE)  # AE восстанавливает вход

X_train_ae, X_val_ae, X_test_ae = X_ae[:split_train], X_ae[split_train:split_val], X_ae[split_val:]
y_train_ae, y_val_ae, y_test_ae = y_ae[:split_train], y_ae[split_train:split_val], y_ae[split_val:]

print(f"LSTM-AE: X_train_ae={X_train_ae.shape}")

# %% [code]
# Шаг 4: Создание и обучение моделей

# Модель 1: NARX-LSTM
model_narx = Sequential([
    LSTM(128, activation='relu', input_shape=(WINDOW_SIZE, len(feature_columns)), return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model_narx.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print("\n=== NARX-LSTM ===")
model_narx.summary()

history_narx = model_narx.fit(
    X_train_narx, y_train_narx,
    validation_data=(X_val_narx, y_val_narx),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Модель 2: LSTM-Autoencoder
model_ae = Sequential([
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    RepeatVector(WINDOW_SIZE),
    LSTM(32, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(len(ae_feature_columns)))
])

model_ae.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
print("\n=== LSTM-AE ===")
model_ae.summary()

history_ae = model_ae.fit(
    X_train_ae, X_train_ae,
    validation_data=(X_val_ae, X_val_ae),
    epochs=50,
    batch_size=32,
    verbose=1
)

# %% [code]
# Шаг 5: Оценка и визуализация

# 5.1 NARX-LSTM — инвертируем масштабирование с ПРАВИЛЬНЫМ скалером
y_pred_narx = model_narx.predict(X_test_narx)

# Используем scaler_narx, а не scaler_ae!
y_test_narx_orig = scaler_narx.inverse_transform(
    np.concatenate([y_test_narx.reshape(-1,1), np.zeros((len(y_test_narx), len(feature_columns)-1))], axis=1)
)[:, 0]

y_pred_narx_orig = scaler_narx.inverse_transform(
    np.concatenate([y_pred_narx, np.zeros((len(y_pred_narx), len(feature_columns)-1))], axis=1)
)[:, 0]

mae_narx = mean_absolute_error(y_test_narx_orig, y_pred_narx_orig)
rmse_narx = np.sqrt(mean_squared_error(y_test_narx_orig, y_pred_narx_orig))

print(f"\n=== Оценка NARX-LSTM ===")
print(f"MAE: {mae_narx:.4f} м³")
print(f"RMSE: {rmse_narx:.4f} м³")

# Визуализация прогноза
plt.figure(figsize=(15, 6))
plt.plot(range(len(y_test_narx_orig)), y_test_narx_orig, label='Фактическое', color='blue', alpha=0.7)
plt.plot(range(len(y_pred_narx_orig)), y_pred_narx_orig, label='Прогноз', color='red', alpha=0.7)
plt.title('Прогноз расхода ГВС: Фактические vs Прогнозируемые значения')
plt.xlabel('Время (часы)')
plt.ylabel('Потребление, м³')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5.2 LSTM-AE — определяем порог
X_test_ae_pred = model_ae.predict(X_test_ae)
mse_errors = np.mean(np.square(X_test_ae - X_test_ae_pred), axis=(1, 2))

X_val_ae_pred = model_ae.predict(X_val_ae)
mse_errors_val = np.mean(np.square(X_val_ae - X_val_ae_pred), axis=(1, 2))
threshold = np.percentile(mse_errors_val, 99)

print(f"\n=== Оценка LSTM-AE ===")
print(f"Порог детекции аномалий (99-й перцентиль): {threshold:.6f}")

anomalies = mse_errors > threshold
anomaly_count = np.sum(anomalies)
print(f"Обнаружено аномалий в тестовом наборе: {anomaly_count} из {len(anomalies)} ({anomaly_count/len(anomalies)*100:.2f}%)")

# Визуализация ошибок
plt.figure(figsize=(15, 6))
plt.plot(range(len(mse_errors)), mse_errors, label='Ошибка восстановления', color='blue', alpha=0.7)
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Порог (99%): {threshold:.6f}')
plt.title('Ошибка восстановления LSTM-AE и порог детекции аномалий')
plt.xlabel('Время (окна по 24 часа)')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Визуализация Delta_GVS_HVS с аномалиями
anomaly_labels = np.zeros(len(df))
for i in range(split_val, len(df) - WINDOW_SIZE):
    if i - split_val < len(anomalies) and anomalies[i - split_val]:
        anomaly_labels[i + WINDOW_SIZE - 1] = 1

plt.figure(figsize=(15, 6))
plt.plot(df['time_index'], df['Delta_GVS_HVS'], label='Delta_GVS_HVS', color='green', alpha=0.7)
plt.scatter(df['time_index'][anomaly_labels == 1], df['Delta_GVS_HVS'][anomaly_labels == 1],
            color='red', s=50, marker='x', label='Аномалия (LSTM-AE)', zorder=5)
plt.title('Динамика Delta_GVS_HVS и обнаруженные аномалии')
plt.xlabel('Время (номер часа с начала наблюдения)')
plt.ylabel('Delta_GVS_HVS (м³)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [code]
# Дополнительно: визуализация обучения
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# NARX-LSTM
axes[0].plot(history_narx.history['loss'], label='Train Loss')
axes[0].plot(history_narx.history['val_loss'], label='Val Loss')
axes[0].set_title('NARX-LSTM: Потери (MSE)')
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Потери')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LSTM-AE
axes[1].plot(history_ae.history['loss'], label='Train Loss')
axes[1].plot(history_ae.history['val_loss'], label='Val Loss')
axes[1].set_title('LSTM-AE: Потери (MAE)')
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Потери')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()