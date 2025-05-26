# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import xlsxwriter
import os

# 1) Создание симулированного датасета
# 1.1) Датасет с изображениями (Симулированные метки: 0 = исправно, 1 = неисправно)
np.random.seed(42)
n_samples = 1000
image_data = np.random.rand(n_samples, 100, 100, 3)  # Симулированные изображения 100x100 RGB
image_labels = np.random.randint(0, 2, n_samples)  # Бинарные метки

# 1.2) Датасет с параметрами датчиков (Симулированные параметры)
sensor_data = pd.DataFrame({
    'temperature': np.random.uniform(20, 50, n_samples),  # Температура
    'pressure': np.random.uniform(100, 200, n_samples),  # Давление
    'voltage': np.random.uniform(110, 130, n_samples),  # Напряжение
    'power_consumption': np.random.uniform(50, 200, n_samples),  # Энергопотребление
    'resistance': np.random.uniform(0.1, 10, n_samples),  # Сопротивление
    'cpu_frequency': np.random.uniform(1, 3, n_samples),  # Частота процессора
    'vibration': np.random.uniform(0, 5, n_samples),  # Вибрация
    'start_time': np.random.uniform(0.1, 2, n_samples),  # Время запуска
    'response_time': np.random.uniform(0.05, 1, n_samples),  # Время отклика
    'cycle_completion_time': np.random.uniform(1, 10, n_samples),  # Время завершения циклов
    'command_delay': np.random.uniform(0.01, 0.5, n_samples),  # Задержка команд
    'failure': image_labels  # Те же метки, что и для изображений
})


# 1.3) Модуль сбора данных
def collect_data():
    """Симулирует сбор данных с датчиков и изображений."""
    return sensor_data, image_data, image_labels


# 1.4) Модуль обработки данных
def process_data(sensor_df, image_data, labels):
    """Обрабатывает сырые данные, нормализуя значения датчиков, исключая столбец failure."""
    sensor_features = sensor_df.drop('failure', axis=1)
    sensor_processed = (sensor_features - sensor_features.mean()) / sensor_features.std()
    sensor_processed['failure'] = sensor_df['failure']
    return sensor_processed, image_data, labels


# 1.5) Модуль предиктивной аналитики
def predictive_analytics(sensor_data):
    """Обучает модель Random Forest для прогнозирования неисправностей."""
    X = sensor_data.drop('failure', axis=1)
    y = sensor_data['failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, report, X_test


# 1.6) Модуль управления обслуживанием
def manage_maintenance(model, X_test, threshold=0.7):
    """Генерирует предупреждения о необходимости обслуживания на основе вероятностей."""
    probas = model.predict_proba(X_test)[:, 1]
    alerts = probas > threshold
    return alerts


# 1.7) Модуль СППР (Система поддержки принятия решений)
def dss_module(alerts):
    """Предоставляет рекомендации для инженеров о возможных проблемах."""
    issues = []
    if any(alerts):
        issues = ["Возможный перегрев", "Неустойчивость напряжения", "Проблема с механической вибрацией"]
    return issues


# 1.8) Модуль визуализации
def visualize_performance(accuracy):
    """Визуализирует точность работы системы."""
    plt.figure(figsize=(8, 6))
    plt.bar(['Точность'], [accuracy], color='blue')
    plt.title('Точность модели')
    plt.ylabel('Точность (%)')
    plt.ylim(0, 1)
    plt.show()


# 1.9) Веб-интерфейс (Базовое приложение Flask)
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/monitor')
def monitor():
    return render_template('monitor.html', data=sensor_data.head().to_dict())


@app.route('/predictions')
def predictions():
    _, accuracy, _, _ = predictive_analytics(sensor_data)
    return render_template('predictions.html', accuracy=accuracy)


# 1.10) Модуль генерации отчетов
def generate_report(accuracy, sensor_data):
    # Отчет в PDF с использованием reportlab
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)

    # Указываем путь к шрифту
    font_path = os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf')
    if not os.path.exists(font_path):
        print(f"Ошибка: Шрифт {font_path} не найден. Убедитесь, что файл DejaVuSans.ttf находится в папке проекта.")
        # Используем стандартный шрифт как запасной вариант (без поддержки кириллицы)
        p.setFont('Helvetica', 12)
    else:
        pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
        p.setFont('DejaVuSans', 12)

    # Заголовок отчета
    p.drawString(100, 750, f"Отчет о состоянии оборудования - Точность: {accuracy * 100:.2f}%")

    # Данные датасета
    y_position = 730
    for line in sensor_data.head().to_string().split('\n'):
        p.drawString(100, y_position, line)
        y_position -= 15

    p.showPage()
    p.save()
    pdf_output = buffer.getvalue()
    buffer.close()

    # Отчет в Excel
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    for row_num, row_data in enumerate(sensor_data.head().values):
        for col_num, value in enumerate(row_data):
            worksheet.write(row_num, col_num, value)
    workbook.close()
    output.seek(0)

    return pdf_output, output.getvalue()


# Запуск системы
if __name__ == "__main__":
    # Сбор и обработка данных
    sensor_df, img_data, labels = collect_data()
    sensor_processed, _, _ = process_data(sensor_df, img_data, labels)

    # Предиктивный анализ
    model, accuracy, report, X_test = predictive_analytics(sensor_processed)
    print(f"Точность: {accuracy * 100:.2f}%")
    print(report)

    # Управление обслуживанием и СППР
    alerts = manage_maintenance(model, X_test)
    issues = dss_module(alerts)
    print("Предупреждения о обслуживании:", alerts.sum(), "проблемы обнаружены")
    print("Возможные проблемы:", issues)

    # Визуализация
    visualize_performance(accuracy)

    # Генерация отчетов
    pdf_report, excel_report = generate_report(accuracy, sensor_processed)
    with open('report.pdf', 'wb') as f:
        f.write(pdf_report)
    with open('report.xlsx', 'wb') as f:
        f.write(excel_report)

    # Запуск веб-приложения
    app.run(debug=True, port=5000)
