
# Система прогнозирования неисправностей медицинского оборудования

## Описание проекта

Этот проект представляет собой систему для прогнозирования неисправностей медицинского оборудования на основе данных с датчиков. Система включает симуляцию датасета, обработку данных, предиктивную аналитику с использованием Random Forest, управление обслуживанием, систему поддержки принятия решений (СППР), визуализацию и генерацию отчетов. Также реализован веб-интерфейс с использованием Flask для мониторинга и отображения прогнозов.

## Требования

- Python 3.12 или выше
- Установленные библиотеки:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - flask
  - reportlab
  - xlsxwriter

## Установка

1. Клонируйте репозиторий или создайте проект локально.
2. Установите зависимости, выполнив следующую команду в терминале:

   ```bash
   pip install numpy pandas matplotlib scikit-learn flask reportlab xlsxwriter
   ```

3. Скачайте шрифт `DejaVuSans.ttf` с сайта DejaVu Fonts и поместите его в корневую папку проекта.

## Структура проекта

- `main.py`: Основной файл с кодом системы.
- `templates/`: Папка с HTML-шаблонами (например, `index.html`, `monitor.html`, `predictions.html`).
- `DejaVuSans.ttf`: Файл шрифта для генерации PDF-отчетов с поддержкой кириллицы.

## Запуск проекта

1. Убедитесь, что все зависимости установлены и шрифт `DejaVuSans.ttf` находится в папке проекта.
2. Выполните скрипт `main.py`:

   ```bash
   python main.py
   ```

3. Откройте браузер и перейдите по адресу `http://127.0.0.1:5000/` для доступа к веб-интерфейсу.

## Использование

- **Главная страница (`/`)**: Отображает приветственное сообщение и ссылки на мониторинг и прогнозы.
- **Мониторинг (`/monitor`)**: Показывает первые записи данных датчиков.
- **Прогнозы (`/predictions`)**: Отображает точность модели предиктивной аналитики.

После запуска будут сгенерированы файлы `report.pdf` и `report.xlsx` с отчетами о состоянии оборудования.

## Описание кода

### Импорт библиотек

Используются библиотеки для работы с данными, машинным обучением, визуализацией и веб-разработкой.

### Создание симулированного датасета

Генерируются синтетические данные для изображений (100x100 RGB) и параметров датчиков (температура, давление, вибрация и т.д.). Метки неисправностей (`failure`) принимают значения 0 (исправно) или 1 (неисправно).

### Модули системы

- **Сбор данных (`collect_data`)**: Возвращает датасет, изображения и метки.
- **Обработка данных (`process_data`)**: Нормализует значения датчиков, сохраняя целевую переменную `failure`.
- **Предиктивная аналитика (`predictive_analytics`)**: Обучает модель Random Forest и вычисляет точность.
- **Управление обслуживанием (`manage_maintenance`)**: Генерирует предупреждения на основе вероятностей (порог 0.7).
- **СППР (`dss_module`)**: Предоставляет рекомендации (например, "Возможный перегрев").
- **Визуализация (`visualize_performance`)**: Создает гистограмму точности.
- **Генерация отчетов (`generate_report`)**: Создает PDF и Excel-отчеты с использованием `reportlab` и `xlsxwriter`.

## Веб-интерфейс

Реализован с помощью Flask с тремя маршрутами: `/`, `/monitor`, `/predictions`. Шаблоны HTML находятся в папке `templates`.

## Выводы и точность

Текущая точность модели составляет около 50%, что связано с синтетической природой данных. Для повышения точности рекомендуется использовать реальный датасет (например, из MIMIC-III).

## Возможные улучшения

- Замена синтетических данных на реальные.
- Настройка гиперпараметров модели (например, с помощью `GridSearchCV`).
- Добавление интерактивных визуализаций на веб-странице.
- Расширение функциональности веб-интерфейса (например, добавление форм для ввода данных).

## Лицензия

Проект распространяется без лицензии (по умолчанию). Для использования в коммерческих целях уточните условия у автора.
