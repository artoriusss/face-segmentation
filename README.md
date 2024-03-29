# face-segmentation
# Виявлення обличчя та сегментація з вбудованням та кластеризацією

Цей проект призначений для виявлення облич та їх сегментації, вбудовання та кластеризації на заданому наборі зображень, які містять людські обличчя. Він використовує різні глибинні моделі для вбудовання облич, такі як VGG16, і використовує алгоритми PCA та K-Means для кластеризації схожих облич.

## Функціональність

- Виявлення облич: Код містить функціональність для виявлення людських облич на зображеннях за допомогою різних попередньо навчених моделей.
- Сегментація облич: Застосовуються техніки попередньої обробки зображень для сегментації виявлених облич від фону.
- Вбудовання облич: Використовується бібліотека DeepFace для вбудовання облич у високовимірні вектори.
- Кластеризація: Використовується PCA для зменшення розмірності та K-Means для групування схожих вбудованих векторів облич.
- Візуалізація: Надає візуалізацію згрупованих облич для зручного сприйняття.

## Структура файлів

- `main.py`: Головна точка входу додатку, яка координує весь процес від завантаження даних до візуалізації.
- `deep_face_embedder.py`: Містить клас `DeepFaceEmbedder`, відповідальний за вбудовання облич та виконання кластеризації за допомогою різних методів.
- `clustering_vgg16.py`: Реалізує вбудовання облич та кластеризацію за допомогою моделі VGG16.
- `path.py`: Містить шляхи до каталогів даних та оброблюваних зображень.
- `setup.py`: Налаштування для встановлення пакету.
- `data_helper.py`: Містить клас `DataHelper`, який надає допоміжні функції для роботи з даними, такі як перевірка облич, отримання межі рамок та розміщення зображень.
- `data_transformation.py`: Містить клас `DataTransformation`, який здійснює перетворення даних, такі як масштабування та PCA.
- `face_croper.py`: Містить клас `FaceCroper`, який відповідає за обрізку облич та вирівнювання їх на зображеннях.
