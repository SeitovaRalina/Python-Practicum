from matplotlib import pyplot as plt
from sklearn.svm import SVC
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class DataClassifierApp:
    def __init__(self):
        self.data = None
        self.best_model = None
        self.report = None
    
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)  # Предполагается, что данные находятся в формате CSV.

        st.success(f"Набор данных {file_path.name} успешно загружен.")
        st.table(self.data.sample(10))  

    def analyse_data(self):
        st.subheader("Информация о данных:")
        
        st.write("Общее количество столбцов : {}".format(len(self.data.columns)))
        st.write("Общее количество строк    : {}".format(len(self.data)))
        st.write("Описательная статистика данных:")
        st.write(self.data.describe())

        st.write('Посмотрим на распределение целевого признака (Class) ')
        names = self.data['Class'].value_counts().index
        values = self.data['Class'].value_counts()
        fig = px.pie(names=names, values=values, title="Class Distribution", width=600)
        fig.update_layout({'title':{'x':0.5}})
        st.write(fig)
        st.write('Замечательно! Классы распределены почти поровну. Это гарантирует, что модель не может быть смещена в сторону какого-либо класса.')

        with plt.style.context('dark_background'):
            st.write("Также давайте взглянем на корреляцию между признаками набора данных")
            fig = plt.figure()
            sns.heatmap(self.data.corr(), annot=True, cmap='twilight_shifted', fmt=".2f")
            st.write(fig)
            st.write('Между признаками наблюдается сильная корреляция.')

            st.markdown('Рассмотрим пару сильно коррелирующих признаков `Area` и `Perimeter`')
            fig = plt.figure(figsize=(12,8))
            sns.scatterplot(data=self.data, x="Area", y="Perimeter", hue="Class", alpha=0.7, palette='twilight_shifted')
            st.write(fig)
            st.write('Линейную взаимосвязь можно легко обнаружить, но классы разделены недостаточно четко.')

    def prepare_data(self):
        st.subheader("Предварительная обработка данных:")

        st.write(f'Число пропущенных значений во фрейме данных: {self.data.isnull().sum().sum ()}')
        st.write(f'Число дупликатов во фрейме данных: {len(self.data)- len(self.data.drop_duplicates())}')
        self.data = pd.get_dummies(self.data, drop_first=True)
        st.write('Фрейм данных после предобработки')
        st.write(self.data.sample(5))
        
    def train_model(self):
        X = self.data.drop("Class_Ürgüp Sivrisi", axis=1)
        y = self.data["Class_Ürgüp Sivrisi"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.best_model = self._tune_hyperparameters(X_train, y_train)
        st.write('Подобранные гиперпараметры для SVM:')
        st.json(self.best_model.get_params())
    
        y_pred = self.best_model.predict(X_test)

        st.markdown(f'**Accuracy:** {accuracy_score(y_test, y_pred):3f}')
        st.markdown('**Сonfusion matrix:**')
        with plt.style.context('dark_background'):
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap='PuBu')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
        st.write(confusion_matrix(y_test, y_pred))

        self.report = classification_report(y_test, y_pred)

    @st.cache_data
    def _tune_hyperparameters(_self, _X_train, y_train):
        model = SVC()
        param_grid = {"C": [1,10,100,1000],
                      "gamma": ["scale", "auto"], 
                      "kernel": ['linear', 'poly', 'rbf']}
        grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
        grid.fit(_X_train, y_train)
        return grid.best_estimator_
    
    @st.cache_data
    def save_report(_self, _file_path):
        with open(_file_path, 'w') as f:
            f.write(_self.report)

app = DataClassifierApp()
st.title('Классификация данных')

st.header('Загрузка данных')
file_path = st.file_uploader("Загрузите файл с данными о семенах тыквы", type=['csv'])
if file_path is not None: 
    app.load_data(file_path)
    st.header('Разведочный анализ и обработка данных')
    app.analyse_data()
    app.prepare_data()
    st.header('Обучение классификатора опорных векторов (SVC)')
    app.train_model()
    st.header("Отчет о проделанной работе:")
    st.code(app.report, language='text')

    save_button = st.button("Сохранить отчет")
    if save_button:
        app.save_report('../reports/classification_report.txt')
        st.success('Успешно сохранено! 😊')