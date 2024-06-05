from matplotlib import pyplot as plt
import time
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle

class DataRegressorApp:
    def __init__(self):
        self.data = None
        self.report = None

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)

        st.success(f"Набор данных {file_path.name} успешно загружен.")
        st.table(self.data.sample(10))  

    def analyse_data(self):
        st.subheader("Информация о данных:")
        
        st.write("Общее количество столбцов : {}".format(len(self.data.columns)))
        st.write("Общее количество строк    : {}".format(len(self.data)))
        st.write("Описательная статистика данных:")
        st.write(self.data.describe(include='all'))

        with plt.style.context('dark_background'):
            st.write('Посмотрим на распределение целевого признака (price)')
            st.session_state.plt1 = plt.figure(figsize = (15,5))
            plt.subplot(1,2,2)
            sns.histplot(x = 'price', data = self.data, kde = True)
            plt.subplot(1,2,1)
            sns.boxplot(x = 'price', data = self.data)
            st.write(st.session_state.plt1)
            st.write('Несмотря на то, что среднее значение составляет около 20000, здесь мы видим, что медиана составляет ≈ 7500. Эта разница объясняется наличием двух разных билетов: бизнес и эконом.')

            st.write('Посмотрим теперь, как зависит цена билета от класса и авиокомпании самолета')
            st.session_state.plt2 = plt.figure()
            sns.barplot(x='airline',y='price',hue="class",data=self.data.sort_values("price"))
            st.write(st.session_state.plt2)
            st.write('Бизнес-рейсы доступны только в двух компаниях: Air India и Vistara. Кроме того, существует большая разница в ценах на билеты бизнес-класса, которая почти в 5 раз превышает стоимость билетов эконом-класса.')
            
            st.write('Рассмотрим также на зависимость стоимости от длительности рейса и класса билета')
            st.session_state.plt3 = plt.figure()
            sns.lineplot(data=self.data,x='duration',y='price',hue='class',palette='hls')
            plt.xlabel('Duration',fontsize=15)
            plt.ylabel('Price',fontsize=15)
            st.write(st.session_state.plt3)
            st.write('С увеличением продолжительности полета стоимость билета также увеличивается как в эконом-, так и в бизнес-классах.')

    def prepare_data(self):
        st.subheader("Предварительная обработка данных:")

        st.write(f'Число пропущенных значений во фрейме данных: {self.data.isnull().sum().sum ()}')
        st.write(f'Число дупликатов во фрейме данных: {len(self.data)- len(self.data.drop_duplicates())}')

        le=LabelEncoder()
        df = self.data
        for col in self.data.columns:
            if self.data[col].dtype=='object':
                df[col]=le.fit_transform(self.data[col])
        
        df.drop(['Unnamed: 0', 'flight'], axis=1, inplace=True)

        names = df.columns
        indexes = df.index

        mmscaler=MinMaxScaler(feature_range=(0,1))
        df = mmscaler.fit_transform(df)

        self.data = pd.DataFrame(df, columns=names, index=indexes)

        st.write('Фрейм данных после предобработки:')
        st.write(self.data.sample(5))
        st.write('- Число столбцов уменьшено на 2\n'+
                 '- Все категориальные признаки закодированы с помощью LabelEncoder\n'+
                 '- Все столбцы масштабированы в интервале (0,1) с помощью MinMaxScaler')

    def train_model(self):
        X = self.data.drop('price', axis=1)
        y = self.data['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with open('../models/model_SVR.pkl', 'rb') as file: # rb stands for read binary which is used to deserialize the ML model from the working environment
            model = pickle.load(file)
        predictions = model.predict(X_test)
        self.report = {
            'mean_squared_error': mean_squared_error(y_test, predictions),
            'r2_score': r2_score(y_test, predictions)
        }

    @st.cache_data
    def save_report(_self, _file_path):
        with open(_file_path, 'w') as f:
            f.write(str(_self.report))

# Создание экземпляра приложения
app = DataRegressorApp()

st.title('Регрессия данных')

st.header('Загрузка данных')
file_path = st.file_uploader("Загрузите файл с данными о бронировании авиабилетов", type=['csv'])
if file_path is not None:
    app.load_data(file_path)
    st.header('Разведочный анализ и обработка данных')
    app.analyse_data()
    app.prepare_data()
    st.header('Обучение классификатора опорных векторов (SVC)')
    app.train_model()
    st.header("Отчет о проделанной работе")
    st.json(str(app.report))

    save_button = st.button("Сохранить отчет")
    if save_button:
        app.save_report('../reports/regression_report.txt')
        st.success('Успешно сохранено! 😊')
