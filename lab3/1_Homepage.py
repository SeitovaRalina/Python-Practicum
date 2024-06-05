import streamlit as st

st.title('Главная страница')

st.subheader("Программа  для классификации и регрессии объектов реализует следующие этапы работы с данными:")
st.write("1) Подготовка данных: Преобразование данных в корректную форму для классификации и обработка любых аномалий в этих данных.\n"+
         "2) Создание обучающих наборов: Разделение данных на обучающие и тестовые наборы.\n"+
         "3) Создание классификатора.\n"+
         "4) Обучение классификатора.\n"+
         "5) Оценка производительности и настройка параметров.")

st.subheader('Классификация на датасете "Pumpkin Seeds Dataset"')

st.write('Для решения задачи классификации был выбран набор данных о семенах тыквы.')
st.markdown('**ВСТУПЛЕНИЕ:**')
st.write('Тыквенные семечки часто употребляются в качестве кондитерских изделий во всем мире из-за их достаточного содержания белков, жиров, углеводов и минералов. Это исследование было проведено на двух наиболее важных и качественных сортах тыквенных семечек, ‘Urgup_Sivrisi’ и ‘Cercevelik’, которые обычно выращиваются в регионах Ургуп и Каракаорен в Турции.')

st.markdown('**ДАТАСЕТ:**')
st.write('В общей сложности было получено 2500 изображений семян тыквы двух сортов. Изображения были обработаны и рассчитаны признаки. Для каждого семени тыквы было получено 12 морфологических признаков.')

st.markdown("**ПРИЗНАКИ:**")
st.write('Чистый набор данных содержит следующие признаки: ')
st.write('1) `Area` - Количество пикселей в изображении тыквенного семечка.\n'+
         '2) `Perimeter` - Длина тыквенного семечка в пикселях.\n'+
         '3) `Major_Axis_Length` - Максимальное расстояние между осями тыквенного семечка.\n'+
         '4) `Minor_Axis_Length` - Минимальное расстояние между осями тыквенного семечка.\n'+
         '5) `Convex_Area` - Отношение площади тыквенного семечка к пикселям ограничивающего прямоугольника.\n'+
         '6) `Equiv_Diameter` - Диаметр кубика, площадь которого равна площади тыквы.\n'+
         '7) `Eccentricity` - Эксцентриситет тыквенного семечка.\n'+
         '8) `Solidity` - Выпуклость тыквенных семечек.\n'+
         '9) `Extent` - Отношение площади тыквенного семечка к пикселям ограничивающей рамки.\n'+
         '10) `Roundness` - Овальность тыквенных семечек без учета их искажения по краям.\n'+
         '11) `Aspect_Ratio` - Соотношение сторон тыквенных семечек.\n'+
         '12) `Compactness` - Площадь тыквенного семечка относительно площади круга с такой же длиной окружности.\n'+
         '13) `Class` - Целевой признак с двумя значениями "Cercevelik" и "Urgup Sivrisi".')


st.subheader('Регрессия на датасете "Flight Price Prediction"')
st.markdown('**ВСТУПЛЕНИЕ:**')
st.write('Цель исследования - проанализировать набор данных о бронировании авиабилетов, полученный с веб-сайта “Ease My Trip”. "Easemytrip" - это интернет-платформа для бронирования авиабилетов и, следовательно, платформа, которую потенциальные пассажиры используют для покупки билетов. Тщательное изучение данных поможет получить ценную информацию, которая будет иметь огромное значение для пассажиров.')

st.markdown('**ДАТАСЕТ:**')
st.write('Набор данных содержит информацию о вариантах бронирования авиабилетов с веб-сайта Easemytrip для перелетов между 6 крупнейшими городами Индии. В очищенном наборе данных 300261 строк и 11 столбцов.')

st.markdown("**ПРИЗНАКИ:**")
st.write('Чистый набор данных содержит следующие признаки: ')
st.write('1) `Airline` - Название авиакомпании хранится в столбце авиакомпания. Это категориальный признак, включающий 6 различных авиакомпаний.\n'+
         '2) `Flight` - Рейс хранит информацию о коде рейса самолета. Это категориальный признак.\n'+
         '3) `Source City` - город, из которого вылетает рейс. Это категориальный признак, содержащий 6 уникальных городов.\n'+
         '4) `Departure Time` -  Это производный категориальный признак, полученный путем группировки периодов времени в ячейки. Он хранит информацию о времени вылета и имеет 6 уникальных меток времени.\n'+
         '5) `Stops` - Категориальный признак с 3 различными значениями, который хранит количество остановок между городами отправления и назначения.\n'+
         '6) `Arrival Time` - Это производный категориальный признак, созданный путем группировки временных интервалов в ячейки. Он имеет шесть различных временных меток и хранит информацию о времени прибытия.\n'+
         '7) `Destination City` - Город, в котором приземлится рейс. Это категориальный признак, содержащий 6 уникальных городов.\n'+
         '8) `Class` - Категориальный признак, содержащий информацию о классе посадочного места; он имеет два различных значения: бизнес и эконом.\n'+
         '9) `Duration` - Общее количество времени, необходимое для поездки между городами в часах.\n'+
         '10) `Days Left` - Это выводимая характеристика, которая рассчитывается путем вычитания даты поездки из даты бронирования.\n'+
         '11) `Price` - Целевая переменная, которая хранит информацию о цене билета.')