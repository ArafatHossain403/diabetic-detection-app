import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import streamlit as st


st.subheader('Logistic Regression Model')

st.write("""
Diabetes Detection App
""")
image = Image.open('C:/Users/Hi/PycharmProjects/webapps/download.jpeg')
st.image(image, caption='ML', use_column_width=True)

df = pd.read_csv("C:/Users/Hi/PycharmProjects/webapps/diabetes.csv")

st.subheader('Data Information')
st.dataframe(df)
st.write(df.describe())
chart = st.bar_chart(df)

X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


def getuser():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 32.0)
    bmi = st.sidebar.slider('bmi', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('dpf', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'bloodPressure': blood_pressure,
                 'skinThickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': bmi,
                 'diabetesPedigreeFunction': dpf,
                 'age': age
                 }

    features = pd.DataFrame(user_data, index=[0])
    return features


user_input = getuser()

st.subheader('user Input:')
st.write(user_input)

lr = LogisticRegression()
lr.fit(X_train, Y_train)


st.subheader('Model Test Accuracy Score using Logistic Regression ')
st.write(str(accuracy_score(Y_test, lr.predict(X_test)) * 100)+'%')
prediction = lr.predict(user_input)
# st.subheader('Classification: ')
# st.write(prediction)

# OUTPUT
st.subheader('Your Report: ')
output = " "
if prediction[0] == 0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)