from django.shortcuts import render
import joblib 
import pandas as pd

model = joblib.load('titanic_project/titanic_rf_model.pkl')

def func(request):
        data = {}
        if request.method == 'POST':
            age = float(request.POST['Age'])
            sibsp = int(request.POST['SibSp'])
            parch = int(request.POST['Parch'])
            fare = float(request.POST['Fare'])
            pclass = int(request.POST['Pclass'])
            sex = request.POST['Sex']
            embarked = request.POST['Embarked']


            df = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex, embarked]],
                columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'])


            df = pd.get_dummies(df , columns = ['Sex' , 'Embarked'] , drop_first = True)


            required_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'Sex_male']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0


            df = df[required_columns]

 
            pred = model.predict(df)[0]
            if pred == 1:
               result = 'Survived 😊'
            else:
               result = 'Did not Survive 😖'

            data = {
                'age' : age,
                'sibsp' : sibsp,
                'parch' : parch,
                'fare' : fare,
                'pclass' : pclass,
                'sex' : sex,
                'embarked' : embarked,
                'result' : result
            }

            return render(request , 'index.html' , data)


        return render(request , 'index.html')