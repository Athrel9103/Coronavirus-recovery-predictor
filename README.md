# Coronavirus-recovery-predictor
import pandas as pd
import numpy as np
from tkinter import *
#Importing Dataset
dataset = pd.read_csv("COVID_Data.csv").dropna()
x = dataset.drop("death_yn", 1).iloc[:, 4:].values
y = dataset.iloc[:, -2].values

#Encoding Categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([("encoder", OneHotEncoder(), [0,1])], remainder="passthrough")
ct1 = ColumnTransformer([("encoder", OneHotEncoder(), [-4, -3, -2, -1])], remainder="passthrough")
x = np.array(ct.fit_transform(x))
x = np.array(ct1.fit_transform(x))

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Splitting Data into training and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)

#Training the Logistic Regression Model on Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(xTrain, yTrain)

#Predicting the Test Set Results
yPred = classifier.predict(xTest)
np.set_printoptions(precision=2)
# print(np.concatenate((yPred.reshape(len(yPred), 1), yTest.reshape(len(yTest), 1)), 1))#yPred.reshape(rows, columns). For the axis cargument, we can either put in 0 for vertical concatenation or 1 for horizontal concatonation
prob = classifier.predict_proba(xTest)
# print(prob)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(yTest, yPred)
# print(cm)#[[Real output is 0], [real output of 1]] and within that it is [[predicted 0, predicted 1], [predicted 0, predicted 1]]
# print(accuracy_score(yTest, yPred))

def predict_rate():
    x_pred = np.array([[sex.get(), age.get(), race.get(), hosp.get(), icu.get(), med.get()]])
    x_pred = np.array(ct.transform(x_pred))
    x_pred = np.array(ct1.transform(x_pred), dtype="float64")
    prob = round(classifier.predict_proba(x_pred)[0][0], 2)
    prob_text = Label(main_window, text=f"The probability of recovery for this patient is {str(prob)}", bg="light gray")
    prob_text.grid(row=3, column=1)

main_window = Tk()
main_window.geometry("700x400")
main_window.title("Predictor of Recovery")
main_window.configure(bg="light gray")

main_window.columnconfigure(0, weight=1)
main_window.columnconfigure(1, weight=2)
main_window.columnconfigure(2, weight=1)

title_frame = Frame(main_window)
title_frame.grid(row=0, column=1)

title = Label(title_frame, text="Coronavirus Recovery Predictor", bg="light gray", font="Times 30 bold")
title.grid(row=0, column=1, sticky="ew")

input_frame = Frame(main_window, bg="light gray")
input_frame.grid(row=1, column=1)


races = ["Asian, Non-Hispanic", "Black, Non-Hispanic", "White, Non-Hispanic", "Hispanic/Latino","American Indian/Alaska Native, Non-Hispanic"]
ages = ["0 - 9 Years", "10 - 19 Years", "20 - 29 Years", "30 - 39 Years", "40 - 49 Years", "50 - 59 Years", "60 - 69 Years", "70 - 79 Years", "80+ Years"]

race_factor = Label(input_frame, text="Please select the patient's race:", bg="light gray", font="Times 15")
race_factor.grid(row=0, column=0, sticky="w", pady=10)
race = StringVar()
race_options = OptionMenu(input_frame, race, *races)
race_options.config(width=20)
race_options.grid(row=0, column=1, sticky="w", columnspan=2)

sex_factor = Label(input_frame, text="Please select the patient's sex:", bg="light gray", font="Times 15")
sex_factor.grid(row=1, column=0, pady=10)
sex = StringVar()
female = Radiobutton(input_frame, text="Female", variable=sex, value="Female")
female.grid(row=1, column=1)
male = Radiobutton(input_frame, text="Male", variable=sex, value="Male")
male.grid(row=1, column=2)

age_factor = Label(input_frame, text="Please select the patient's age range:", bg="light gray", font="Times 15")
age_factor.grid(row=2, column=0, pady=10)
age = StringVar()
age_options = OptionMenu(input_frame, age, *ages)
age_options.config(width=20)
age_options.grid(row=2, column=1, sticky="w", columnspan=2)

hosp_factor = Label(input_frame, text="Should/Has the patient been hospitalized?", bg="light gray", font="Times 15")
hosp_factor.grid(row=3, column=0, pady=10)
hosp = StringVar()
yes = Radiobutton(input_frame, text="Yes", variable=hosp, value="Yes")
yes.grid(row=3, column=1)
no = Radiobutton(input_frame, text="No", variable=hosp, value="No")
no.grid(row=3, column=2)

icu_factor = Label(input_frame, text="Should/Has the patient go to the ICU?", bg="light gray", font="Times 15")
icu_factor.grid(row=4, column=0, sticky="e", pady=10)
icu = StringVar()
yes = Radiobutton(input_frame, text="Yes", variable=icu, value="Yes")
yes.grid(row=4, column=1)
no = Radiobutton(input_frame, text="No", variable=icu, value="No")
no.grid(row=4, column=2)

med_factor = Label(input_frame, text="Any preexisting medical conditions?", bg="light gray", font="Times 15")
med_factor.grid(row=5, column=0, pady=10)
med = StringVar()
yes = Radiobutton(input_frame, text="Yes", variable=med, value="Yes")
yes.grid(row=5, column=1)
no = Radiobutton(input_frame, text="No", variable=med, value="No")
no.grid(row=5, column=2)

predict_button = Button(main_window, text="Predict", command=predict_rate)
predict_button.grid(row=2, column=1)

main_window.mainloop()
