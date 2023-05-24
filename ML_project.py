from tkinter import *
window=Tk()
window.title('Window 1')
window.geometry('600x500')
window.configure(bg='green')
root=Tk()
root.title('Window 2')
root.geometry('600x500')
root.configure(bg='green')

from tkinter import filedialog
from sklearn.impute import SimpleImputer
import pandas as pd 


def UploadAction(event=None):
    global data
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    import pandas as pd 
    data=pd.read_csv(filename)
def preprocessing():
    global data1

    import pandas as pd 
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    data.iloc[:,12]=le.fit_transform(data.iloc[:,12])
    lbl4['text']=data.head() 

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    pd.DataFrame(scaler.fit_transform(data))
    lbl5['text']=(pd.DataFrame(scaler.fit_transform(data))).head()

def regression():
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    X = data[['parks']]
    y = data['price']
    X.head()
    y.head()
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)
    print(X_test.shape)
    print(y_test.shape)
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    predictedprice = regr.predict(X_test)
    print(predictedprice )
    from matplotlib import pyplot as plt 
    plt.scatter(X_test, y_test,color='g') 
    plt.plot(X_test,predictedprice)

    plt.show()
    from sklearn.metrics import mean_squared_error
    mse=mean_squared_error(y_test, predictedprice)
    lbl8['text']=mse
    predictedprice2 = regr.predict([[900]])
    print(predictedprice2)
def UploadAction2(event=None):
    global data2
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    import pandas as pd 
    data2=pd.read_csv(filename)
def most_frequant():
    global data2
    from sklearn.impute import SimpleImputer
    imputer=SimpleImputer(strategy='most_frequent')#mean or median or most frequant
    data2.iloc[:,0:12]=imputer.fit_transform(data2.iloc[:,0:12])
    lbl23['text']=data2.head()
def min_max_scaler():
    global data2
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    data2=pd.DataFrame(scaler.fit_transform(data2))
    lbl23['text']=data2.head()
def knn():
    import pandas as pd
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split


    x= data2.iloc[:, :-1].values  
  
    y= data2.iloc[:, -1].values  
    print(x.shape)
    print(y.shape)
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3) 
    from sklearn.neighbors import KNeighborsClassifier
    model =KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train,y_train)
    predictions =model.predict(x_test)
    from sklearn.metrics import confusion_matrix
    matrix=confusion_matrix(y_test,predictions)
    lbl21['text']=matrix
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_test,predictions)
    lbl10['text']=acc
    from sklearn.metrics import precision_score
    pre=precision_score(y_test,predictions)
    lbl14['text']=pre

    from sklearn.metrics import recall_score
    rec=recall_score(y_test,predictions)
    lbl12['text']=rec
    from sklearn.metrics import f1_score
    f1s=f1_score(y_test,predictions)
    lbl18['text']=f1s
def svm():
    import pandas as pd
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    x= data2.iloc[:, :-1].values  
    from sklearn import svm

    y= data2.iloc[:, -1].values 
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3) 
    from sklearn.ensemble import RandomForestClassifier
    classifier=svm.SVC(kernel='linear')
    classifier.fit(x_train,y_train)
    predictions3=classifier.predict(x_test)
    from sklearn.metrics import confusion_matrix
    matrix=confusion_matrix(y_test,predictions3)
    lbl21['text']=matrix
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_test,predictions3)
    lbl10['text']=acc
    from sklearn.metrics import precision_score
    pre=precision_score(y_test,predictions3)
    lbl14['text']=pre

    from sklearn.metrics import recall_score
    rec=recall_score(y_test,predictions3)
    lbl12['text']=rec
    from sklearn.metrics import f1_score
    f1s=f1_score(y_test,predictions3)
    lbl18['text']=f1s
    
#------------------------------------------------------------------------------------------------------


frame1=Frame(window,width=100,bg='#808080')   
frame1.grid(row=30,column=30,padx=30,pady=30,ipadx=30,ipady=30)
frame1.configure(width='100')
btn1=Button(frame1,text='تحميل',command=UploadAction,font=(25),fg='black')
btn1.grid(row=0,column=3)
btn1=Radiobutton(frame1,command=preprocessing,text='Preprocessing',font=(25),bg='black',fg='blue')
btn1.grid(row=0,column=0)
btn2=Radiobutton(frame1,command=regression,text='Regression',font=(25),bg='black',fg='blue')
btn2.grid(row=0,column=1)
lbl3=Label(frame1,text='Preprocessing OutPut',font=(17),fg='blue',bg='#808080')
lbl3.grid(row=4,column=0,padx=10,pady=10)
lbl4=Label(frame1,text='label encoder',font=(17),fg='blue',bg='#808080')
lbl4.grid(row=4,column=1,padx=10,pady=10)
lbl5=Label(frame1,text='Normalization',font=(17),fg='blue',bg='#93c572')
lbl5.grid(row=4,column=2,padx=10,pady=10)
lbl6=Label(frame1,text='Regression Output',font=(17),fg='blue',bg='#808080')
lbl6.grid(row=5,column=0,padx=10,pady=10)
lbl8=Label(frame1,text='MSE',font=(20),fg='blue',bg='#808080')
lbl8.grid(row=5,column=1,padx=10,pady=10)

#------------------------------------------------------------------------------------------------------
frame2=Frame(root,width=100,bg='#93c572')   
frame2.grid(row=0,column=0,padx=20,pady=20,ipadx=20,ipady=20)
frame2.configure(width='100')

lbl2=Label(frame2,text='Classification',font=(25),fg='blue',bg='#808080')
lbl2.grid(row=0,column=0)
btn6=Button(frame2,command=UploadAction2,text='تحميل',font=(25),fg='black')
btn6.grid(row=0,column=2)
btn7=Label(frame2,text='preprocessing',font=(25),fg='blue',bg='#455a64')
btn7.grid(row=0,column=1)
lbl15=Radiobutton(frame2,command=most_frequant,text='most frequant',font=(25),bg='black',fg='blue')
lbl15.grid(row=2,column=0)
lbl16=Radiobutton(frame2,command=min_max_scaler,text='min-max-scaler',font=(25),bg='black',fg='blue')
lbl16.grid(row=2,column=2)

#------------------------------------------------------------------------------------------------------


btn5=Radiobutton(frame2,command=knn,text='KNN',font=(25),bg='black',fg='blue')
btn5.grid(row=1,column=0)
btn7=Radiobutton(frame2,command=svm,text='SVM',font=(25),bg='black',fg='blue')
btn7.grid(row=1,column=1)
frame3=Frame(root,width=100)   
frame3.grid(row=0,column=1,padx=20,pady=20,ipadx=20,ipady=20)
frame3.configure(width='100')
lbl22=Label(frame3,text='Classification Output',font=(17),fg='black')
lbl22.grid(row=0,column=0,padx=10,pady=10)
lbl20=Label(frame3,text='confusion_matrix',width='15',font=(17),fg='black')
lbl20.grid(row=1,column=0,padx=10,pady=10)
lbl21=Label(frame3,text='',width='25',font=(17),bg='blue',fg='black')
lbl21.grid(row=1,column=1,padx=10,pady=10)
lbl9=Label(frame3,text='Accuracy',font=(17),fg='black')
lbl9.grid(row=2,column=0,padx=10,pady=10)

lbl10=Label(frame3,width='25',text='',font=(17),bg='blue',fg='black')
lbl10.grid(row=2,column=1,padx=10,pady=10)
lbl11=Label(frame3,text='Recall',font=(17),fg='black')
lbl11.grid(row=3,column=0,padx=10,pady=10)
lbl12=Label(frame3,width='25',text='',font=(17),bg='blue',fg='black')
lbl12.grid(row=3,column=1,padx=10,pady=10)
lbl13=Label(frame3,text='Percision',font=(17),fg='black')
lbl13.grid(row=4,column=0,padx=10,pady=10)
lbl14=Label(frame3,width='25',text='',font=(17),bg='blue',fg='black')
lbl14.grid(row=4,column=1,padx=10,pady=10)
lbl17=Label(frame3,width='20',text='F1-score',font=(17),fg='black')
lbl17.grid(row=5,column=0,padx=10,pady=10)
lbl18=Label(frame3,width='25',text='',font=(17),bg='blue',fg='black')
lbl18.grid(row=5,column=1,padx=10,pady=10)

window.mainloop()
