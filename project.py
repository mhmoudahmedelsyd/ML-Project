import math
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd

from tkinter import *
from tkinter.ttk import *
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVC
from sklearn.decomposition import PCA
# #from imblearn.over_sampling import SMOTE
from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier



file_path = None
df = None



class MyFrame:
    def __init__(self, master):
        self.master = master
        root.pack_propagate(False)
        self.master.title("My Frame")
        self.master.geometry("800x670")
        self.master.resizable(0, 0)
        self.frame = tk.Frame(self.master, bg="grey", width=400, height=670)
        self.frame.pack(anchor=tk.NW)
        self.brrowseButton = tk.Button(self.frame, text="Browse DataSet", bg="red", fg="white",font=("Helvetica", 10, "bold"), command=self.browse_csv)
        self.brrowseButton.place(x=100, y=10, width=200, height=30)

        # treeframe
        self.treefram = tk.LabelFrame(self.frame, text="Data Set is ")
        self.treefram.place(x=0, y=50, height=300, width=400)
        ##treeview
        self.treeview = ttk.Treeview(self.treefram)
        self.treeview.place(relheight=1, relwidth=1)
        self.treeview.column("#0", width=50, minwidth=0, stretch=tk.NO)

        # treescroll
        self.treescrolly = tk.Scrollbar(self.treefram, orient="vertical", command=self.treeview.yview)
        self.treescrollx = tk.Scrollbar(self.treefram, orient="horizontal", command=self.treeview.xview)
        self.treeview.configure(xscrollcommand=self.treescrollx.set, yscrollcommand=self.treescrolly.set)
        self.treescrollx.pack(side="bottom", fill="x")
        self.treescrolly.pack(side="right", fill="y")
        self.labeldescription = tk.Label(self.frame, text="Data Description is")
        self.labeldescription.place(x=150, y=355, width=100, height=30)

        # textframe
        self.textfram = tk.LabelFrame(self.frame, text="Describe")
        self.textfram.place(x=0, y=390, height=280, width=400)
        self.textdescription = tk.Text(self.textfram)
        self.textdescription.place(relheight=1, relwidth=1)
        self.textscrolly = tk.Scrollbar(self.textfram, orient="vertical", command=self.textdescription.yview)
        self.textscrollx = tk.Scrollbar(self.textfram, orient="horizontal", command=self.textdescription.xview)
        self.textdescription.configure(xscrollcommand=self.textscrollx.set, yscrollcommand=self.textscrolly.set)
        self.textscrollx.pack(side="bottom", fill="x")
        self.textscrolly.pack(side="right", fill="y")

        # rightframe
        self.rightframe = tk.Frame(self.master, bg="bisque", width=400, height=670)
        self.rightframe.place(x=400, y=0)
        # simple imputer

        self.treefram2 = tk.LabelFrame(self.rightframe, text="Data Set is ")
        self.treefram2.place(x=0, y=0, height=270, width=400)
        ##treeview
        self.treeview2 = ttk.Treeview(self.treefram2)
        self.treeview2.place(relheight=1, relwidth=1)
        self.treeview2.column("#0", width=50, minwidth=0, stretch=tk.NO)
        # treescroll
        self.treescrollyy = tk.Scrollbar(self.treefram2, orient="vertical", command=self.treeview2.yview)
        self.treescrollxx = tk.Scrollbar(self.treefram2, orient="horizontal", command=self.treeview2.xview)
        self.treeview.configure(xscrollcommand=self.treescrollxx.set, yscrollcommand=self.treescrollyy.set)
        self.treescrollxx.pack(side="bottom", fill="x")
        self.treescrollyy.pack(side="right", fill="y")

        self.label = tk.Label(self.rightframe, text="Preprocessing", bg="white", fg="red", font=("Helvetica", 15, "bold"))
        self.label.place(x=100, y=271, width=200, height=20)

        self.rangeslider = tk.Scale(self.rightframe, from_=0, to=100, orient=tk.HORIZONTAL, fg="black",highlightthickness=0, )
        self.rangeslider.place(x=70, y=293, width=250, height=40)

        self.imputer_button = tk.Button(self.rightframe, text="Simple Imputer", bg="red", fg="white", font=("Helvetica", 10, "bold"),command=self.imputer)
        self.imputer_button.place(x=2, y=335, width=100, height=30)
       
        # label encoder
        self.label_encoder_button = tk.Button(self.rightframe, text="Label Encoder", bg="red", fg="white", font=("Helvetica", 10, "bold"),command=self.label_encoder)
        self.label_encoder_button.place(x=105, y=335, width=100, height=30)

        self.split_button = tk.Button(self.rightframe, text="Split Data",bg="red", fg="white", font=("Helvetica", 10, "bold"),command=self.split_data)
        self.split_button.place(x=210, y=335, width=75, height=30)
        # stander scaler
        self.scaler_button = tk.Button(self.rightframe, text="Stander Scaler",bg="red", fg="white", font=("Helvetica", 10, "bold"),command=self.scaler)
        self.scaler_button.place(x=290, y=335, width=100, height=30)

        
        self.label = tk.Label(self.rightframe, text="LinearRegression", bg="white", fg="red", font=("Helvetica", 15, "bold"))
        self.label.place(x=100, y=375, width=200, height=20)
        # LinearRegression
        self.Linear_Regression_button = tk.Button(self.rightframe, text="Linear Regression",bg="red", fg="white", font=("Helvetica", 10, "bold"),command=self.Linear_Regression)
        self.Linear_Regression_button.place(x=100, y=400, width=200, height=30)
          
        self.labelmse = tk.Label(self.rightframe, text="MSE", bg="white", fg="red", font=("Helvetica", 12, "bold"))
        self.labelmse.place(x=42, y=435, width=40, height=20)

        self.labelmseR = tk.Label(self.rightframe, text="NaN", bg="white", fg="red", font=("Helvetica", 10, "italic"))
        self.labelmseR.place(x=87, y=435, width=40, height=20)

        self.labelmae = tk.Label(self.rightframe, text="MAE", bg="white", fg="red", font=("Helvetica", 12, "bold"))
        self.labelmae.place(x=134, y=435, width=40, height=20)

        self.labelmaeR = tk.Label(self.rightframe, text="NaN", bg="white", fg="red", font=("Helvetica", 10, "italic"))
        self.labelmaeR.place(x=179, y=435, width=40, height=20)
        
        self.labelrmse = tk.Label(self.rightframe, text="RMSE", bg="white", fg="red", font=("Helvetica", 12, "bold"))
        self.labelrmse.place(x=226, y=435, width=50, height=20)
        
        self.labelrmseR = tk.Label(self.rightframe, text="NaN", bg="white", fg="red", font=("Helvetica", 10, "italic"))
        self.labelrmseR.place(x=283, y=435, width=40, height=20)
        

        # PCA
        self.feature_selection = tk.Label(self.rightframe, text="Feature Selection", bg="white", fg="red", font=("Helvetica", 15, "bold"))
        self.feature_selection.place(x=100, y=460, width=200, height=20)
        self.pca_button = tk.Button(self.rightframe, text="PCA",bg="red", fg="white", font=("Helvetica", 10, "bold"),command=self.pca)
        self.pca_button.place(x=150, y=490, width=200, height=30)
        self.labelpca = tk.Label(self.rightframe, text="N PCA", bg="white", fg="red", font=("Helvetica", 12, "bold"))
        self.labelpca.place(x=32, y=490, width=55, height=30)
        self.pcan= tk.Entry(self.rightframe, bg="white", fg="red", font=("Helvetica", 12, "bold"))
        self.pcan.place(x=90, y=490, width=55, height=30)

      



         # SVM
        self.Classification = tk.Label(self.rightframe, text="Classification", bg="white", fg="red", font=("Helvetica", 15, "bold"))
        self.Classification.place(x=100, y=530, width=200, height=20)
        self.svm_button = tk.Button(self.rightframe, text="SVM",bg="red", fg="white", font=("Helvetica", 10, "bold"),command=self.svm)
        self.svm_button.place(x=100, y=560, width=200, height=30)
        # # KNN
        self.knn_button = tk.Button(self.rightframe, text="KNN",bg="red", fg="white", font=("Helvetica", 10, "bold"),command=self.knn)
        self.knn_button.place(x=100, y=635 ,width=200, height=30)
        self.labelknn=tk.Label(self.rightframe, text="N_Neighbors", bg="white", fg="red", font=("Helvetica", 12, "bold"))
        self.labelknn.place(x=100, y=600, width=100, height=30)
        self.nknn=tk.Entry(self.rightframe, bg="white", fg="red", font=("Helvetica", 12, "bold"))
        self.nknn.place(x=210, y=600 ,width=50, height=30)



     


        #text view
        # self.text_view = tk.Text(self.rightframe)
        # self.text_view.pack(pady=10)


        # # textframe
        # self.textfram = tk.LabelFrame(self.rightframe, text="Describe")
        # self.textfram.place(x=0, y=390, height=240, width=400)
        # self.textdescription = tk.Text(self.textfram)
        # self.textdescription.place(relheight=1, relwidth=1)
        # self.textscrolly = tk.Scrollbar(self.textfram, orient="vertical", command=self.textdescription.yview)
        # self.textscrollx = tk.Scrollbar(self.textfram, orient="horizontal", command=self.textdescription.xview)
        # self.textdescription.configure(xscrollcommand=self.textscrollx.set, yscrollcommand=self.textscrolly.set)
        # self.textscrollx.pack(side="bottom", fill="x")
        # self.textscrolly.pack(side="right", fill="y")




    # CODE

    def browse_csv(self):
        # open a file dialog to browse for a CSV file
        global file_path
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            # clear the existing table
            self.treeview.delete(*self.treeview.get_children())
            self.textdescription.delete("1.0", "end")
            global df
            df = pd.read_csv(file_path)
            num_rows, num_cols = df.shape
            object_cols = df.select_dtypes(include=['object'])
            numeric_cols = df.select_dtypes(include=['int64', 'float64'])
            output_str = ''
            output_num = ''
            output_null = df.isnull().any().any()
            # numeric_cols
            for col in numeric_cols:
                num_unique = df[col].nunique()
                output_num += f"{col} has {num_unique} number_col values\n"
                # category
            for col in object_cols:
                cate_unique = df[col].nunique()
                output_str += f"{col} has {cate_unique} categories_col values\n"
            description = f"Dataset has {num_rows} rows and {num_cols} columns.\n{output_str}{output_num}IsNullValues:{output_null} "
            self.textdescription.insert(tk.END, description)

        self.treeview["columns"] = list(df.columns)
        self.treeview.heading("#0", text="Index")
        for col in df.columns:
            self.treeview.heading(col, text=col)
        index = 0
        for _, row in df.iterrows():
            self.treeview.insert(parent="", index="end", iid=index, text=index, values=list(row))
            index += 1


    def imputer(self):
        #check if there is null values
            self.treeview2.delete(*self.treeview2.get_children())
            null_cols = df.columns[df.isnull().any()]
            for col in null_cols:
                if df[col].isnull().any():
                 imputer = SimpleImputer(strategy='most_frequent')
                 imputer.fit(df[[col]])
                 df[col] = imputer.transform(df[[col]])
            self.treeview2["columns"] = list(df.columns)
            self.treeview2.heading("#0", text="Index")
            print(df)
            for col in df.columns:
                self.treeview2.heading(col, text=col)
                index = 0
            for _, row in df.iterrows():
                self.treeview2.insert(parent="", index="end", iid=index, text=index, values=list(row))
                index += 1
       


    
            # print(df)
    def label_encoder(self):
        self.treeview2.delete(*self.treeview2.get_children())
        null_cols = df.columns[df.isnull().any()]
        for col in null_cols:
            if df[col].isnull().any():
                imputer = SimpleImputer(strategy='most_frequent')
                imputer.fit(df[[col]])
                df[col] = imputer.transform(df[[col]])
            ##check if there is string values
        le = LabelEncoder()
        columns = df.select_dtypes(['object']).columns
        for col in columns:
            df[col] = le.fit_transform(df[col])
        self.treeview2["columns"] = list(df.columns)
        self.treeview2.heading("#0", text="Index")
        print(df)
        for col in df.columns:
            self.treeview2.heading(col, text=col)
            index = 0
        for _, row in df.iterrows():
            self.treeview2.insert(parent="", index="end", iid=index, text=index, values=list(row))
            index += 1
        print(df)



    def split_data(self):
         #split data 
         global featrures
         global  traget 
         featrures= pd.DataFrame(df.iloc[:, :-1].values)
         traget= df.iloc[:, -1].values
         global x_train, x_test, y_train, y_test
         x_train, x_test, y_train, y_test= train_test_split(featrures, traget, test_size= self.rangeslider.get()/100, random_state=0)
         

         self.treeview2["columns"] = list(featrures.columns)
         self.treeview2.heading("#0", text="Index")
         print(featrures)
         for col in featrures.columns:
            self.treeview2.heading(col, text=col)
            index = 0
         for _, row in featrures.iterrows():
            self.treeview2.insert(parent="", index="end", iid=index, text=index, values=list(row))
            index += 1
         



    def scaler(self):
     
         self.treeview2.delete(*self.treeview2.get_children())
         #split data
         numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
         featrures= df.iloc[:, :-1].values
         print(featrures)
         scaler=StandardScaler(copy=True,with_mean=True,with_std=True)
         featrures=pd.DataFrame(scaler.fit_transform(featrures))
         print(featrures)

         self.treeview2["columns"] = list(featrures.columns)
         self.treeview2.heading("#0", text="Index")
        #  print(df)
         for col in featrures.columns:
            self.treeview2.heading(col, text=col)
            index = 0
         for _, row in featrures.iterrows():
            self.treeview2.insert(parent="", index="end", iid=index, text=index, values=list(row))
            index += 1
        #  print(df)


    def Linear_Regression(self):    
         model = LinearRegression()
         model.fit(x_train, y_train)
         y_pred = model.predict(x_test)

         self.labelmaeR.config(text="%.2f" % mean_absolute_error(y_test, y_pred))
         self.labelmseR.config(text="%.2f" % mean_squared_error(y_test, y_pred))
         self.labelrmseR.config(text="%.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
         print(y_pred)
         print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
         print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
         print("ROOT Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
     
         




    def pca(self):
         self.treeview2.delete(*self.treeview2.get_children())
         
         #split data   
         
        #pca
         pca = PCA(n_components=int(self.pcan.get()))
         principal_components = pd.DataFrame(pca.fit_transform(featrures))
         print(principal_components)

         self.treeview2["columns"] = list(principal_components.columns)
         self.treeview2.heading("#0", text="Index")
        #  print(df)
         for col in principal_components.columns:
            self.treeview2.heading(col, text=col)
            index = 0
         for _, row in principal_components.iterrows():
            self.treeview2.insert(parent="", index="end", iid=index, text=index, values=list(row))
            index += 1
         

    def svm(self):
    



         print(y_train,)
         print(x_train,)
        # #SVM
        #  classifier=svm.SVC(kernel="rbf")
        #  classifier.fit(x_train,y_train)
        #  prediction=classifier.predict(x_test)
        #  matrix=confusion_matrix(y_test,prediction)
        #  print(matrix)
        #  acc=accuracy_score(y_test,prediction)
        #  print("Accurency",acc)
        #  rec=recall_score(y_test,prediction)
        #  print("reccal",rec)
        #  f1=f1_score(y_test,prediction)
        #  print("f1: ",f1)
        ##
     
         
        #  clf = DecisionTreeClassifier()
        #  clf = svm.SVC(kernel="rbf")
        #  clf.fit(x_train, y_train)
        #  prediction = clf.predict(x_test)
         
        #  cm = confusion_matrix(y_test, y_pred)
        #  print(cm)


         classifier=svm.SVC(kernel="poly")
         classifier.fit(x_train,y_train)
         prediction=classifier.predict(x_test)
         print(prediction)
         print(y_test)
         matrix=confusion_matrix(y_test,prediction)
         print(matrix)
         #accuracy score
         acc=accuracy_score(y_test,prediction)
         print("accuracy",acc)
         #prediction
         pre=precision_score(y_test,prediction)
         print("prediction",pre)
         #recall
         rec=recall_score(y_test,prediction)
         print("recall",rec)
         #f1 score
         f1=f1_score(y_test,prediction)
         print("f1 score",f1)

         newWindow = Toplevel(root)
         newWindow.title("New Window")
         newWindow.resizable(0, 0)
         newWindow.geometry("300x250")
         Label(newWindow,text ="Support Vector Machine",background="white",foreground="red",font=("Helvetica", 12, "bold")).pack()
         Label(newWindow,text ="Accuracy:",background="white",foreground="black",font=("Helvetica", 12, "bold")).place(x=10,y=50)
         Label(newWindow,text ="Prediction:",background="white",foreground="black",font=("Helvetica", 12, "bold")).place(x=10,y=100)
         Label(newWindow,text ="Recall:",background="white",foreground="black",font=("Helvetica", 12, "bold")).place(x=10,y=150)
         Label(newWindow,text ="F1:",background="white",foreground="black",font=("Helvetica", 12, "bold")).place(x=10,y=200)
         Accuracy=tk.Label(newWindow,text ="NAN",background="white",foreground="black",font=("Helvetica", 12, "bold"))
         Accuracy.place(x=100,y=50)
         Accuracy.config(text="%.5f" % acc)
         Prediction=Label(newWindow,text ="NAN",background="white",foreground="black",font=("Helvetica", 12, "bold"))
         Prediction.place(x=100,y=100)
         Prediction.config(text="%.5f" % pre)
         Recall=Label(newWindow,text ="NAN",background="white",foreground="black",font=("Helvetica", 12, "bold"))
         Recall.place(x=100,y=150)
         Recall.config(text="%.5f" % rec)
         F1=Label(newWindow,text ="NAN",background="white",foreground="black",font=("Helvetica", 12, "bold"))
         F1.place(x=100,y=200)
         F1.config(text="%.5f" % f1)
         






        
      


    def knn(self):
         self.treeview2.delete(*self.treeview2.get_children())



         model2=KNeighborsClassifier(n_neighbors=int(self.nknn.get()))
         model2.fit(x_train,y_train)
         prediction2=model2.predict(x_test)
         matrix2=confusion_matrix(y_test,prediction2)
         print(matrix2)
         #accuracy score
         acc2=accuracy_score(y_test,prediction2)
         print("accuracy",acc2)
         #prediction
         pre2=precision_score(y_test,prediction2)
         print("prediction",pre2)
         #recall
         rec2=recall_score(y_test,prediction2)
         print("recall",rec2)
         #f1 score
         f2=f1_score(y_test,prediction2)
         print("f1 score",f2)
     
              
         newWindow2 = Toplevel(root)
         newWindow2.title("New Window")
         newWindow2.resizable(0, 0)
         newWindow2.geometry("300x300")
         Label(newWindow2,text ="K-Nearest Neighbor",background="white",foreground="red",font=("Helvetica", 12, "bold")).pack()
         Label(newWindow2,text ="Accuracy:",background="white",foreground="black",font=("Helvetica", 12, "bold")).place(x=10,y=50)
         Label(newWindow2,text ="Prediction:",background="white",foreground="black",font=("Helvetica", 12, "bold")).place(x=10,y=100)
         Label(newWindow2,text ="Recall:",background="white",foreground="black",font=("Helvetica", 12, "bold")).place(x=10,y=150)
         Label(newWindow2,text ="F1:",background="white",foreground="black",font=("Helvetica", 12, "bold")).place(x=10,y=200)

         Accuracy=tk.Label(newWindow2,text ="NAN",background="white",foreground="black",font=("Helvetica", 12, "bold"))
         Accuracy.place(x=100,y=50)
         Accuracy.config(text="%.5f" % acc2)
         Prediction=Label(newWindow2,text ="NAN",background="white",foreground="black",font=("Helvetica", 12, "bold"))
         Prediction.place(x=100,y=100)
         Prediction.config(text="%.5f" % pre2)
         Recall=Label(newWindow2,text ="NAN",background="white",foreground="black",font=("Helvetica", 12, "bold"))
         Recall.place(x=100,y=150)
         Recall.config(text="%.5f" % rec2)
         F1=Label(newWindow2,text ="NAN",background="white",foreground="black",font=("Helvetica", 12, "bold"))
         F1.place(x=100,y=200)
         F1.config(text="%.5f" % f2)


        











root = tk.Tk()
# create an instance of MyFrame and run the GUI
app = MyFrame(root)
root.mainloop()