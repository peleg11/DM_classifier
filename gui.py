from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Combobox
from Project import *
import pandas as pd

'''
Shmuel Atias 300987443
Dmitry Korkin 336377429
Shay Peleg 302725643
'''

window = Tk()
window.title("window")

window.geometry('800x600')


# ///////////////////////1:choose file

def trainfile():
    file = askopenfilename()

    global train
    train = pd.read_csv(file)
    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=1)


def testfile():
    file = askopenfilename()
    global test
    test = pd.read_csv(file)
    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=2)


def structurefile():
    file = askopenfilename()
    global structure
    structure = pd.read_csv(file, sep=' ', header=None, names=['att', 'att_name', 'types'], usecols=[1, 2])
    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=3)


label1 = Label(window, text="Step1: Choose Files")
label1.grid(column=0, row=0)
label111 = Label(window, text="                                   ")
label111.grid(column=1, row=0)
label1111 = Label(window, text="                                   ")
label1111.grid(column=2, row=0)
label11111 = Label(window, text="                                   ")
label11111.grid(column=3, row=0)
label11111 = Label(window, text="                                   ")
label11111.grid(column=4, row=0)
btnTrain = Button(window, text="Choose Train file", command=trainfile)
btnTrain.grid(column=1, row=1)
btnTest = Button(window, text="Choose Test file", command=testfile)
btnTest.grid(column=1, row=2)
btnStrct = Button(window, text="Choose Structor file", command=structurefile)
btnStrct.grid(column=1, row=3)
# *****************////////////////////////


# ///////////////////////3:choose model
label2 = Label(window, text="Step2: Choose Model")
label2.grid(column=0, row=5)


def getV(root):
    global a
    a = combo.get()

    global proj
    if a == "NaiveBase":
        proj = NaiveBayesClassifier(structure, train, test)
    if a == "SklearnNaiveBase":
        proj = SklearnNB(structure, train, test)
    if a == "ID3":
        proj = ID3(structure, train, test)
    if a == "SklearnID3":
        proj = SklearnID3(structure, train, test)
    if a == "KNN":
        proj = KNNClassifier(structure, train, test)
    if a == "K-Means":
        proj = KMeansClustering(structure, train, test)

    print(a)
    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=4)


combo = Combobox(window)
combo['values'] = ("NaiveBase", "SklearnNaiveBase", "ID3", "SklearnID3", "KNN", "K-Means")
combo.current(1)  # defualt value
combo.grid(column=1, row=4)
button1 = Button(window, text=u"Accept")
button1.grid(column=2, row=4)
button1.bind("<Button-1>", getV)


# *****************////////////////////////
# ///////////////////////2:load model
def loadmodel(root):
    name = T.get()
    proj.loadModel(name)

    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=5)


label22 = Label(window, text="Step2:Load Model(if you have)")
label22.grid(column=0, row=5)

T = Entry(window)
T.grid(column=1, row=5)
button1 = Button(window, text=u"Accept")
button1.grid(column=2, row=5)
button1.bind("<Button-1>", loadmodel)


# *****************////////////////////////
# ///////////////////////4:Standartization/Normalization
def cleandata():
    proj.clean()
    cl = Label(window, text="Data is clean ", fg="black")
    cl.grid(column=2, row=6)
    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=6)


label3 = Label(window, text="Step3: Clean data")
label3.grid(column=0, row=6)

btn4 = Button(window, text="CleanData", command=cleandata)
btn4.grid(column=1, row=6)


# *****************////////////////////////
# ///////////////////////5:discritization

def discr(root):
    global b
    b = combo1.get()

    print(b)
    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=8)


label3 = Label(window, text="Step4.1:Choose kind of Discritization")
label3.grid(column=0, row=8)

combo1 = Combobox(window)
combo1['values'] = (
    "equalwidth_pandas", "equalfreaqncy_pandas", "equalwidth", "equalfreaqncy", "entropybinning_external")
combo1.current(0)
combo1.grid(column=1, row=8)
button11 = Button(window, text=u"Accept")
button11.grid(column=2, row=8)
button11.bind("<Button-1>", discr)


def choosebins(root):
    c = combo2.get()
    print(c)
    proj.discretization((int)(c), b)

    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=9)


label3 = Label(window, text="Step4.2: Choose num of bins")
label3.grid(column=0, row=9)

combo2 = Combobox(window)
combo2['values'] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
combo2.current(1)
combo2.grid(column=1, row=9)
button2 = Button(window, text=u"Accept")
button2.grid(column=2, row=9)
button2.bind("<Button-1>", choosebins)


# *****************////////////////////////
# ///////////////////////6:discritization

def std_nrml(root):
    d = combo3.get()
    print(d)
    if d == "None":
        pass
    if d == "standartization":
        proj.standardization()
    if d == "normalization":
        proj.normalization()

    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=7)


label4 = Label(window, text="Step5:Choose Standartization/Normalization")
label4.grid(column=0, row=7)

combo3 = Combobox(window)
combo3['values'] = ("None", "standartization", "normalization")
combo3.current(1)
combo3.grid(column=1, row=7)
button3 = Button(window, text=u"Accept")
button3.grid(column=2, row=7)
button3.bind("<Button-1>", std_nrml)


# *****************////////////////////////
# ///////////////////////7:Build Model
def build():
    proj.buildModel()
    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=10)


label5 = Label(window, text="Step5:Build")
label5.grid(column=0, row=10)
model = Button(window, text="Build", command=build)
model.grid(column=1, row=10)


# *****************////////////////////////
# ///////////////////////8: save model
def save_model(root):
    name = name_model.get()
    print(name)
    proj.saveModel(name)

    w = Label(window, text="V", font=("Helvetica", 16), fg="green")
    w.grid(column=3, row=11)


label55 = Label(window, text="Step5:Save model(if you want)")
label55.grid(column=0, row=11)
name_model = Entry(window)
name_model.grid(column=1, row=11)
button4 = Button(window, text=u"Accept")
button4.grid(column=2, row=11)
button4.bind("<Button-1>", save_model)


# *****************////////////////////////
# ///////////////////////9: result

def result():
    la = Label(window, text=proj.result())
    la.grid(row=12, column=2)


label5 = Label(window, text="Step6:Result")
label5.grid(column=0, row=12)
model = Button(window, text="Result", command=result)
model.grid(column=1, row=12)


# *****************////////////////////////
# ///////////////////////10: confusion matrix

def matrix():
    la = Label(window, text=proj.matrix())
    la.grid(row=13, column=2)


# w = Label(window, text="V", font=("Helvetica", 16), fg="green")
# w.grid(column=3, row=6)


label5 = Label(window, text="Confusion Matrix")
label5.grid(column=0, row=13)
model = Button(window, text="Get Matrix", command=matrix)
model.grid(column=1, row=13)


# *****************////////////////////////
# ///////////////////////10: report


def report():
    la = Label(window, text=proj.report())
    la.grid(row=14, column=2)


# w = Label(window, text="V", font=("Helvetica", 16), fg="green")
# w.grid(column=3, row=6)


label5 = Label(window, text="Classification report")
label5.grid(column=0, row=14)
model = Button(window, text="Get report", command=report)
model.grid(column=1, row=14)

window.mainloop()
'''''
#Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
#filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file


'''
