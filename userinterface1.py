# -*- coding: utf-8 -*-
"""
Created on Sun May 23 19:55:19 2021

@author: USER
"""

from tkinter import *
from tkinter import filedialog

#master = Tk()

root = Tk()

root.title("dorsal vein")
root.geometry("1920x1080")



head=Label(root,
           fg='white',
          bg='black',
          width = 100,
          height = 3 ,
          font=("Arial Bold", 10),
          text="Dorsal Vein")

head.pack()


def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)

button = Button(root, text='BROWSE', command=UploadAction
                ,bg='red',fg="black" ,width=20 ,height=3)

button.pack()
button.place(x=50, y=100)

button = Button(
    text="CLEAR",
    width=20,
    height=3,
    bg='red',
    fg="black",
)
button.place(x=50, y=200)


button = Button(text="PREPROCESS", width=20,  height=3,fg="black",bg='red')
button.place(x=50, y=300)

l1 = Label(root, text = "ACQUIRED IMAGE:").place(x = 450, y = 100)
                                                 
canvas = Canvas(root, width = 300, height = 300)      
canvas.pack()      
img = PhotoImage(file="img3.png")      
canvas.create_image(20,20, anchor=NW, image=img)

l2 = Label(root, text = "KNUCKLE TIP EXTRACTION:").place(x = 390, y = 400)
                                                 
canvas1 = Canvas(root, width = 300, height = 300)      
canvas1.pack()      
img2 = PhotoImage(file="label.png")      
canvas1.create_image(20,20, anchor=NW, image=img2)

l3 = Label(root, text = "REGION OF INTEREST:").place(x = 420, y = 700)
                                                 
canvas1 = Canvas(root, width = 300, height = 300)      
canvas1.pack()      
img3 = PhotoImage(file="roi.png")      
canvas1.create_image(20,20, anchor=NW, image=img3)



root.mainloop()

