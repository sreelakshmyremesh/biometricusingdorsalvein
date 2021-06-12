# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:32:01 2021

@author: USER
"""
from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog


root=Tk()
root.title("image")
root.iconbitmap(r"C:\GUI\images")

 
def open():
    global my_image
    root.filename = filedialog.askopenfilename(initialdir="/GUI/images",title="select image",filetypes=(("png files","*.png"),("jpg files","*.jpg"),("all files","*.*")))
    my_label=Label(root,text=root.filename).pack()
    my_image=ImageTk.PhotoImage(Image.open(root.filename))
    my_image_label=Label(image=my_image).pack()

my_btn=Button(root,text="Browse",command=open,bg='white',fg="black" ,width=10 ,height=1).pack()




button = Button(
    text="SPELLCHECK",
    width=20,
    height=2,
    bg='white',
    fg="black")
button.place(x=100, y=200)




button = Button(
    text="SPEECH",
    width=20,
    height=2,
    bg='white',
    fg="black")
button.place(x=100, y=300)

root.mainloop()
