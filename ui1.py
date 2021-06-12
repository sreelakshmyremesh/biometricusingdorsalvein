# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:08:57 2021

@author: USER
"""

from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog

#master = Tk()

root=Tk()
root.title("Dorsal Vein")
root.geometry("1920x1080")
root.iconbitmap(r"C:\GUI\images")

 
l1 = Label(root, text = "ACQUIRED IMAGE:").place(x = 450, y = 100)

def open():
    global my_image
    root.filename = filedialog.askopenfilename(initialdir="/GUI/images/img3",title="select image",filetypes=(("png files","*.png"),("jpg files","*.jpg"),("all files","*.*")))
    my_label=Label(root,text=root.filename).pack()
    my_image=ImageTk.PhotoImage(Image.open(root.filename))
    my_image_label=Label(image=my_image).pack()

my_btn=Button(root,text="Browse",command=open,bg='red',fg="black" ,width=20 ,height=3)
my_btn.pack()
my_btn.place(x=50, y=100)






head=Label(root,
           fg='white',
          bg='black',
          width = 100,
          height = 3 ,
          font=("Arial Bold", 10),
          text="Dorsal Vein")

head.pack()


    
button = Button(text="PREPROCESS",  width=20,  height=3,fg="black",bg='red')
button.place(x=50, y=300)





root.mainloop()