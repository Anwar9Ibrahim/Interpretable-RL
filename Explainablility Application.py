import tkinter as Tk
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import threading
import datetime
import imutils
import cv2
import os
import matplotlib, sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pylab as plt
from scipy import ndimage
from matplotlib.figure import Figure
import torch


from explainability_board import Explainability_Board

directory = os.getcwd()
"""
root = Tk()  # create root window
root.title("Explainablility Application")  # title of the GUI window
root.maxsize(1200, 600)  # specify the max size the window can expand to
root.config(bg="skyblue")  # specify background color

# Create left and right frames
left_frame_top = Frame(root, width=560, height=400, bg='grey')
left_frame_top.grid(row=0, column=0, padx=10, pady=5)

middle_frame = Frame(root, width=650, height=400, bg='grey')
middle_frame.grid(row=0, column=1, padx=10, pady=5)

right_frame = Frame(root, width=650, height=400, bg='grey')
right_frame.grid(row=0, column=2, padx=10, pady=5)

# Create frames and labels in left_frame
Label(left_frame_top, text="Original Image").grid(row=0, column=0, padx=5, pady=5)
"""



root = Tk()
root.wm_title("Explainablility Application")
root.maxsize(2000, 1000)
root.config(bg="white")  # specify background color

# Create initial plot
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
ax.plot([1, 2, 3], [1, 2, 3])

# Create canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

info_label= Label(root, text="let's start")
info_label.pack()


i=0
def Run():
   global i, canvas
   i+=1
   memory_row(episode=0 , row=i)

def Back():
   global i, canvas
   i-=1
   memory_row(episode=0 , row=i)

# Create button
#to run frame by frame
B_next = Button(root, text ="Run", height=2, width=5, background="green", padx=5, pady=5, command = Run)
B_back = Button(root, text ="Back", height=2, width=5, background="pink", padx=5, pady=5, command = Back)
B_next.pack()
B_back.pack()

#to run untill pressing something else


#define the Explainability_Board
ex_b= Explainability_Board()
renders= ex_b.load_renders()

def plot_saliency_map(image_tensor: np.array, saliency_map: np.array) -> None:
    plt.figure(dpi=150)
    plt.imshow(image_tensor)#, cmap="gray")
    plt.imshow(
        saliency_map,
        #norm=colors.TwoSlopeNorm(vcenter=0),
        cmap="jet",
        alpha=0.5,  # make saliency map trasparent to see original picture
    )
    plt.title("saliency map")
    plt.axis("off")
    plt.show()

def get_images(ex_b,renders,episode , row):
    render_a= renders[row]
    user_expectaion_a = ex_b.xexp_renders[row]
    saliency_a= ex_b.episodes[episode][0][row]["heatmap"][0][0]
    state_a, action, reward, next_state, done, q_values_FM = ex_b.episodes[episode][0][row]["memory_row"]
    output= ex_b.episodes[episode][0][row]["output"]
    score=ex_b.episodes[episode][1]
    if type(q_values_FM)== torch.Tensor:
        q_values_FM=q_values_FM.cpu().detach().numpy()[0]

    return render_a, saliency_a, state_a,user_expectaion_a, q_values_FM, output, reward, score, action

def memory_row(episode , row):
    global canvas, info_label
    canvas.get_tk_widget().pack_forget()
    info_label.pack_forget()

    render_a, saliency_a, state_a,user_expectaion_a, q_values_FM, output,reward,score, action = get_images(ex_b,renders,episode , row)

    info_label= Label(root, text="Iteration_"+str(i)+ "  "+ output+" our agent choose the action "+str(action)+" got the reward of "+str(reward)+ " and the total score at this episode is "+str(score) ,background="white", font=("Arial", 15))
    info_label.pack()
    #show one row of memory
    fig = plt.figure(figsize=(20, 14))
    # setting values to rows and column variables
    rows = 2
    columns = 3
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 1)
    im_s=plt.imshow(saliency_a)
    plt.title("Saliency")

    fig.add_subplot(rows, columns, 2)
    im_i=plt.imshow(state_a[0][0])
    plt.title("Input")

    fig.add_subplot(rows, columns, 3)
    im_r=plt.imshow(render_a)
    plt.title("Render")

    fig.add_subplot(rows, columns, 4)
    im_u=plt.imshow(user_expectaion_a)
    plt.title("User expectaions")



    ax_= fig.add_subplot(rows, columns, 5)
    categories = ['NOOP', 'Fire', 'Right', 'Left', 'Right Fire', 'Left Fire' ]
    index= np.argmax(q_values_FM)
    barlist= ax_.bar(categories, q_values_FM)
    barlist[index].set_color('r')
    ax_.set_ylim(-2, 2)
    plt.title("Q-values")

    fig.add_subplot(rows, columns, 6)
    im_s_= plt.imshow(state_a[0][0])#, cmap="gray")
    im_s_a= plt.imshow(
        saliency_a,
        #norm=colors.TwoSlopeNorm(vcenter=0),
        cmap="jet",
        alpha=0.5,  # make saliency map trasparent to see original picture
    )
    plt.title("saliency map, on top of input")
    plt.axis("off")


    ax = plt.gca()
    ax.set_xticklabels([]) 
    ax.set_yticklabels([]) 

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    


  

#memory_row()

memory_row(episode=0 , row=i)




"""render_img =  ImageTk.PhotoImage(image=Image.fromarray(render_a))
saliency_img= ImageTk.PhotoImage(image=Image.fromarray(saliency_a, mode="L"))
state_img= ImageTk.PhotoImage(image=Image.fromarray(state_a[0][0], mode="L"))


# load image to be "edited"
Label(left_frame_top, image=render_img).grid(row=1, column=0, padx=5, pady=5)

# Display image in right_frame
Label(right_frame, image=saliency_img).grid(row=1,column=0, padx=5, pady=5)

Label(middle_frame, image=state_img).grid(row=1,column=0, padx=5, pady=5)
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
left_frame_top = Frame(root, width=560, height=400, bg='grey')
left_frame_top.grid(row=0, column=0, padx=10, pady=5)
"""

# load image to be "edited"
#Label(left_frame_top, image=render_img).grid(row=1, column=0, padx=5, pady=5)


"""
# Create tool bar frame
tool_bar = Frame(left_frame_top, width=180, height=185)
tool_bar.grid(row=2, column=0, padx=5, pady=5)
# Example labels that serve as placeholders for other widgets
Label(tool_bar, text="Tools", relief=RAISED).grid(row=0, column=0, padx=5, pady=3, ipadx=10)  # ipadx is padding inside the Label widget
Label(tool_bar, text="Filters", relief=RAISED).grid(row=0, column=1, padx=5, pady=3, ipadx=10)
"""
# Example labels that could be displayed under the "Tool" menu
"""Label(tool_bar, text="Select").grid(row=1, column=0, padx=5, pady=5)
Label(tool_bar, text="Crop").grid(row=2, column=0, padx=5, pady=5)
Label(tool_bar, text="Rotate & Flip").grid(row=3, column=0, padx=5, pady=5)
Label(tool_bar, text="Resize").grid(row=4, column=0, padx=5, pady=5)
Label(tool_bar, text="Exposure").grid(row=5, column=0, padx=5, pady=5)"""

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg






# Start main loop

root.mainloop()