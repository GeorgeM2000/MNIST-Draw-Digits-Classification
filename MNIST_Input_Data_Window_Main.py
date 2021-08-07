import pygame
from MNIST_Input_Data_Functionality import Grid
from enum import Enum, auto
import tensorflow as tf
import tkinter as tk
import os

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (400,100) 

surface = pygame.display.set_mode((800, 560)) 
pygame.display.set_caption('MNIST Classifier')

buttonChoice = 0
isAnyOperationValid = True


class States(Enum):
    running = auto()


def text_objects(text, font):
    textSurface = font.render(text, True, (0,0,0))
    return textSurface, textSurface.get_rect()

# Window to show the predicted result
def open_window(text):
    window = tk.Tk()        # Creating a tkinter window
    window.title('Result')
    window.geometry('300x200')      # Initialize tkinter window with dimensions 300 x 200           

    # Creating a Label  
    tk.Label(window, text= 'I predicted ' + text, font= ('Helvetica 18')).place(relx=0.5, rely=0.5, anchor= tk.CENTER)

    # Creating a Button
    tk.Button(window, text= "Close", background= "white", foreground= "red", font= ('Helvetica 13 bold'), command= window.destroy).pack(side=tk.BOTTOM, pady=30)
    window.mainloop()

# Configuration for the buttons
def actionButton(x,y,w,h,msg,Ri,Gi,Bi,Ra,Ga,Ba,actionSelected):
    mouse = pygame.mouse.get_pos()
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(surface, (Ra, Ga, Ba), (x,y,w,h))
        if pygame.mouse.get_pressed()[0]:
            global buttonChoice
            global isAnyOperationValid
            if actionSelected == "Draw":
                buttonChoice = 1
                return
            elif actionSelected == "Predict" and isAnyOperationValid == True:
                buttonChoice = 4
                return
            elif actionSelected == "TryAgain":
                buttonChoice = 2
                return
            elif actionSelected == "Quit":
                buttonChoice = 3
                return
    else:
        pygame.draw.rect(surface, (Ri, Gi, Bi), (x,y,w,h))

    smallText = pygame.font.Font("freesansbold.ttf",20)
    textSurf, textRect = text_objects(msg,smallText)
    textRect.center = (int(x+(w/2)), int(y+(h/2)))
    surface.blit(textSurf, textRect)



state = States.running
grid = Grid()
grid.draw(surface)
runningState = True

# Class names
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Choose the model file
tflite_model_file = 'model1.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output details
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']



while runningState:

    # Button creation
    actionButton(570,20,60,35,"Draw",192,192,192,160,160,160,"Draw")
    actionButton(650,20,105,35,"Predict",102,102,255,51,51,255,"Predict")
    actionButton(570,100,115,40,"Try again",192,192,192,160,160,160,"TryAgain")
    actionButton(700,100,60,40,"Quit",192,192,192,160,160,160,"Quit")

    
    
    if buttonChoice == 1 and isAnyOperationValid:
        if pygame.mouse.get_pressed()[0]:
            mousePosition = pygame.mouse.get_pos()
            if mousePosition[0]//20 >= 0 and mousePosition[0]//20 < 28*20 and mousePosition[1]//20 >= 0 and mousePosition[1]//20 < 28*20:
                grid.drawCell(mousePosition[0]//20, mousePosition[1]//20, surface, "Draw")
        
    for event in pygame.event.get():
        
        if event.type == pygame.MOUSEBUTTONDOWN and state == States.running:
            if pygame.mouse.get_pressed()[0]:
                
                if buttonChoice == 2:
                    isAnyOperationValid = True
                    buttonChoice = 0
                    grid.reload(surface)

                elif buttonChoice == 3:
                    runningState = False
                
                elif buttonChoice == 4 and isAnyOperationValid:
                    isAnyOperationValid = False
                    result = grid.predict(surface, class_names, interpreter, input_index, output_index)
                    open_window(result)
                    
        elif event.type == pygame.QUIT:
            runningState = False

    pygame.display.flip()
