#import streamlit as st
import pickle
from collections import deque
import os
import gym
import cv2

import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image

from AgentClass import Agent

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

#import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
Hyperparameters
"""
ENVIRONMENT = "PongDeterministic-v4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directory = os.getcwd()
MEMORY_PATH= directory+"\\pong-cnn-memory.pkl"
RENDERS_PATH= directory+"\\pong-cnn-renders.pkl"
MODEL_PATH = directory+"\\pong-cnn-"
image_folder=directory+"\\inputs"

LOAD_MODEL_FROM_FILE = True  # Load model from file
LOAD_FILE_EPISODE = 1350  # Load Xth episode from file

weightsPath = MODEL_PATH + str(LOAD_FILE_EPISODE) + '.pkl'
epsilonPath = MODEL_PATH + str(LOAD_FILE_EPISODE) + '.json'

BATCH_SIZE = 32 #64 # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode

MAX_MEMORY_LEN = 30000#50000  # Max memory len
MIN_MEMORY_LEN = 20000 #40000 # Min memory len before start train

GAMMA = 0.97  # Discount rate
ALPHA = 0.00025  # Learning rate
EPSILON_DECAY = 0.99  # Epsilon decay rate by step

RENDER_GAME_WINDOW = False  # Opens a new window to render the game (Won't work on colab default)

"""
helper functions
"""
class Explainability_Board():
  def __init__(self):
    self.environment = gym.make(ENVIRONMENT,  render_mode='rgb_array')  
    #create an instance of agent class
    self.xAgent = Agent(self.environment) 
    #load pretrained weights if we already trined
    if LOAD_MODEL_FROM_FILE:
        self.xAgent.online_model.load_state_dict(torch.load(weightsPath))

        with open(epsilonPath) as outfile:
            param = json.load(outfile)
            self.xAgent.epsilon = param.get('epsilon')

    self.xexp_renders= deque(maxlen=MAX_MEMORY_LEN)
    self.episodes= dict()
    self.load_agent_memory()
    self.xexp_renders= self.load_renders()
    self.create_episodes()

        
  #load the agent memory to process it from the stored pkl file before
  def load_agent_memory(self,path=MEMORY_PATH):
    # from collections import deque
    Load_memory=path
    # Open the .pkl file and load its contents
    with open(Load_memory, 'rb') as file:
        data = pickle.load(file)

    # Create a deque object and pass the loaded contents
    self.xAgent.memory = deque(data)

    # Close the file
    file.close()
  
  #load the renders from pikle file

  def load_renders(self, path=RENDERS_PATH):
      
    # open the pickle file in binary mode
    with open(path, 'rb') as f:
        # use pickle.load() to deserialize and load the dictionary from the file
        my_dict = pickle.load(f)

    # print the dictionary
    return my_dict

#get saliency:
  def getSaliecny(self, state):
    sali_model= self.xAgent.online_model
    state = torch.tensor(state, dtype=torch.float, device=DEVICE)
    state= torch.autograd.Variable(state, requires_grad=True)
    q_values = sali_model(state)
    action = torch.argmax(q_values).item()
    q_values.sum().backward()
    heatmap=state.grad.cpu().numpy()

    return heatmap, q_values, action
  
  #this function is to get the ball position in a frame
  def remove_unnecessary_info(self, frame): 
      #store ball positions 
      ballPosition= []
      #remove unnecessary information 
      for i in range(self.xAgent.target_w):# the y's 
          for j in range(self.xAgent.target_h):#the x's
              if(frame[i][j] not in [0.25098039215686274,0.2549019607843137] and i not in [58, 59,60,61,62,63, 4,3,2,1,0] ):
                  frame[i][j]=0.0
                  if (j not in [56,57,58,59,60] and j not in [1,2,3,4,5,6,7]): #our paddle 56,57 // opponent paddle 6,7
                      ballPosition.append((j,i)) #(x,y)

              elif (frame[i][j] in [0.25098039215686274,0.2549019607843137] ):
                  frame[i][j]==1.0
              else:
                  frame[i][j]==1.0
        
      if len(ballPosition)== 0:
          ballPosition.append((0,0))
                      
      return frame, ballPosition
  
  #state, action, reward, next_state, done =agent.memory[22]
  def get_balls_coordinates_for_state(self, state): 
      #get ball coordinates
      frames={}
      coordeinates= {}
      for i in range(4): #i=0 is the last frame that was added
        #remove unnecessary information 
        frames["state_"+str(i)], coordeinates["state_"+str(i)]= self.remove_unnecessary_info(state[0][i])
      return frames, coordeinates

  def reverse_get_coordinates(self, x, y):
    #get the corrdianates of the renders
    # State size for Pong environment is (210, 160, 3). 
    state_size_h = self.xAgent.state_size_h
    state_size_w = self.xAgent.state_size_w


    # Image pre process params
    target_h = self.xAgent.target_h  # Height after process
    target_w = self.xAgent.target_w  # Widht after process

    crop_dim = self.xAgent.crop_dim # Cut 20 px from top to get rid of the score table

    ######
    j= x# the x's
    i= y# the y's

    #the y's
    reversed_i_resize= (i * (state_size_h-crop_dim[0]))/target_h
    reversed_i= reversed_i_resize +crop_dim[0]
    # the x's
    reversed_j_resize= (j * state_size_w)/target_w
    reversed_j= reversed_j_resize+ crop_dim[2]

    return (int(reversed_j),int(reversed_i))#(x,y)

  #print circle and arrow function 
  def users_expectation(self, row):
    #get the ball position in the state
    state, action, reward, next_state, done, _ =self.xAgent.memory[row]
    gray=state[0][0]
    #get the frame that we want to draw ontop of
    target= self.xexp_renders[row]

    ball_centers= []
    #frames if we want to show the user's expectations on top of the input
    frames, coordeinates= self.get_balls_coordinates_for_state(state)
    for i in range(4):
        #get ball centers for the satet's frames
        ball_centers.append((coordeinates["state_"+str(i)][0]))
        
    x1=ball_centers[-1][0]
    y1=ball_centers[-1][1]

    x2=ball_centers[0][0]
    y2=ball_centers[0][1]

    #const value
    x_exp= 56

    #there is no ball
    #if the ball has just appeared or just disappeared 
    #check if the coordinates are zeroes then don't draw
    if((x1 == 0 and y1== 0) or (x2== 0 and y2== 0) ):
      output= " there is no ball yet, because a new episode has just begun"
      return target, output

    #there is a ball and we an draw
    else:
      #if the ball is moving in a stright line the slop contains deviding on zero
      if (x2-x1)== 0:
        y_exp= x2
        output= " ball moving in a stright line"
      #there is a ball and we can draw the arrow and the circle
      else:
        slop= ((y2-y1)/(x2-x1))
        y_exp= int((slop* (x_exp- x1)+ y1))
        output= " "
      
      #get correct cordinates to draw on the renders
      #draw the arrows
      start_point = self.reverse_get_coordinates(x1,y1)
      end_point = self.reverse_get_coordinates(x2,y2)
      color = (0, 255, 0) # Green color
      thickness = 1

      target=cv2.arrowedLine(target, start_point, end_point,color, thickness)

      #draw the circle
      #get the Center coordinates
      center_coordinates = self.reverse_get_coordinates(x_exp, y_exp) #(x,y)

        
      #if the center of the circle is outside the frame boarders then we will wait till it bounces
      if center_coordinates[1] <= self.xAgent.state_size_h:
        output+= "let's draw the red circle is the expected place for the paddel"
        # print("draw")
        # Radius of circle
        radius = 3
          
        # Blue color in BGR
        color = (255, 0, 0)
          
        # Line thickness of 2 px
        thickness = 1
          
        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        target=cv2.circle(target, center_coordinates, radius, color, thickness)

      else:
        output+= " we will wait until the ball bounces off the boarders"

      return target, output

#we want to show a video of the agent palying next to it
  def create_episodes(self):
    i= 0
    templist= []
    total_reward=0

    for row in range(len(self.xAgent.memory)):
      state, action, reward, next_state, done, q_values_FM =self.xAgent.memory[row]  
      total_reward += reward
      #get the heatmaps to append them to the episode
      heatmap, q_values, action_FHM= self.getSaliecny(state)
      #the function plot_saliency_map(state[0][0], heatmap[0][0]) will only plot it returns void

      #get user expectation to add them also to the mix
      target, output= self.users_expectation(row)
      templist.append({"memory_row":self.xAgent.memory[row],"render": self.xexp_renders[row],"heatmap": heatmap,"q_values": q_values,"action":action_FHM, "AC_pic":target, "output":output})

      #if the episode is over move to a different
      if done == True:
        #print(i)
        self.episodes[i]=[templist, total_reward]
        templist= []
        total_reward=0
        i+=1
