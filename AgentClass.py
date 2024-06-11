"""
import needed libraries
"""
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

import pickle

from collections import deque
import os
import matplotlib.pyplot as plt

#%matplotlib inline
"""
hyperparameters
"""
ENVIRONMENT = "PongDeterministic-v4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODELS = True  # Save models to file so you can test later
MODEL_PATH = "C:\\Users\\User\\Downloads\\Pong_anwar\\pong-cnn-"#"/content/pong-cnn-"  # Models path for saving or loading
DRIVE_PATH= "/content/gdrive/MyDrive/Pong/"#Model path for saving on drive
#MODEL_PATH= "/kaggle/working/pong-cnn-"

SAVE_MODEL_INTERVAL = 10  # Save models at every X epoch
TRAIN_MODEL = False  # Train model while playing (Make it False when testing a model)
#/content/pong-cnn-580.pkl
LOAD_MODEL_FROM_FILE = True  # Load model from file
LOAD_FILE_EPISODE = 1350  # Load Xth episode from file

LOAD_FROM_PATH="C:\\Users\\User\\Downloads\\Pong_anwar\\pong-cnn-1350.pkl"#"/content/pong-cnn-1300.pkl"# "/kaggle/input/model-850/pong-cnn-850.pkl" #"/kaggle/input/model-1300/pong-cnn-1300.pkl" #"
Load_epsilon= "C:\\Users\\User\\Downloads\\Pong_anwar\\pong-cnn-1300.json"#"/content/pong-cnn-1300.json"#"/kaggle/input/model-300/pong-cnn-300.json" #

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
agent classes
"""

class Agent:
        def __init__(self, environment):
            """
            Hyperparameters definition for Agent
            """
            # State size for Pong environment is (210, 160, 3). 
            self.state_size_h = environment.observation_space.shape[0]
            self.state_size_w = environment.observation_space.shape[1]
            self.state_size_c = environment.observation_space.shape[2]

            # actions size for Pong environment is 6
            self.action_size = environment.action_space.n

            # Image pre process params
            self.target_h = 64  # Height after process
            self.target_w = 64  # Widht after process

            self.crop_dim = [20, self.state_size_h, 0, self.state_size_w]  # Cut 20 px from top to get rid of the score table

            # Trust rate to our experiences
            self.gamma = GAMMA  # Discount factor for future predictions
            self.alpha = ALPHA  # Learning Rate

            # After many experinces epsilon will be 0.05
            # So we will do less Explore more Exploit
            self.epsilon = 1  # Explore or Exploit
            self.epsilon_decay = EPSILON_DECAY  # Adaptive Epsilon Decay Rate
            self.epsilon_minimum = 0.05  # Minimum for Explore

            # Deque to stor experience replay .
            self.memory = deque(maxlen=MAX_MEMORY_LEN)
            self.episode= []

            # initialize the two model for DDQN algorithm online model, target model
            self.online_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
            self.target_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
            self.target_model.load_state_dict(self.online_model.state_dict())
            #we put target model in evaluation mode because we don't want it to train 
            self.target_model.eval()

            # Adam used as optimizer
            self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)
        
        #Process image crop resize, grayscale and normalize the images
        def preProcess(self, image):
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
            frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
            frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
            frame = frame.reshape(self.target_w, self.target_h) / 255  # Normalize

            return frame

        #epsilon greedy algorithm to explor and exploit
        def act(self, state):
            act_protocol = 'Explore' if random.uniform(0, 1) <= self.epsilon else 'Exploit'

            if act_protocol == 'Explore':
                action = random.randrange(self.action_size)
                q_values=[0,0,0,0,0,0]
            else:
                with torch.no_grad():
                    state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                    q_values = self.online_model.forward(state)  # (1, action_size)
                    action = torch.argmax(q_values).item()  # Returns the indices of the maximum value of all elements

            return action, q_values
        #experience replay to train the model
        def train(self):
            if len(self.memory) < MIN_MEMORY_LEN:
                loss, max_q = [0, 0]
                return loss, max_q
            # sample a minibatch from the memory
            state, action, reward, next_state, done, _ = zip(*random.sample(self.memory, BATCH_SIZE))

            # Concat batches in one array
            # (np.arr, np.arr) ==> np.BIGarr
            state = np.concatenate(state)
            next_state = np.concatenate(next_state)

            # Convert them to tensors
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
            action = torch.tensor(action, dtype=torch.long, device=DEVICE)
            reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
            done = torch.tensor(done, dtype=torch.float, device=DEVICE)

            # Make predictions
            state_q_values = self.online_model(state)
            next_states_q_values = self.online_model(next_state)
            next_states_target_q_values = self.target_model(next_state)

            # Find selected action's q_value
            selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            # Get indice of the max value of next_states_q_values
            # Use that indice to get a q_value from next_states_target_q_values
            # We use greedy for policy So it called off-policy
            next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
            # Use Bellman function to find expected q value
            expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

            # Calc loss with expected_q_value and q_value
            loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()
            
            del state, next_state, action, reward, done, next_states_q_values, next_states_target_q_values,selected_q_value,expected_q_value
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss, torch.max(state_q_values).item()

        #add new experince to the replay memory 
        def storeResultsToEpisode(self, state, action, reward, nextState, done, q_values):
            self.episode.append([state[None, :], action, reward, nextState[None, :], done, q_values])
            #self.memory.append([state[None, :], action, reward, nextState[None, :], done, q_values])

        def storeResultsToMemory(self, total_reward, target_reward):
            #save the last frame of the episode
            #self.episode.append([state[None, :], action, reward, nextState[None, :], done, q_values])
            #THIS PART IS FOR THE TRAINING 
            """if len(self.memory) < MIN_MEMORY_LEN:
                store= True
            elif len(self.memory) >= MIN_MEMORY_LEN and total_reward>= target_reward:
                store= True
            else:
                store= False"""
            #THIS PART IS FOR THE TESTING
            
            if total_reward >= 17:
                store= True
            else:
                store= False
            
            if store:
                for i in range(len(self.episode)):
                    state, action, reward, nextState, done, q_values= self.episode[i]
                    self.memory.append([state, action, reward, nextState, done, q_values])
                del self.episode
                self.episode=[]
            return store

        #decay epsilon at every step to allow our model to exploit more as it trains
        def adaptiveEpsilon(self):
            if self.epsilon > self.epsilon_minimum:
                self.epsilon *= self.epsilon_decay
        

        
#CNN this will be the structure for both the online and target model.
class DuelCNN(nn.Module):
    def __init__(self, h, w, output_size):
        super(DuelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # Action layer
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node
        
    #calculate the Convelotional layers output size
    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer

        q = Vx + (Ax - Ax.mean())

        return q
    
    
