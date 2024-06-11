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
from AgentClass import Agent

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
The training loop
"""
def train():
        
    # create the environment 
    environment = gym.make(ENVIRONMENT,  render_mode='rgb_array')  
    #create an instance of agent class
    agent = Agent(environment)  
    done=False

    store= False

    #save frames 
    # renders= []
    Episode_renders=[]
    exp_renders=deque(maxlen=MAX_MEMORY_LEN)
    rewards=[]

    #load pretrained weights if we already trined
    if LOAD_MODEL_FROM_FILE:
        agent.online_model.load_state_dict(torch.load(LOAD_FROM_PATH))

        with open(Load_epsilon) as outfile:
            param = json.load(outfile)
            agent.epsilon = param.get('epsilon')
            

        startEpisode = LOAD_FILE_EPISODE + 1
    #start with eps=1 and random weights because we didn't train yet
    else:
        startEpisode = 1

    #data structure deque to store the last 100 episode rewards
    last_100_ep_reward = deque(maxlen=100)  
    total_step = 1  
    for episode in range(startEpisode, MAX_EPISODE):

        startTime = time.time()  # store the time
        # Reset env at the beginning of each episode
        state = environment.reset()  
        #get the observations from the environment
        state= environment.render()

        #process the observations to get suitable images that can be fed to the CNNs
        state = agent.preProcess(state)  


        #stack observations to create the state "4 consecutive observations makes a state"
        state = np.stack((state, state, state, state))

        total_max_q_val = 0  # Total max q vals
        total_reward = 0  # Total reward for each episode
        total_loss = 0  # Total loss for each episode
        for step in range(MAX_STEP):

            #if we want to show how is the model playing we can render the opervations 
            #this doesn't work probably yet since we are working with colab and we dont have a monitor to show the observations
            if RENDER_GAME_WINDOW:
                Episode_renders.append(environment.render())


            # use epsilon greedy to select an action to be perforemed on the environemt
            action, q_values = agent.act(state)  
            #get the next observation, reward, and done to show of we reached a terminal state.
            next_state, reward, done,_,  info = environment.step(action) 


            #process the new observations to create the next state
            next_state = agent.preProcess(next_state)  

            #add the new observation to the already defined state to create the new state
            next_state = np.stack((next_state, state[0], state[1], state[2]))

            # Store the transition in memory
            agent.storeResultsToEpisode(state, action, reward, next_state, done, q_values)  # Store to mem

            # Update state 
            state = next_state  
        

            if TRAIN_MODEL and store:
                # Perform one step of the optimization (on the target network)
                #using the experience replay algorithm
                loss, max_q_val = agent.train()  
            else:
                loss, max_q_val = [0, 0]

            total_loss += loss
            total_max_q_val += max_q_val
            total_reward += reward
            total_step += 1


            if total_step % 1000 == 0:
                # decay epsilon each 1000 steps
                agent.adaptiveEpsilon()  

            # we compleated an episode 
            if done:  
                '''
                rewards.append(total_reward)
                max_reward= max(rewards)
                target_reward= max([max_reward-4, -21])
                
                #for priotized memory
                saved=agent.storeResultsToMemory( total_reward, target_reward)
                print(saved, total_reward, target_reward)
                if saved and RENDER_GAME_WINDOW:
                    
                    for i in range(len(Episode_renders)):
                        exp_renders.append(Episode_renders[i])
                    Episode_renders=[]
                    save_agent_memory(agent.memory, MODEL_PATH+'memory.pkl')
                    save_agent_memory(exp_renders, MODEL_PATH+'renders.pkl')
                '''
                
                #store the finish time
                currentTime = time.time()  
                # get the episode duration
                time_passed = currentTime - startTime  
                # Get current dateTime as HH:MM:SS
                current_time_format = time.strftime("%H:%M:%S", time.gmtime())  
                # Create epsilon dict to save model as file
                epsilonDict = {'epsilon': agent.epsilon}   
                
                # Save model to the over come Ram craches 
                if SAVE_MODELS and episode % SAVE_MODEL_INTERVAL == 0:  
                    weightsPath = MODEL_PATH + str(episode) + '.pkl'
                    epsilonPath = MODEL_PATH + str(episode) + '.json'
                    # memoryPath= MODEL_PATH+ "memory" + str(episode) + '.pkl'

                    torch.save(agent.online_model.state_dict(), weightsPath)
                    with open(epsilonPath, 'w') as outfile:
                        json.dump(epsilonDict, outfile)



                if TRAIN_MODEL and store:
                    # Update target model at the end of the episode target_model = online_model
                    agent.target_model.load_state_dict(agent.online_model.state_dict())  

                last_100_ep_reward.append(total_reward)
                avg_max_q_val = total_max_q_val / step

                #create output file to show the results later
                outStr = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} Epsilon:{:.2f} Duration:{:.2f} Step:{} CStep:{}".format(
                    episode, current_time_format, total_reward, total_loss, np.mean(last_100_ep_reward), avg_max_q_val, agent.epsilon, time_passed, step, total_step
                )

                print(outStr)

                #save the output file, model and epsilon value to the drive
                if SAVE_MODELS:
                    outputPath = MODEL_PATH + "out" + '.txt'  # Save outStr to file
                    with open(outputPath, 'a') as outfile:
                        outfile.write(outStr+"\n")
                    #save_weights_to_drive(weightsPath,DRIVE_PATH)
                    #save_weights_to_drive(epsilonPath,DRIVE_PATH)
                    #save_weights_to_drive(outputPath,DRIVE_PATH)
                    # save_weights_to_drive(memoryPath,DRIVE_PATH)

                break
            

"""
call the function to start training
"""
train()