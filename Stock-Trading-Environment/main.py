import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

import pandas as pd
import tkinter as tk
from tkinter import *
#CREATE THE GUI FOR THE STOCK TRADING DATA
root = tk.Tk()
root.title="Stock Trading"
root.geometry("800x800")  
#THIS IS WHERE THE LINE CHART IS DISPLAYED.
#THE HEIGHT AND WIDTH ARE GIVEN AND PLACED ONTO THE SCREEN
cnv=Canvas(root,height=300,width=500)
cnv.pack()
#THESE TWO LOOPS SET THE GRID LINES IN THE CHART
#TWENTY PIXELS APART.
#THIS LOOP DRAWS THE HORIZONTAL LINES
for i in range(0,500,20):
    cnv.create_line(i,0,i,300,width=2, fill='gray')
#THIS DRAWS THE VERTICAL LINES
for i in range(0,300,20):
    cnv.create_line(0,i,500,i,width=2, fill='gray')
#THIS INITIALIZES THE LIST OF DICTIONARY DATA FOR THE STOCK TRADING ACTIVITYU,
#ONE RECORD FOR EACH TIMESTEP
stepSummary=[]
df = pd.read_csv('./data/AAPL.csv')
df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)
stockInfo=[]
obs = env.reset()
prevX=0
prevY=0
#THIS IS WHERE THE STOCK TRADING DATA ARE EXTRACTED,
#PRINTED TO THE SCREEN AND STORED IN A DICTIONARY LIST.
for i in range(1000):   
#THE STATE ACTION PAIR THAT DECIDES ON A CHOICE BETWEEN BUYING AND SELLING,
#OR ALTERNATIVELY HOLDING
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print ("REWARDS: " + str(rewards))
#THE RENDERING IS WHERE THE RESULTING VALUES ARE EXTRACTED
    stepSummary.append(env.render())
#NOW THE GUI SCREEN DISPLAYS
root.configure(bg='light green')
#THE TEXT BOX SHOWS ALL THE TIME STEP DATA AT A GLANCE
T=tk.Text(root, height=30, width=60)
T.pack()  
fullSummary="" 
#THIS IS WHERE THE DATA ARE SHOWN IN THE TEXT BOX
#THE OUTER LOOP ITERATES THROUGH THE LIST OF DICTIONARY RECORDS
for i in range(len(stepSummary)):
#THE INNER LOOP ITERATES THROUGH THE ITEMS IN EACH RECORD
    for k, v in stepSummary[i].items():
        fullSummary += str((k,v)) #THIS CUMULATIVELY STORES THE ITEMS
    fullSummary += 2 * "\n"
#FINALLY, THIS IS WHERE THE LINE CHART IS PLOTTED.
#AT EACH POINT THE LINE IS DRAWN FOR THE PREVIOUS COORDINATES TO THE NEW ONES.
    newX=i
    newY=stepSummary[i]['Profit']
#HERE THE LINE IS DRAWN FROM ONE POINT TO THE NEXT.
#NOTE THAT THE PURPOSE OF THE CALCULATIONS IS TO SCALE THE STOCK TRADING ENVIRONMENT WITH RESPECT TO THE DIMENTIONS OF THE CANVAS.
#IN ADDITION, THE Y-VALUES IN THE CREATE_LINE COMMAND ARE CALCULATED IN ORDER TO FLIP OVER THE RANGE OF Y-VALUES
#SO THAT THE Y-VALUE IS PLOTTED STARTING FROM THE BOTTOM TO IMITATE THE CARTESIAN COORDINATE SYSTEM, WHERE THE POSITIVE
#Y-VALUES START AT THE LOWER LEFT.
    cnv.create_line( (prevX/1000) * 500, 250 - ((prevY/10000)*250), (newX/1000) * 500, 250 - ((newY/10000) * 250), width=5, fill='red')
    prevX=newX
    prevY=newY
T.insert(tk.END,  fullSummary )
root.update()
tk.mainloop()