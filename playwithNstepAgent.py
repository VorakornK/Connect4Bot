from module import *
from IPython import display
import os

env = ConnectFourEnvironment()

RandAgent = RandomAgent()

done = False
Agent = 1

while not done:
    action = RandAgent.select_action(env) if env.current_player == Agent else int(input("Enter column: "))
    next_state, reward, done = env.step(action)
    os.system("clear")
    env.display_board()
    