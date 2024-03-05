from module import *
import torch
import os
import time


def RandomVsAgent(agent, random_agent, env, AI_number):
    env.reset()
    done = False

    while not done:
        state = env.get_state()

        if env.current_player == AI_number:
            action = agent.select_action(state, env, False)
        else:
            action = random_agent.select_action(env)

        next_state, _, done = env.step(action)
        
    winner = env.get_winner()
    
    return winner == AI_number

    if winner == 0:
        print("It's a tie!")
    elif winner == AI_number:
        print("AI wins!")
    else:
        print("Human wins!")

# Create ConnectFourEnvironment and DQNAgent
env = ConnectFourEnvironment()
agent = DQNAgent(state_size=(6, 7), action_size=7)  # Assuming state size is the shape of the board
random_agent = RandomAgent(action_size=7)
# Load your trained model state
agent.policy_net.load_state_dict(torch.load("20000.pth"))
# agent.policy_net.eval()

# Play a game against the loaded model

AI_win_first = 0
AI_win_second = 0
for i in range(10000):
    if i % 2 == 0:
        AI_win_first += RandomVsAgent(agent, random_agent, env, 1)
    else:
        AI_win_second += RandomVsAgent(agent, random_agent, env, 2)
        
print(AI_win_first, AI_win_second)

