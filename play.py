from module import *
import torch
import os
import time

AI_number = 2

def play_game_with_model(agent, env):
    env.reset()
    done = False

    while not done:
        # Display the current board
        os.system("clear")
        env.display_board()
        # time.sleep(0.5)
        
        # Get the current state
        state = env.get_state()

        if env.current_player == AI_number:
            # Let the agent choose an action
            action = agent.select_action(state, env)
        else:
            # For human player (player 2), let them input the move
            try:
                # action = agent.select_action(state, env)
                # input("AI chooses column " + str(action.item()) + ". Press Enter to continue...")
                action = int(input("Enter your move (column index): "))
            except ValueError:
                print("Invalid input. Please enter a valid column index.")
                continue

        # Take a step in the environment
        next_state, _, done = env.step(action)
        
    os.system("clear")
    # Display the final board
    env.display_board()

# Create ConnectFourEnvironment and DQNAgent
env = ConnectFourEnvironment()
agent = DQNAgent(state_size=(6, 7), action_size=7)  # Assuming state size is the shape of the board

# Load your trained model state
agent.policy_net.load_state_dict(torch.load("1000.pth"))
# agent.policy_net.eval()

# Play a game against the loaded model
play_game_with_model(agent, env)

# Determine the winner
winner = env.get_winner()

if winner == 0:
    print("It's a tie!")
elif winner == AI_number:
    print("AI wins!")
else:
    print("Human wins!")