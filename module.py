import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import numpy as np
import random
from minimax import *


class DQN(nn.Module):
    def __init__(self, input_size = 42, num_actions = 7):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x
    
# class DQN(nn.Module):
#     def __init__(self, input_dim=(1, 6, 7), output_dim=7):
#         super(DQN, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=input_dim[0], out_channels=128, kernel_size=4, padding=0)
#         self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=0)
        
#         # Fully connected layers
#         # Adjusted input size based on the output shape of conv2
#         self.fc1 = nn.Linear(2 * 3, 128)  # Flattened size from conv2
#         self.fc2 = nn.Linear(128, 128)
        
#         # Output heads
#         self.head1 = nn.Linear(128, output_dim)

#     def forward(self, x):
#         # Convolutional layers
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
        
#         # Flatten
#         x = x.view(x.size(0), -1)  # Flatten the output
#         # x = x.view(x.size(0), -1)  # Flatten the output
        
#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        
#         # Output heads
#         output1 = self.head1(x)
#         return output1


class ConnectFourEnvironment:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.reset()
        
        self.winning_lines = []
        for i in range(self.rows):
            for j in range(self.cols - 3):
                self.winning_lines.append(([i for _ in range(4)], [j + k for k in range(4)]))
        for i in range(self.rows - 3):
            for j in range(self.cols):
                self.winning_lines.append(([i + k for k in range(4)], [j for _ in range(4)]))
        for i in range(self.rows - 3):
            for j in range(self.cols - 3):
                self.winning_lines.append(([i, i + 1, i + 2, i + 3], [j, j + 1, j + 2, j + 3]))
        for i in range(3, self.rows):
            for j in range(self.cols - 3):
                self.winning_lines.append(([i, i - 1, i - 2, i - 3], [j, j + 1, j + 2, j + 3]))

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1

    def get_invalid_actions(self):
        return np.array([0 if self.board[0, col] == 0 else 1 for col in range(self.cols)])

    def is_valid_action(self, action):
        return 0 <= action < self.cols and self.board[0, action] == 0

    def make_move(self, board, action, player=None):
        if player is None:
            player = self.current_player
            
        row = np.argmax(board[::-1, action] == 0)
        row = self.rows - 1 - row
        board[row, action] = player

    def get_state(self):
        return self.board.reshape(1, self.rows, self.cols)

    def get_winner(self, board=None):
        if board is None:
            board = self.board
        for line in self.winning_lines:
            line_values = board[line]
            if np.all(line_values == 1):
                return 1
            elif np.all(line_values == 2):
                return 2
        return 0
    
    def evaluate(self, window, player, board=None):
        if board is None:
            board = self.board
        
        score = 0
        opponent = 3 - player
        
        player_count = np.count_nonzero(window == player)
        opponent_count = np.count_nonzero(window == opponent)
        empty_count = np.count_nonzero(window == 0)
        
        if player_count == 4:
            score += 100
        elif player_count == 3 and empty_count == 1:
            score += 5
        elif player_count == 2 and empty_count == 2:
            score += 2
            
        if opponent_count == 3 and empty_count == 1:
            score -= 5
          
        return score
    
    def get_score(self, player, board=None):
        if board is None:
            board = self.board
        score = 0
        
        center = [int(i) for i in list(board[:, self.cols // 2])]
        center_count = center.count(player)
        score += center_count * 3
        
        for line in self.winning_lines:
            line_values = board[line]
            score += self.evaluate(line_values, self.current_player)
        return score
    
    def get_done(self):
        return self.get_winner() != 0 or not any(self.board[0, :] == 0)
    
    def find_winning_move(self, player):
        invalid_moves = self.get_invalid_actions()
        for action, invalid in enumerate(invalid_moves):
            if not invalid:
                temp_board = self.board.copy()
                self.make_move(temp_board, action, player)
                if self.get_winner(temp_board) == player:
                    return action
        return -1
            
    
    def step(self, action):
        if not self.is_valid_action(action):
            raise ValueError("Invalid action. Please choose a valid action.")
        
        opponent = 3 - self.current_player
        
        opponent_winning_move = self.find_winning_move(3 - self.current_player)
        before_action_score = self.get_score(self.current_player)

        self.make_move(self.board, action)
        
        after_action_score = self.get_score(self.current_player)
        
        winner = self.get_winner()
        done = self.get_done()
        
        reward = 0.0
    
        if opponent_winning_move == action:
            reward += 10.0 # Prevent opponent from winning
        if opponent_winning_move != -1 and opponent_winning_move != action:
            reward += -100.0 # Not Prevent opponent from winning
        elif winner == 0 and done:
            reward +=  20.0  # It's a tie
        elif winner == self.current_player:
            reward += 100.0
            
        reward += after_action_score - before_action_score # Reward for making a move
                
        self.current_player = 3 - self.current_player
        
        return self.get_state(), reward, done
    
    def display_board(self, board=None):
        if board is None:
            board = self.board
        # Display column indices
        print(" ", end=" ")
        for col in range(self.cols):
            print(col, end=" ")
        print("\n+---------------+")

        # Display the board
        for row in range(self.rows):
            print("|", end=" ")
            for col in range(self.cols):
                print(board[row, col], end=" ")
            print("|")

        print("+---------------+")
        

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.epsilon = epsilon
        # self.gamma = gamma
        # self.gamma = torch.tensor(gamma, device=self.device)

        # Define the Q-networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Define the optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

        # Experience Replay Memory
        self.memory = []

    def select_action(self, state, env, epsilon):
        invalid_actions = env.get_invalid_actions()
        opponent = 3 - env.current_player
        winning_move = env.find_winning_move(opponent)
        if winning_move != -1:
            return torch.tensor(winning_move, dtype=torch.long).to(self.device)
        if random.random() < epsilon:
            # idx = np.nonzero(invalid_actions == 0)[0]
            # return torch.tensor(random.choice(idx), dtype=torch.long).to(self.device)
            return torch.tensor(OneStepAgent.select_action(env), dtype=torch.long).to(self.device)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state)
                q_values[0, invalid_actions == 1] = -np.inf
                return torch.argmax(q_values).item()


    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # expected_q_values = reward_batch + self.gamma * next_q_values
        expected_q_values = reward_batch + 0.95 * next_q_values

        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def select_action(self, env):
        valid_actions = env.get_invalid_actions()
        idx = np.nonzero(valid_actions == 0)[0]
        return random.choice(idx)
    
class NStep_Agent:
    def __init__(self, n=1):
        self.n = n

    def select_action(self, env):
        col, _ = minimax(np.flip(env.board, 0), self.n, -np.inf, np.inf, True, env.current_player)
        if env.is_valid_action(col):
            return col
        return random.choice(np.nonzero(env.get_invalid_actions() == 0)[0])

class VerticalLineAgent:
    pass

OneStepAgent = NStep_Agent()
