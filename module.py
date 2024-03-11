import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import numpy as np
import random



class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(6 * 7, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        x = x.view(-1, 6 * 7)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConnectFourEnvironment:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1

    def get_invalid_actions(self):
        return np.array([0 if self.board[0, col] == 0 else 1 for col in range(self.cols)])

    def is_valid_action(self, action):
        if action < 0 or action >= self.cols:
            return False
        return self.get_invalid_actions()[action] == 0

    def make_move(self, board, action, player=None):
        if player is None:
            player = self.current_player
        for row in range(self.rows - 1, -1, -1):
            if board[row, action] == 0:
                board[row, action] = player
                break

    def get_state(self):
        return self.board.copy()

    def check_line(self, line):
        for player in [1, 2]:
            for i in range(len(line) - 3):
                if np.all(line[i:i + 4] == player):
                    return player
        return 0

    def get_winner(self, board=None):
        if board is None:
            board = self.board
        # Check for a winner in rows, columns, and diagonals
        for i in range(self.rows):
            row_result = self.check_line(board[i, :])
            if row_result:
                return row_result

        for j in range(self.cols):
            col_result = self.check_line(board[:, j])
            if col_result:
                return col_result

        for i in range(self.rows - 3):
            for j in range(self.cols - 3):
                diag_result = self.check_line(board[i:i + 4, j:j + 4].diagonal())
                if diag_result:
                    return diag_result

                rev_diag_result = self.check_line(np.fliplr(board[i:i + 4, j:j + 4]).diagonal())
                if rev_diag_result:
                    return rev_diag_result

        return 0  # No winner yet
    
    def find_opponent_win(self, player):
        invalid_moves = self.get_invalid_actions()
        for action, invalid in enumerate(invalid_moves):
            if invalid == 1:
                continue
            temp_board = self.board.copy()
            self.make_move(temp_board, action, player)
            if self.get_winner(temp_board) == player:
                return action
        return -1
            
    
    def step(self, action):
        if not self.is_valid_action(action):
            print("Invalid action:", action)
            raise ValueError("Invalid action. Please choose a valid action.")
        
        opponent_win = self.find_opponent_win(3 - self.current_player)

        self.make_move(self.board, action)
        state = self.get_state()
        
        winner = self.get_winner()
        done = winner != 0 or not any(self.board[0, :] == 0)  # Check for a winner or a full board
        
        reward = 0.0
    
        if opponent_win == action:
            reward += 1.0 # Prevent opponent from winning
        if opponent_win != -1 and opponent_win != action:
            reward += -100.0
        elif winner == 0:
            reward += -2.0  # It's a tie
        elif winner == self.current_player:
            reward += 2.0
                
        self.current_player = 3 - self.current_player
        
        return state, reward, done
    
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
    def __init__(self, state_size, action_size, gamma=0.99):
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
        valid_actions = env.get_invalid_actions()
        # print(valid_actions)
        if random.random() < epsilon:
            idx = np.nonzero(valid_actions == 0)[0]
            return torch.tensor(random.choice(idx), dtype=torch.long).to(self.device)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state).cpu().numpy()
                q_values[0][valid_actions == 1] = float('-inf')

                # print(q_values)
                return torch.tensor(np.argmax(q_values), dtype=torch.long).to(self.device)


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
        expected_q_values = reward_batch + 0.99 * next_q_values

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