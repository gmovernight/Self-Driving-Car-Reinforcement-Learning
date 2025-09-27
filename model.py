import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Simple DQN network
class Network(nn.Module):
    def __init__(self, input_size: int, nb_action: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)  # Q-values


class Replay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []

    def push(self, event_tuple):
        # event_tuple: (state, next_state, action, reward)
        self.memory.append(event_tuple)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        states, next_states, actions, rewards = zip(*batch)
        return (
            torch.cat(states, 0),
            torch.cat(next_states, 0),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
        )


class DQN:
    def __init__(self, input_size: int, nb_action: int, gamma: float):
        self.gamma = gamma
        self.model = Network(input_size, nb_action)
        self.memory = Replay(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.last_state = torch.zeros(1, input_size, dtype=torch.float32)
        self.last_action = 0
        self.last_reward = 0.0
        self.reward_window = []

    @torch.no_grad()
    def select_action(self, state, temperature: float = 3.0):
        # lower temperature => less random thrashing
        q = self.model(state)
        probs = F.softmax(q / max(1e-6, temperature), dim=1)
        action = torch.multinomial(probs, num_samples=1)
        return int(action[0, 0])

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        # Q(s,a)
        q_pred = self.model(batch_state).gather(1, batch_action.view(-1, 1)).squeeze(1)
        # max_a' Q(s', a')
        with torch.no_grad():
            q_next_max = self.model(batch_next_state).max(1)[0]
            q_target = batch_reward + self.gamma * q_next_max
        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, reward, signal):
        new_state = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        self.memory.push((self.last_state, new_state, self.last_action, reward))

        action = self.select_action(new_state)  # temp set in select_action default

        if len(self.memory.memory) > 100:
            bs, bns, ba, br = self.memory.sample(100)
            self.learn(bs, bns, ba, br)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            self.reward_window.pop(0)

        return action

    def save(self, path: str = "last_brain.pth"):
        torch.save(
            {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()},
            path,
        )

    def load(self, path: str = "last_brain.pth"):
        if os.path.isfile(path):
            print("Loading existing file")
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded!")
        else:
            print("No saved file found!")

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)
