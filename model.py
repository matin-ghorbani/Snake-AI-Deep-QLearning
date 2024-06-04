import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)

        self.models_folder_path = 'model'
        os.makedirs(self.models_folder_path, exist_ok=True)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        return self.lin2(x)

    def save(self, file_name='snake_model.pth'):
        file_name = os.path.join(self.models_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)

        # Q_new = reward + gamma * max(next_pred Q value) -> Only do this if not done
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(pred[idx])

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()
        self.optimizer.step()
