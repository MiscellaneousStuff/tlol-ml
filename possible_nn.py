import torch
import torch.nn as nn
import torch.nn.functional as F

from tlol.datasets.replay_dataset import *

"""
ACTION_DIM   = 6
MOVE_X_DIM   = 9
MOVE_Y_DIM   = 9
OFFSET_X_DIM = 9
OFFSET_Y_DIM = 9
UNIT_DIM     = 100


class ProcessSet(nn.Module):
    def __init__(self, ins, outs):
        self.fc1 = nn.Linear(ins, ins // 2)
        self.fc2 = nn.Linear(ins // 2, outs)

    def forward(self, x):
        x = F.relu(self.fc1)
        x = F.relu(self.fc2)
        return x


class UnitModule(nn.Module):
    def __init__(self, champ=False):
        self.abilities = ProcessSet()
        self.champ = champ
        if champ:
            pass

    def forward(self, x):
        x = self.abilities(x)
        if self.champ:
            pass

class UnitTypeModule(nn.Module):
    def __init__(self, champ=False):
        self.unit_module = UnitModule(champ=champ)

    def forward(self, x):
        x = self.unit_module(x)
        x = nn.MaxPool1d(x.shape[1], stride=1)
        return x

class Agent(nn.Module):
    def __init__(self, model_size=1024):
        # Observations
        self.allied_champs = UnitTypeModule(champs=True)
        self.enemy_champs  = UnitTypeModule(champs=True)
        self.neutral_units = UnitTypeModule(champs=False)
        self.allied_units  = UnitTypeModule(champs=False)
        self.enemy_units   = UnitTypeModule(champs=False)

        # Decide
        self.lstm     = nn.LSTM(
            input_size=0,
            model_size=model_size,
            num_layers=1)
        self.action   = nn.Linear(model_size, ACTION_DIM)

        # Action Heads
        self.offset_x = nn.Linear(model_size, MOVE_X_DIM)
        self.offset_y = nn.Linear(model_size, MOVE_Y_DIM)
        self.move_x   = nn.Linear(model_size, OFFSET_X_DIM)
        self.move_y   = nn.Linear(model_size, OFFSET_Y_DIM)
        self.unit     = nn.Linear(model_size, UNIT_DIM)

    def forward(self, data):
        # Global
        global_ = data["obs"][0:2]

        # Observations
        allied_champs = data["champs"][0:data["champs"] // 2]
        allied_champs = self.allied_champs(allied_champs)

        enemy_champs  = data["champs"][data["champs"] // 2:]
        enemy_champs  = self.enemy_champs()
        
        allied_units  = self.neutral_units()
        enemy_units  = self.allied_units()
        neutral_units = self.enemy_units()


        # Decide
        x = torch.cat((
            allied_champs,
            enemy_champs,
            allied_units,
            enemy_champs,
            neutral_units), axis=1)
        x, _     = self.lstm(x)
        action   = self.action(x)

        # Action Heads
        move_x   = self.move_x(x)
        move_y   = self.move_y(x)
        offset_x = self.offset_x(x)
        offset_y = self.offset_y(x)
        unit     = self.unit(x)

        return action, move_x, move_y, offset_x, offset_y, unit

"""


class ProcessSet(nn.Module):
    def __init__(self, ins, outs):
        self.fc1 = nn.Linear(ins, ins // 2)
        self.fc2 = nn.Linear(ins // 2, outs)

    def forward(self, x):
        x = F.relu(self.fc1)
        x = F.relu(self.fc2)
        return x


class UnitModule(nn.Module):
    def __init__(self, ins, outs):
        super().__init__()

        self.fc1 = nn.Linear(ins, ins // 2)
        self.fc2 = nn.Linear(ins // 2, outs)

        # self.abilities = ProcessSet()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Agent(nn.Module):
    def __init__(self, example, unit_out=16, model_size=1024):
        super().__init__()

        # Observation
        self.allied_champs = UnitModule(51, unit_out)

        # Decide
        ins = 2 + unit_out # global + allied champs
        self.lstm = nn.LSTM(ins, model_size, num_layers=1, batch_first=True)

        # Action Head
        self.action = nn.Linear(model_size, 1)
        
    def forward(self, x):
        # Observation
        globals = x["raw"].iloc[:, 0:2].to_numpy()
        globals = torch.Tensor(globals).unsqueeze(0)

        # Observation - Allied Champs
        allied_champs_a = x["champs"].iloc[:, 0:14].to_numpy()
        allied_champs_b = x["champs"].iloc[:, 28:65].to_numpy()
        allied_champs   = np.concatenate((allied_champs_a, allied_champs_b), axis=1)
        allied_champs   = torch.Tensor(allied_champs).unsqueeze(0)
        allied_champs   = self.allied_champs(allied_champs)

        obs = torch.cat((globals, allied_champs), axis=2)

        # Decide
        x, _ = self.lstm(obs)

        # Action Head
        x = self.action(x)

        return x


if __name__ == "__main__":
    dataset = TLoLReplayDataset("./full_db")

    example = dataset[0]
    agent   = Agent(example)
    action  = agent(example)

    print(action)