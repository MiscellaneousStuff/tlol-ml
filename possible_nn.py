import torch
import torch.nn as nn
import torch.nn.functional as F

from tlol.datasets.replay_dataset import *


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
        self.enemy_champs  = UnitModule(51, unit_out)
        self.neutral_units = UnitModule(17, unit_out // 4)

        # Decide
        ins = 2 + unit_out*5 + unit_out*5 + (unit_out // 4)*24
        print("DECIDE INS:", ins)
        self.lstm_fc = nn.Linear(ins, ins)
        self.lstm = nn.LSTM(ins, model_size, num_layers=1, batch_first=True)

        # Action Head
        self.action = nn.Linear(model_size, 1)

    def forward(self, x):
        # Observation
        globals = x["raw"].iloc[:, 0:2].to_numpy()
        globals = torch.Tensor(globals).unsqueeze(0)

        # Observation - Allied Champs
        allied_champs_s = []
        for i in range(UNIT_COUNTS["champs"] // 2):
            off             = i * UNIT_FEATURE_COUNTS["champs"]
            allied_champs_a = x["champs"].iloc[:, 0+off:14+off].to_numpy()
            allied_champs_b = x["champs"].iloc[:, 28+off:65+off].to_numpy()
            allied_champs   = np.concatenate((allied_champs_a, allied_champs_b), axis=1)
            allied_champs   = torch.Tensor(allied_champs).unsqueeze(0)
            allied_champs   = self.allied_champs(allied_champs)
            allied_champs_s.append(allied_champs)
        allied_champs_s = torch.cat(tuple(allied_champs_s), axis=2)

        # Observation - Enemy Champs
        enemy_champs_s = []
        for i in range(UNIT_COUNTS["champs"] // 2):
            enemy_champs_a = x["champs"].iloc[:, 0:14].to_numpy()
            enemy_champs_b = x["champs"].iloc[:, 28:65].to_numpy()
            enemy_champs   = np.concatenate((enemy_champs_a, enemy_champs_b), axis=1)
            enemy_champs   = torch.Tensor(enemy_champs).unsqueeze(0)
            enemy_champs   = self.enemy_champs(enemy_champs)
            enemy_champs_s.append(enemy_champs)
        enemy_champs_s = torch.cat(tuple(enemy_champs_s), axis=2)

        # Observation - Neutral Units
        neutral_units_s = []
        for i in range(UNIT_COUNTS["jungle"]):
            off          = i * UNIT_FEATURE_COUNTS["jungle"]
            jungle_camps = x["jungle"].iloc[:, off+0:off+17].to_numpy()
            jungle_camps = torch.Tensor(jungle_camps).unsqueeze(0)
            jungle_camps = self.neutral_units(jungle_camps)
            neutral_units_s.append(jungle_camps)
        neutral_units_s = torch.cat(tuple(neutral_units_s), axis=2)

        # Observation - Concat
        obs = torch.cat((
            globals,
            allied_champs_s,
            enemy_champs_s,
            neutral_units_s
        ), axis=2)

        print(obs.shape)

        # Decide
        x    = F.relu(self.lstm_fc(obs))
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