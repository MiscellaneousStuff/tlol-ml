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


class UnitTypeModule(nn.Module):
    def __init__(self,
                 unit_count,
                 feature_count,
                 is_enemy,
                 unit_ins,
                 unit_outs,
                 unit_s_outs):
        super().__init__()
        self.unit_module = UnitModule(unit_ins, unit_outs)
        unit_s_ins = unit_count * unit_outs
        self.fc1 = nn.Linear(unit_s_ins, unit_s_ins // 2)
        self.fc2 = nn.Linear(unit_s_ins // 2, unit_s_outs)

        self.unit_count = unit_count
        self.feature_count = feature_count
        self.is_enemy = is_enemy

    def forward(self, x):
        units_s = []

        for i in range(self.unit_count):
            off   = i * self.feature_count
            units = x.iloc[0:2, off+0:off+self.feature_count].to_numpy()
            units = torch.Tensor(units).unsqueeze(0)
            units = self.unit_module(units)
            units_s.append(units)
        units_s = torch.cat(tuple(units_s), axis=2)

        x = F.relu(self.fc1(units_s))
        x = self.fc2(x)

        return x


class Agent(nn.Module):
    def __init__(self, example, unit_out=16, model_size=1024):
        super().__init__()

        # Observation - Old
        self.allied_champs = UnitModule(51, unit_out)
        self.enemy_champs  = UnitModule(51, unit_out)
        # self.neutral_units = UnitModule(17, unit_out // 4)
        self.neutral_units = UnitTypeModule(
            unit_count=24,
            feature_count=17,
            is_enemy=False,
            unit_ins=17,
            unit_outs=unit_out // 4,
            unit_s_outs=(unit_out // 4)*24)
        self.allied_units  = UnitModule(17, unit_out // 4)
        self.enemy_units   = UnitModule(17, unit_out // 4)

        # Observation - New
        # Decide
        ins = \
            2 + \
            unit_out*5 + \
            unit_out*5 + \
            (unit_out // 4)*24 + \
            (unit_out // 4)*30 + \
            (unit_out // 4)*11 + \
            (unit_out // 4)*30 + \
            (unit_out // 4)*11

        print("DECIDE INS:", ins)
        self.lstm_fc = nn.Linear(ins, ins)
        self.lstm = nn.LSTM(ins, model_size, num_layers=1, batch_first=True)

        # Action Head
        self.action = nn.Linear(model_size, 1)

    def forward(self, x):
        # Observation
        globals = x["raw"].iloc[0:2, 0:2].to_numpy()
        globals = torch.Tensor(globals).unsqueeze(0)

        # Observation - Allied Champs
        allied_champs_s = []
        for i in range(UNIT_COUNTS["champs"] // 2):
            off             = i * UNIT_FEATURE_COUNTS["champs"]
            allied_champs_a = x["champs"].iloc[0:2, 0+off:14+off].to_numpy()
            allied_champs_b = x["champs"].iloc[0:2, 28+off:65+off].to_numpy()
            allied_champs   = np.concatenate((allied_champs_a, allied_champs_b), axis=1)
            allied_champs   = torch.Tensor(allied_champs).unsqueeze(0)
            allied_champs   = self.allied_champs(allied_champs)
            allied_champs_s.append(allied_champs)
        allied_champs_s = torch.cat(tuple(allied_champs_s), axis=2)

        # Observation - Enemy Champs
        enemy_champs_s = []
        for i in range(UNIT_COUNTS["champs"] // 2):
            off            = i * UNIT_FEATURE_COUNTS["champs"]
            off            += (UNIT_COUNTS["champs"] // 2) * UNIT_FEATURE_COUNTS["champs"]
            enemy_champs_a = x["champs"].iloc[0:2, off+0:off+14].to_numpy()
            enemy_champs_b = x["champs"].iloc[0:2, off+28:off+65].to_numpy()
            enemy_champs   = np.concatenate((enemy_champs_a, enemy_champs_b), axis=1)
            enemy_champs   = torch.Tensor(enemy_champs).unsqueeze(0)
            enemy_champs   = self.enemy_champs(enemy_champs)
            enemy_champs_s.append(enemy_champs)
        enemy_champs_s = torch.cat(tuple(enemy_champs_s), axis=2)

        neutral_units_s = self.neutral_units(x["jungle"])

        # Observation - Allied Units
        allied_units_s = []
        for i in range(UNIT_COUNTS["minions"] // 2):
            off = i * UNIT_FEATURE_COUNTS["minions"]
            allied_minions = x["minions"].iloc[0:2, off+0:off+17].to_numpy()
            allied_minions = torch.Tensor(allied_minions).unsqueeze(0)
            allied_minions = self.allied_units(allied_minions)
            allied_units_s.append(allied_minions)
        for i in range(UNIT_COUNTS["turrets"] // 2):
            off = i * UNIT_FEATURE_COUNTS["turrets"]
            allied_turrets = x["turrets"].iloc[0:2, off+0:off+17].to_numpy()
            allied_turrets = torch.Tensor(allied_turrets).unsqueeze(0)
            allied_turrets = self.allied_units(allied_turrets)
            allied_units_s.append(allied_turrets)
        allied_units_s = torch.cat(tuple(allied_units_s), axis=2)

        # Observation - Enemy Units
        enemy_units_s = []
        for i in range(UNIT_COUNTS["minions"] // 2):
            off            = i * UNIT_FEATURE_COUNTS["minions"]
            off            += (UNIT_COUNTS["minions"] // 2) * UNIT_FEATURE_COUNTS["minions"]
            enemy_minions = x["minions"].iloc[0:2, off+0:off+17].to_numpy()
            enemy_minions = torch.Tensor(enemy_minions).unsqueeze(0)
            enemy_minions = self.enemy_units(enemy_minions)
            enemy_units_s.append(enemy_minions)
        for i in range(UNIT_COUNTS["turrets"] // 2):
            off            = i * UNIT_FEATURE_COUNTS["turrets"]
            off            += (UNIT_COUNTS["turrets"] // 2) * UNIT_FEATURE_COUNTS["turrets"]
            enemy_turrets = x["turrets"].iloc[0:2, off+0:off+17].to_numpy()
            enemy_turrets = torch.Tensor(enemy_turrets).unsqueeze(0)
            enemy_turrets = self.allied_units(enemy_turrets)
            enemy_units_s.append(enemy_turrets)
        enemy_units_s = torch.cat(tuple(enemy_units_s), axis=2)

        # Observation - Concat
        obs = torch.cat((
            globals,
            allied_champs_s,
            enemy_champs_s,
            neutral_units_s,
            allied_units_s,
            enemy_units_s
        ), axis=2)

        print("obs.shape:", obs.shape)

        # Decide
        x    = F.relu(self.lstm_fc(obs))
        x, _ = self.lstm(obs)

        # Action Head
        x = F.sigmoid(self.action(x))

        return x


if __name__ == "__main__":
    dataset = TLoLReplayDataset("./full_db")

    example = dataset[0]
    agent   = Agent(example)
    action  = agent(example)

    print(action)