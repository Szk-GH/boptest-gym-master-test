import random
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper, NormalizedActionWrapper, SaveAndTestCallback
from stable_baselines3 import DQN
import os
import utilities
import requests
from collections import OrderedDict
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from test_and_plot import plot_results
from gymnasium.core import Wrapper
import json

url = 'http://127.0.0.1:80'
seed = 123456

# Seed for random starting times of episodes
random.seed(seed)


max_episode_length  = 14*24*3600
warmup_period       = 1*24*3600

import requests

# 测试服务是否可达
try:
    response = requests.get(f"{url}/testcases")
    print("Available testcases:", response.json())
except Exception as e:
    print(f"无法连接到 Boptest 服务: {e}")
    raise

# env = BoptestGymEnv(
#     url=url,
#     testcase='bestest_hydronic_heat_pump',
#     actions=['oveHeaPumY_u'],
#     observations=OrderedDict([('time', (0, 604800)),
#                               ('reaTZon_y', (280., 310.)),
#                               ('TDryBul', (265, 303)),
#                               ('HDirNor', (0, 862)),
#                               ('InternalGainsRad[1]', (0, 219)),
#                               ('PriceElectricPowerHighlyDynamic', (-0.4, 0.4)),
#                               ('LowerSetp[1]', (280., 310.)),
#                               ('UpperSetp[1]', (280., 310.))]),
#     predictive_period=24 * 3600,  # 预测未来24小时的数据
#     regressive_period=6 * 3600,  # 回顾过去6小时的数据
#     scenario={'electricity_price': 'highly_dynamic'},
#     random_start_time=True,  # 随机初始化训练起始时间
#     max_episode_length=max_episode_length,
#     warmup_period=warmup_period,
#     step_period=900,  # 控制步长（15分钟执行一次动作）
#     render_episodes=None,  # 是否可视化训练过程
#     log_dir=None)  # 保存训练日志和结果的目录
#
# obs, _ = env.reset()
#
#
# print("time dtype:", obs["time"].dtype)
# print("reaTZon_y dtype:", obs["reaTZon_y"].dtype)
# print("TDryBul dtype:", obs["TDryBul"].dtype)
# print("HDirNor dtype:", obs["TDryBul"].dtype)
# print("InternalGainsRad dtype:", obs["InternalGainsRad[1]"].dtype)
# print("PriceElectricPowerHighlyDynamic dtype:", obs["PriceElectricPowerHighlyDynamic"].dtype)
# print("LowerSetp dtype:", obs["LowerSetp[1]"].dtype)
# print("UpperSetp dtype:", obs["UpperSetp[1]"].dtype)
# print("predictive_period dtype:", env.predictive_period.dtype)
# print("regressive_period dtype:", env.regressive_period.dtype)
# # print("scenario dtype:", env.scenario.dtype)
# print("warmup_period dtype:", env.warmup_period.dtype)
# print("step_period dtype:", env.step_period.dtype)