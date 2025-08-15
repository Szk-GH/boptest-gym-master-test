import requests

# # url for the BOPTEST service
url = 'http://127.0.0.1:80'

### 第一部分

# # Select test case and get identifier
# testcase = 'bestest_hydronic_heat_pump'
# # Check if already started a test case and stop it if so before starting another
# try:
#   requests.put('{0}/stop/{1}'.format(url, testid))
# except:
#   pass
#
# # Select and start a new test case
# testid = \
# requests.post('{0}/testcases/{1}/select'.format(url,testcase)).json()['testid']
#
# # Get test case name
# name = requests.get('{0}/name/{1}'.format(url, testid)).json()['payload']
# print(name)
#
# # Get inputs available
# inputs = requests.get('{0}/inputs/{1}'.format(url, testid)).json()['payload']
# print('TEST CASE INPUTS ---------------------------------------------')
# print(inputs.keys())
# # Get measurements available
# print('TEST CASE MEASUREMENTS ---------------------------------------')
# measurements = requests.get('{0}/measurements/{1}'.format(url, testid)).json()['payload']
# print(measurements.keys())
#
# requests.put('{0}/stop/{1}'.format(url, testid))



### 第二部分

from boptestGymEnv import BoptestGymEnv

# Instantiate environment
# env = BoptestGymEnv(url                   = url,
#                     testcase              = 'bestest_hydronic_heat_pump',
#                     actions               = ['oveHeaPumY_u'],
#                     observations          = {'reaTZon_y':(280.,310.)},
#                     random_start_time     = False,
#                     start_time            = 31*24*3600,
#                     max_episode_length    = 24*3600,
#                     warmup_period         = 24*3600,
#                     step_period           = 3600)
#
# obs, _ = env.reset()
# print('Zone temperature: {:.2f} degC'.format(obs[0]-273.15))   # the zone operative temperature (`reaTZon_y`)
# print('Episode starting day: {:.1f} (from beginning of the year)'.format(env.start_time/24/3600))
#
# # 考察任意环境的observation与action空间：该环境有一个Box（连续有界）的观察空间，即室内建筑温度。
# # 动作空间是一个从0到1的连续变量，0表示不加热，1表示热泵满负荷工作。
# print('Observation space of the building environment:')
# print(env.observation_space)
# print('Action space of the building environment:')
# print(env.action_space)

'''
BOPTEST-Gym 还附带其他一些在训练强化学习 (RL) 智能体时可能有用的功能，例如离散化和规范化观察和动作空间的能力。
例如，我们现在处理的是连续动作环境，这意味着智能体可以决定采取 0 到 1 之间的任何动作。
然而，让智能体决定是否需要打开（action=1）或关闭（action=0）加热可能更有帮助。为此，我们可以将环境包裹在一个只有一个动作箱（一个箱有两个极值）的离散化包装器 (wrapper) 中。
包装器的概念在 Gym 环境中非常强大。有了它们，我们能够自定义环境的观察、动作、阶跃函数等。无论应用了多少个包装器，
“env.unwrapped”始终返回内部的原始环境对象。让我们看看它如何与 BOPTEST-Gym 配合使用：
'''

from boptestGymEnv import DiscretizedActionWrapper
# env = DiscretizedActionWrapper(env,n_bins_act=1)
# print('Action space of the wrapped agent:')
# print(env.action_space)
# print('Action space of the original agent:')
# print(env.unwrapped.action_space)


# 与建筑环境进行一次体验互动（一天），只运行一次，并使用一个滞后控制器。当温度低于预设温度设定值时，它会打开暖气；当温度高于设定值时，它会关闭暖气。我们首先配置这样的控制器：
import numpy as np
# np.set_printoptions(precision=3)

# class SimpleController(object):
#     '''Simple controller for this emulator.
#
#     '''
#     def __init__(self, TSet=22+273.15):
#         self.TSet = TSet
#
#     def predict(self, obs):
#         # Compute control
#         if obs[0]<self.TSet:
#             action = np.asarray(1) # Turn on heating
#         else:
#             action = np.asarray(0) # No heating needed
#
#         return action
#
# model = SimpleController(TSet=22+273.15)
#
# done = False
# obs, _ = env.reset()

# 当室内温度低于设定值时，控制器如何决定打开加热（“action = 1”），当温度高于设定值时，控制器如何关闭加热。
# from IPython.display import clear_output
# while not done:
#   # Clear the display output at each step
#   clear_output(wait=True)
#   # Compute control signal
#   action = model.predict(obs)
#   # Print the current operative temperature and decided action
#   print('-------------------------------------------------------------------')
#   print('Operative temperature [degC]  = {:.2f}'.format(obs[0]-273.15))
#   print('Action                [ - ]   = {:.0f}'.format(action))
#   print('-------------------------------------------------------------------')
#   # Implement action
#   obs,reward,terminated,truncated,info = env.step(action) # send the action to the environment
#   done = (terminated or truncated)

'''
# 控制器被实例化为“model”，因为强化学习智能体通常使用model（例如，任何通用函数逼近器或神经网络）来表示其策略。实例名称的选择是任意的，
# 但“model”在历史上已被不同的强化学习框架（例如Stable Baselines）所接受。我们也可以选用其他名称，但我们使用“model”来熟悉这一约定。
# “predict”方法用于根据强化学习智能体的模型估计要采取的操作。
'''



### 第三部分
'''
我们将基于非常著名的*q-learning*算法开发一个非常简单的强化学习智能体。
假设我们在时间t处于某个状态S，并采取了行动A。作为回报，我们在下一个时间步获得奖励r，最终处于状态S'：
我们实施了一种 Epsilon-greedy方法来平衡强化学习代理的探索和利用。也就是说，智能体有时会选择random action（探索），有时会选择“智能”动作（利用）。
智能体选择随机动作的频率由 Epsilon (eps) 决定，并遵循线性衰减的规律。

我们的 Q_Learning_Agent 仅包含三个方法：
  __init__   构造函数。
  predict   根据观察结果决定动作的方法。
  learn   使用上面解释的 q-learning 方法进行学习的方法。
'''




# class Q_Learning_Agent(object):
#
#   def __init__(self, env, eps_min=0.01, eps_decay=0.01, alpha=0.05, gamma=0.9):
#     '''Constructor of a q-learning agent. Assumes discrete state and action spaces.
#     # 假设离散状态和动作空间
#     '''
#     self.env       = env
#     self.eps_min   = eps_min
#     self.eps_decay = eps_decay
#     self.alpha     = alpha
#     self.gamma     = gamma
#
#     # Initialize epsilon
#     self.eps       = 1.0
#
#     # Initialize q-function as a null function
#     self.q = np.zeros((env.observation_space.n,
#                        env.action_space.n))
#
#   def predict(self, obs, deterministic=True):
#     '''Method to select an action with an epsilon-greedy policy.
#
#     '''
#     if deterministic:
#       # Use q-function to decide action
#       return np.argmax(self.q[obs])
#     else:
#       if self.eps > self.eps_min:
#         # Linearly decreasing schedule
#         self.eps -= self.eps_decay
#       if np.random.random() < self.eps:
#         # Explore with random action
#         return np.random.choice([a for a in range(env.action_space.n)])
#       else:
#         # Exploit the information of our q-function
#         return np.argmax(self.q[obs])
#
#   def learn(self, total_episodes=10):
#     '''Learn from a number of interactions with the environment.
#
#     '''
#     for i in range(total_episodes):
#       # Initialize enviornment
#       done = False
#       obs, _  = env.reset()
#       # Print episode number and starting day from beginning of the year:
#       print('-------------------------------------------------------------------')
#       print('Episode number: {0}, starting day: {1:.1f} ' \
#             '(from beginning of the year)'.format(i+1, env.unwrapped.start_time/24/3600))
#
#       while not done:
#         # Get action with epsilon-greedy policy and simulate
#         act                   = self.predict(obs, deterministic=False)
#         nxt_obs, rew, terminated, truncated, _ = env.step(act)
#         done = (terminated or truncated)
#         # Compute temporal difference target and error to udpate q-function
#         td_target         = rew + self.gamma*np.max(self.q[nxt_obs])
#         td_error          = td_target - self.q[obs][act]
#         self.q[obs][act] += self.alpha*td_error
#         # Make our next observation the current observation
#         obs = nxt_obs
#       # Print the q-function after every episode to show progress
#       print('q(s,a) = ')
#       print(self.q)
#
#
# # 现在已经准备好了强化学习智能体，接下来在 BOPTEST-Gym 中测试它！我们将利用 BOPTEST-Gym 的功能来：
# # - 定义环境的自定义奖励函数
# # - 实例化环境并定义其状态和动作空间
# # - 训练我们的强化学习智能体 排除那些对学习不感兴趣的时期（只保留冬季时期）
# # - 离散化环境的状态和动作空间
# # - 定义环境的自定义奖励函数，奖励函数的定义至关重要，因为它驱动着智能体的学习。`BoptestGymEnv` Class允许覆盖其 `get_reward` 方法（该方法在每个控制步骤中都会调用），以便自由定义任何选择的奖励函数
#
# '''
# 在示例中，目标是实现一个强化学习智能体来识别哪些行为能够保持建筑物内的舒适度，我们应该相应地对奖励函数进行编码。
# 我们可以通过积分舒适度范围之外的温度偏差来实现这个函数,然而，这种方法容易出错。
# 我们通常希望直接使用来自environment的信号来定义奖励，最好是那些与我们要优化的函数直接相关的信号，这样我们才能确保达到最佳效果。
# 在 BOPTEST 中，我们使用 `GET /kpis` API 获取当前所谓的核心 KPI，包括：
#     **Thermal discomfort**：以 [K, h/zone] 为单位报告，定义区域温度与测试用例 FMU 中预先定义的每个区域舒适度上限和下限的累积偏差，并对所有区域取平均值。空气系统使用Air temperature，辐射系统使用operative temperature
#     **Indoor Air Quality (IAQ) Discomfort**：以 [ppm, h/zone] 为单位报告，定义区域内 CO2浓度水平超出可接受浓度范围的程度，可接受浓度范围在每个区域的测试用例 FMU 中预先定义，并对所有区域取平均值
#     **Energy Use**：以 [kWh/m^2] 为单位报告，定义 HVAC 能耗
#     **Cost**：以 [USD/m^2] 或 [EUR/m^2] 为单位报告，定义与 HVAC 能源使用相关的运营成本
#     **Emissions**：以 [kg, CO_2/m^2] 为单位报告，定义 HVAC 能源使用产生的二氧化碳排放量
#     **Computational time ratio**：定义控制器计算时间与测试模拟控制步长之间的平均比率。控制器计算时间是通过两个模拟器之间的时间间隔来衡量的
# '''
#
# # 核心 KPI 通常在模拟结束时计算，以评估控制器的性能，但它们也可以随时计算。预热期不计入 KPI 的计算中。
# # 下文介绍 `GET /kpi` 定义 `get_reward` 方法。在每个控制步骤中，我们都会检查是否存在不适感增量。如果没有不适感增量，我们奖励智能体1，否则返回0（无奖励），限制奖励是加速学习的好方法。
#
#
# # Redefine reward function
#
# class BoptestGymEnvCustomReward(BoptestGymEnv):
#     '''Define a custom reward for this building
#
#     '''
#     def get_reward(self):
#         '''
#
#         自定义奖励函数。为了加快学习速度，我们使用了一个经过裁剪的奖励函数，当不适感没有增加时，其值为 1，否则为 0。
#         我们使用 BOPTEST `GET /kpis` API 调用来计算从episode开始的总累积不适感。
#         请注意，这是 BOPTEST 在评估控制器时使用的真实值。
#
#         '''
#         # Compute BOPTEST core kpis
#         kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
#         # Calculate objective integrand function as the total discomfort
#         objective_integrand = kpis['tdis_tot']
#         # Give reward if there is not immediate increment in discomfort
#         if objective_integrand == self.objective_integrand:
#           reward=1
#         else:
#           reward=0
#         # Record current objective integrand for next evaluation
#         self.objective_integrand = objective_integrand
#         return reward
#
# '''
# 与“SimpleController”示例类似，现在我们将使用一个智能体，它仅观察当前室内温度并决定是否打开或关闭供暖。我们使用自己实现的“Q_Learning_Agent”来查看它是否能够学习如何做到这一点。
# 为此，我们将让我们的强化学习智能体与建筑物进行一些经验轮次的交互。
# 由于我们现在要运行多个轮次进行训练，因此我们希望停止之前的环境，并启动一个全年随机初始化建筑物模拟器的环境。
# 这样，我们就可以在建筑物环境中使用不同的边界条件数据来训练我们的智能体。由于我们只专注于学习供暖行为，因此我们将排除春季、夏季和秋季的训练时间。
# '''
#
# import random
#
# # Seed for random starting times of episodes
# seed = 123456
# random.seed(seed)
# # Seed for random exploration and epsilon-greedy schedule
# np.random.seed(seed)
#
# # Winter period goes from December 21 (day 355) to March 20 (day 79)
# excluding_periods = [(79*24*3600, 355*24*3600)]
# # Temperature setpoints
# lower_setp = 21 + 273.15
# upper_setp = 24 + 273.15
# # Instantiate environment
# env = BoptestGymEnvCustomReward(url                   = url,
#                                 testcase              = 'bestest_hydronic_heat_pump',
#                                 actions               = ['oveHeaPumY_u'],
#                                 observations          = {'reaTZon_y':(lower_setp,upper_setp)},
#                                 random_start_time     = True,
#                                 excluding_periods     = excluding_periods,
#                                 max_episode_length    = 2*24*3600,
#                                 warmup_period         = 24*3600,
#                                 step_period           = 3600,
#                                 render_episodes       = True)
#
# '''
# 我们将zone temperature设置为环境状态的唯一观测值。我们还将该变量的下限和上限分别设置为21°C和24°C，这是有人居住期间舒适温度范围的边界。
# 环境可以使用这些边界进行归一化或离散化。实际上，我们将对动作空间和观察空间进行离散化，以加快学习速度。我们决定只为动作空间设置一个箱体（两种可能的动作：开启或关闭暖气）。
# 我们将观察空间分成三个箱体，舒适温度范围的外边界作为观察空间的箱体（“outs_are_bins=True”）。也就是说，观察空间定义为[-∞,21,24,+∞]，如下图左侧所示。
# 请注意，只有中间的箱体始终舒适，而其他箱体可能会导致不适。如果我们设置了“outs_are_bins=False”，所有箱体都将位于舒适温度范围内。
# 后者会让智能体了解舒适范围内的温度（接近下限、中间或接近上限），但如果温度超出范围，就会引发错误。
# '''
#
# from boptestGymEnv import DiscretizedObservationWrapper
# env = DiscretizedActionWrapper(env, n_bins_act=1)
# env = DiscretizedObservationWrapper(env, n_bins_obs=3, outs_are_bins=True)
#
# # 训练智能体
# '''
# 唯一缺少的步骤是让我们的强化学习智能体通过在环境中不断积累经验来进行学习。我们使用之前定义的 learn 方法来实现这一点。
# 需要注意的是，由于我们设置了 render_episodes=True，因此在每轮结束后，我们都会收到一个包含相关变量的图表，这有助于检查智能体在早期阶段是否按预期进行学习。
# 如果智能体没有表现出任何活力，我们可以提前停止学习过程，使用新的学习设置，从而节省宝贵的时间和计算成本。
# '''
# model = Q_Learning_Agent(env, eps_min=0.01, eps_decay=0.001, alpha=0.1, gamma=0.9)
# model.learn(total_episodes=10)
#
# # 由于我们的环境已经用一维状态和动作空间定义，我们可以在训练后绘制 q 函数
#
# # import matplotlib.pyplot as plt
# #
# # acts   = ['a=0','a=1']
# # stas   = ['T<21', '21<T<24', 'T>24']
# # colors = ['b',    'g',       'r']
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.set_xlabel('actions',           labelpad=6,  fontsize=12)
# # ax.set_ylabel('states',            labelpad=10, fontsize=12)
# # ax.set_zlabel('$\mathbf{q(s,a)}$', labelpad=0,  fontsize=15)
# # plt.xticks(ticks=range(len(acts)), labels=acts)
# # plt.yticks(ticks=range(len(stas)), labels=stas)
# #
# # for i, s in enumerate(stas):
# #   x = np.arange(len(acts))
# #   h = model.q[i,:]
# #
# #   # Set color
# #   color = [colors[i]]*len(acts)
# #
# #   # Plot the 3D bar graph
# #   ax.bar(x, h, zs=i, zdir='y', color=color, alpha=0.8)
# #
# # plt.show()
#
# '''
# 有时，了解处于特定状态的价值（与要采取的行动无关）很有用，我们可以轻松地计算并绘制我们案例的价值函数：
# '''
#
# # # Compute the state-value function
# # v = np.amax(model.q, axis=1)
# #
# # # Plot state-value function
# # fig = plt.figure()
# #
# # ax = fig.add_subplot(111)
# # ax.set_xlabel('states', labelpad=10, fontsize=12)
# # ax.set_ylabel('$\mathbf{v(s)}$', labelpad=0,  fontsize=15)
# # plt.xticks(ticks=range(len(stas)), labels=stas)
# # x = np.arange(len(stas))
# # ax.bar(x, v, color=colors, alpha=0.8)
# # plt.show()
#
# '''
# 我们训练智能体时采用了一种离策略方法：这些动作由与智能体实际遵循的策略不同的策略驱动。这是因为智能体使用了 epsilon-greedy 策略来探索更具回报性的动作。
# 如果我们对学习到的策略感到满意，可以通过在 predict 方法中设置 deterministic=True 来测试它。例如，让我们测试一下我们学习到的智能体在二月第一天的表现：
# '''
#
# env.stop()
# env = BoptestGymEnvCustomReward(url                   = url,
#                                 testcase              = 'bestest_hydronic_heat_pump',
#                                 actions               = ['oveHeaPumY_u'],
#                                 observations          = {'reaTZon_y':(lower_setp,upper_setp)},
#                                 random_start_time     = False,
#                                 start_time            = 31*24*3600,
#                                 max_episode_length    = 24*3600,
#                                 warmup_period         = 24*3600,
#                                 step_period           = 3600)
# env = DiscretizedActionWrapper(env, n_bins_act=1)
# env = DiscretizedObservationWrapper(env, n_bins_obs=3, outs_are_bins=True)
#
# done = False
# obs, _ = env.reset()
#
# from IPython.display import clear_output
# while not done:
#   # Clear the display output at each step
#   clear_output(wait=True)
#   # Compute control signal
#   action = model.predict(obs, deterministic=True)
#   # Print the current operative temperature and decided action
#   print('-------------------------------------------------------------------')
#   print('State  [Bin #]  = {:.0f}'.format(obs))
#   print('Action [ - ]    = {:.0f}'.format(action))
#   print('-------------------------------------------------------------------')
#   # Implement action
#   obs,reward,terminated,truncated,info = env.step(action) # send the action to the environment
#   done = (terminated or truncated)
#
# '''
# 现在不再涉及随机性。智能体通过始终在 `s=0` 时选择动作 `a=1` 来利用其策略，因为它已经学习到这是该状态下值最高的动作。
# 现在，我们可以通过使用 BOPTEST 计算核心 KPI 来评估我们学到的策略：
# '''
#
# env.get_kpis()


### 第四部分
'''
最后，我们将通过扩展观察空间，添加周内时间以及环境温度、太阳辐射、内部增益、电价或温度设定值等信息，来实例化一个更完整的建筑环境。
借助 BOPTEST-Gym，我们还可以建立一个预测期和一个回归期，分别包含边界条件数据的预测值和测量数据的过去观测值。
由于其高维状态-动作空间，智能体可能需要进行更多交互才能解决此环境问题。幸运的是，有一些现成的先进强化学习算法，它们运用您之前学到的学习原理，同时实现各种技巧来加速和稳定学习。
例如，我们可以访问 Stable-Baselines3 中的高级深度 Q 网络 (DQN) 算法来学习这个更复杂的环境。我们在这里设置智能体学习 10 个步骤，以展示如何启动此学习过程。
'''

from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper
from stable_baselines3 import DQN


# Decide the state-action space of your test case
env = BoptestGymEnv(
        url                  = url,
        testcase             = 'bestest_hydronic_heat_pump',
        actions              = ['oveHeaPumY_u'],
        observations         = {'time':(0,604800),
                                'reaTZon_y':(280.,310.),
                                'TDryBul':(265,303),
                                'HDirNor':(0,862),
                                'InternalGainsRad[1]':(0,219),
                                'PriceElectricPowerHighlyDynamic':(-0.4,0.4),
                                'LowerSetp[1]':(280.,310.),
                                'UpperSetp[1]':(280.,310.)},
        predictive_period    = 24*3600,
        regressive_period    = 6*3600,
        random_start_time    = True,
        max_episode_length   = 24*3600,
        warmup_period        = 24*3600,
        step_period          = 3600)

# Normalize observations and discretize action space
env = NormalizedObservationWrapper(env)
env = DiscretizedActionWrapper(env,n_bins_act=10)


# Instantiate an RL agent
model = DQN('MlpPolicy', env, verbose=1, gamma=0.99,
            learning_rate=5e-4, batch_size=24, seed=123456,
            buffer_size=365*24, learning_starts=24, train_freq=1)

# Main training loop
model.learn(total_timesteps=10)

# Loop for one episode of experience (one day as set in max_episode_length)
done = False
obs, _ = env.reset()
while not done:
  action, _ = model.predict(obs, deterministic=True)
  obs,reward,terminated,truncated,info = env.step(action)
  done = (terminated or truncated)

# Obtain KPIs for evaluation
env.get_kpis()
print(env.get_kpis())


print('Observation space of the building environment (dimension):')
print(env.observation_space.shape)
print('Action space of the building environment:')
print(env.action_space)
