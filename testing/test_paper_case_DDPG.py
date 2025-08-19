
import random
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper, NormalizedActionWrapper, SaveAndTestCallback
from stable_baselines3 import DDPG
import os
import utilities
import requests
from collections import OrderedDict
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from test_and_plot import plot_results
from gymnasium.core import Wrapper
import json
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

url = 'http://127.0.0.1:80'
seed = 123456

# Seed for random starting times of episodes
random.seed(seed)


def train_RL(algorithm          = 'DDPG' ,
             start_time_tests    = [(23-7)*24*3600, (115-7)*24*3600],
             episode_length_test = 14*24*3600,
             warmup_period       = 1*24*3600,
             max_episode_length  = 14*24*3600,  #  训练时每个episode的最大长度（14天）
             mode                = 'train',
             training_timesteps  = 1e5,
             render              = False,  #  是否可视化训练过程
             expert_traj         = None,
             model_name          = 'last_model'):

    # 测试时间段1：第16天 → 第30天  测试时间段2：第108天 → 第122天  夏季固定排除：第173天 → 第266天
    excluding_periods = []
    for start_time_test in start_time_tests:
        excluding_periods.append((start_time_test,
                                  start_time_test + episode_length_test))
    # Summer period (from June 21st till September 22nd).
    # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173 * 24 * 3600, 266 * 24 * 3600))

    # Create a log directory
    # 为强化学习训练过程创建一个专用的日志目录，用于保存训练结果、模型检查点或其他输出文件
    log_dir = os.path.join(utilities.get_root_path(), 'testing',
        'agents', '{}_{}_logdir'.format(algorithm,training_timesteps))
    log_dir = log_dir.replace('+', '')
    os.makedirs(log_dir, exist_ok=True)

    # Redefine reward function
    # 自定义了一个奖励函数，继承自基础的 BoptestGymEnv类
    class BoptestGymEnvCustomReward(BoptestGymEnv):
        '''Define a custom reward for this building

        '''
        def get_reward(self):
            '''Custom reward function

            '''

            # Compute BOPTEST core kpis
            kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']

            # Calculate objective integrand function at this point
            objective_integrand = kpis['cost_tot'] + 10*(kpis['tdis_tot'])

            # Compute reward
            reward = -(objective_integrand - self.objective_integrand)

            self.objective_integrand = objective_integrand

            return reward

    env = BoptestGymEnvCustomReward(
                                    url=url,
                                    testcase='bestest_hydronic_heat_pump',
                                    actions=['oveHeaPumY_u'],
                                    observations=OrderedDict([('time', (0, 604800)),
                                                              ('reaTZon_y', (280., 310.)),
                                                              ('TDryBul', (265, 303)),
                                                              ('HDirNor', (0, 862)),
                                                              ('InternalGainsRad[1]', (0, 219)),
                                                              ('PriceElectricPowerHighlyDynamic', (-0.4, 0.4)),
                                                              ('LowerSetp[1]', (280., 310.)),
                                                              ('UpperSetp[1]', (280., 310.))]),
                                    predictive_period=24 * 3600,  #  预测未来24小时的数据
                                    regressive_period=6 * 3600,   #  回顾过去6小时的数据
                                    scenario={'electricity_price': 'highly_dynamic'},
                                    random_start_time=True,   #  随机初始化训练起始时间
                                    excluding_periods=excluding_periods,
                                    max_episode_length=max_episode_length,
                                    warmup_period=warmup_period,
                                    step_period=900,   #  控制步长（15分钟执行一次动作）
                                    render_episodes=render,     #  是否可视化训练过程
                                    log_dir=log_dir)     #  保存训练日志和结果的目录

    # 通过环境包装器（Wrapper）对强化学习环境进行了两种不同的预处理，分别是观测值标准化和动作离散化-适应某些只能处理离散动作的算法（如DQN），或简化控制策略
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)

    #  为连续动作空间添加动作噪声，使用Ornstein-Uhlenbeck (OU) 噪声:帮助智能体（Agent）在探索环境时更高效，
    n_actions = env.action_space.shape[0]   # 获取动作空间的维度（即智能体可以执行多少个连续动作）
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions),
        theta=0.15
    )


    # Modify the environment to include the callback
    # 通过 Monitor 包装器对强化学习环境进行了运行监控和数据记录：自动记录每个训练episode的关键指标（如累计奖励、步数、耗时等）；将数据保存为CSV文件（monitor.csv），便于后续分析和可视化。
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))

    if mode == 'train':
        model = DDPG( 'MlpPolicy',
                    env,
                    verbose=1,   # 输出训练日志（1=基本进度，2=详细调试）
                    gamma=0.99,   # 折扣因子
                    seed=seed,   # 固定随机种子（确保实验可复现）
                    learning_rate=1e-4,   # 神经网络优化器的学习率
                    batch_size=512,   # 每次从经验回放缓冲中采样的数据量
                    buffer_size=100000,
                    learning_starts=96,   # 训练开始前先收集24步数据填充缓冲区
                    train_freq=1,   # 每1步训练一次网络
                    action_noise=action_noise,
                    gradient_steps=1,
                    policy_kwargs= {"net_arch":[400,300]},   # 两隐藏层：400 → 300维
                    tensorboard_log=log_dir)   # 将训练日志输出到TensorBoard（可视化工具）

        # Create the callback test and save the agent while training
        # 创建一个自定义回调函数 SaveAndTestCallback，用于在强化学习训练过程中实现定期保存模型和性能测试的功能
        callback = SaveAndTestCallback(env,
                                       check_freq=1e10,    # 测试频率
                                       save_freq=1e5,    # 保存频率
                                       log_dir=log_dir,    # 模型和测试结果的保存目录
                                       test=False)    # 是否启用定期测试

        # set up logger
        # 配置强化学习模型的日志系统，将训练过程中的关键指标（如奖励、损失等）记录到CSV文件中，便于后续分析和可视化
        new_logger = configure(log_dir, ['csv'])   # 设置日志存储路径和格式（此处选择CSV格式
        model.set_logger(new_logger)   # 将配置好的日志器绑定到DQN模型

        # Main training loop
        # 完成模型训练和最终保存
        model.learn(total_timesteps=int(training_timesteps), callback=callback)   # 总训练步数并自定义回调函数

        # Save the agent
        # 将训练完成的模型保存到指定路径，包含：神经网络权重、优化器状态、环境参数
        model.save(os.path.join(log_dir,model_name))

    elif mode == 'load':
        # Load the trained agent
        env = DiscretizedActionWrapper(env)
        model = (DDPG.load(os.path.join(log_dir,model_name)))

    elif mode == 'empty':
        model = None

    else:
        raise ValueError('mode should be either train, load, or empty')

    return env, model, start_time_tests, log_dir


def run_event(env,
              model,
              start_time_tests,
              episode_length_test,
              warmup_period_test,
              log_dir=os.getcwd(),
              model_name='last_model',
              save_to_file=True,
              plot=True):
    ''' Perform test in peak heat period (February).

    '''

    # 确保布尔参数转换为字符串
    save_str = str(save_to_file).lower()  # 将True/False转为'true'/'false'
    plot_str = str(plot).lower()
    model_name = f"{model_name}_save{save_str}_plot{plot_str}" # 将布尔值转为字符串

    '''
    Test model agent in env.
    '''
    # Set a fixed start time
    if isinstance(env, Wrapper):
        env.unwrapped.random_start_time = False
        env.unwrapped.start_time = start_time_tests[0]
        env.unwrapped.max_episode_length = episode_length_test
        env.unwrapped.warmup_period = warmup_period_test
    else:
        env.random_start_time = False
        env.start_time = start_time_tests[0]
        env.max_episode_length = episode_length_test
        env.warmup_period = warmup_period_test

    # Reset environment
    obs, _ = env.reset()

    # Simulation loop
    done = False
    observations = [obs]
    actions = []
    rewards = []
    print('Simulating...')
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        done = (terminated or truncated)

    kpis = env.get_kpis()

    if save_to_file:
        os.makedirs(os.path.join(log_dir, 'results_tests_' + model_name + '_' + env.scenario['electricity_price']),
                    exist_ok=True)
        with open(os.path.join(log_dir, 'results_tests_' + model_name + '_' + env.scenario['electricity_price'],
                               'kpis_{}.json'.format(str(int(start_time_tests[0] / 3600 / 24)))), 'w') as f:
            json.dump(kpis, f)

    if plot:
        plot_results(env, rewards, save_to_file=save_to_file, log_dir=log_dir, model_name=model_name)

    # Back to random start time, just in case we're testing in the loop
    if isinstance(env, Wrapper):
        env.unwrapped.random_start_time = True
    else:
        env.random_start_time = True


    return observations, actions, rewards, kpis


if __name__ == "__main__":
    render = False
    plot = True  # Plot does not work together with render

    # 训练模型
    env, model, start_time_tests, log_dir = train_RL(algorithm='DDPG', mode='train', training_timesteps=1e5,
                                                     render=render,expert_traj=os.path.join('trajectories', 'expert_traj_cont_28.npz'))

    # 测试参数
    warmup_period_test = 1 * 24 * 3600
    episode_length_test = 14 * 24 * 3600
    save_to_file = True

    # 执行测试
    obs, act, rew, kpi = run_event(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, save_to_file, plot)

    # 输出结果
    print("KPIs:", kpi)
















