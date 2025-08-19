import requests
from collections import OrderedDict
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper, NormalizedActionWrapper
import numpy as np
import os
import json
import utilities
from test_and_plot import plot_results


# url for the BOPTEST service
url = 'http://127.0.0.1:80'

algorithm = 'PI_controller'
log_dir = os.path.join(utilities.get_root_path(), 'testing',
                       'agents', '{}_logdir'.format(algorithm))
log_dir = log_dir.replace('+', '')


class BoptestGymEnvCustomReward(BoptestGymEnv):
    '''Define a custom reward for this building

    '''

    def get_reward(self):
        '''Custom reward function

        '''

        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']

        # Calculate objective integrand function at this point
        # objective_integrand = kpis['cost_tot']*12.*16 + 100 * kpis['tdis_tot']
        objective_integrand = kpis['cost_tot'] + 10 * (kpis['tdis_tot'])

        # Compute reward
        reward = -(objective_integrand - self.objective_integrand)

        self.objective_integrand = objective_integrand

        return reward



# 实现PI控制器
class PIController:
    def __init__(self, TSet=294.15, Kp=20, Ki=1):
        self.TSet = TSet
        self.Kp = Kp
        self.Ki = Ki
        self.integral = 0
        self.last_temp = None

    def predict(self, obs):
        current_temp = obs[1]
        error = self.TSet - current_temp
        self.integral += error

        # PI控制
        action = self.Kp * error + self.Ki * self.integral
        action = np.clip(action, int(0), int(1))  # 限制到[0,1]

        return np.array([action])


if __name__ == "__main__":

    # Instantiate environment
    env = BoptestGymEnv(url=url,
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
                        scenario={'electricity_price': 'highly_dynamic'},
                        random_start_time=False,
                        start_time=(23 - 7) * 24 * 3600,
                        max_episode_length=14 * 24 * 3600,
                        warmup_period=24 * 3600,
                        step_period=900,  # 控制步长（15分钟执行一次动作）
                        log_dir=log_dir)

    env = NormalizedActionWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act=1)
    print("动作空间:", env.action_space)  # 确认是否为Box(0,1)


    model = PIController(TSet=295.15)  # 22°C

    # Reset environment
    obs, _ = env.reset()

    # Simulation loop
    done = False
    observations = [obs]
    actions = []
    rewards = []
    print('Simulating...')

    # 当室内温度低于设定值时，控制器如何决定打开加热（“action = 1”），当温度高于设定值时，控制器如何关闭加热。
    while not done:
        action = model.predict(obs)
        action = int(action)  # 强制转换为整数
        obs, reward, terminated, truncated, _ = env.step(action)  # send the action to the environment
        print(f"温度变化: {obs[1] - 273.15:.2f}°C ")
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        done = (terminated or truncated)

    kpis = env.get_kpis()
    print(kpis)

    os.makedirs(log_dir, exist_ok=True)
    start_time_tests = (23 - 7) * 24 * 3600

    os.makedirs(os.path.join(log_dir, 'results_tests_' + algorithm + '_' + env.scenario['electricity_price']),
                exist_ok=True)
    with open(os.path.join(log_dir, 'results_tests_' + algorithm + '_' + env.scenario['electricity_price'],
                           'kpis_{}.json'.format(str(int(start_time_tests / 3600 / 24)))), 'w') as f:
        json.dump(kpis, f)

    plot_results(env, rewards, save_to_file=True, log_dir=log_dir, model_name=algorithm)

    print('end')
