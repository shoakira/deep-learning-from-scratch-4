if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        # 新しいGym APIではresetがタプル(state, info)を返す場合がある
        if isinstance(state, tuple):
            state = state[0]
            
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []


episodes = 3000
try:
    # 新しいGym API用 - v1にアップグレード
    env = gym.make('CartPole-v1')
except TypeError:
    # 古いGym API用
    env = gym.make('CartPole-v1')  # v1にアップグレード

agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()
    # 新しいAPIでは(state, info)のタプルを返す場合に対応
    if isinstance(state, tuple):
        state = state[0]
    done = False
    sum_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        
        # 新しいGym APIに対応
        try:
            # 新しいGym API (v0.26.0以降)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except ValueError:
            # 古いGym API
            next_state, reward, done, info = env.step(action)
        
        # 次の状態がタプルの場合も対応
        if isinstance(next_state, tuple):
            next_state = next_state[0]

        agent.add(reward, prob)
        state = next_state
        sum_reward += reward

    agent.update()

    reward_history.append(sum_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, sum_reward))


# plot
from common.utils import plot_total_reward
import matplotlib.pyplot as plt
import numpy as np

# 移動平均の計算関数
def moving_average(data, window_size):
    """指定されたウィンドウサイズでの移動平均を計算"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# グラフ描画設定
plt.figure(figsize=(10, 6))

# 原データ（透明度を下げて背景に表示）
plt.plot(reward_history, alpha=0.2, color='gray', label='各エピソードの報酬')

# 移動平均を計算して表示
window_size = 50  # 平均化するウィンドウサイズ
if len(reward_history) > window_size:
    smoothed_rewards = moving_average(reward_history, window_size)
    plt.plot(np.arange(window_size-1, len(reward_history)), smoothed_rewards, 
             linewidth=2, color='blue', label=f'{window_size}エピソード移動平均')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning curve of REINFORCE algorithm')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 平均報酬を表示（最後の100エピソードの平均）
if len(reward_history) >= 100:
    last_100_avg = np.mean(reward_history[-100:])
    print(f"最後の100エピソードの平均報酬: {last_100_avg:.1f}")