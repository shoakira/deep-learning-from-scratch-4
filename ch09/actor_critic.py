if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gym
# commonをdezeroに戻す
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class PolicyNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)
        return x


class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet()
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        # タプルの場合は最初の要素を使用する
        if isinstance(state, tuple):
            state = state[0]
            
        state = state[np.newaxis, :]  # add batch axis
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        # タプルの場合は最初の要素を使用する
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
            
        state = state[np.newaxis, :]  # add batch axis
        next_state = next_state[np.newaxis, :]

        # ========== (1) Update V network ===========
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(v, target)

        # ========== (2) Update pi network ===========
        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()


episodes = 1000
env = gym.make('CartPole-v1')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

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

        agent.update(state, prob, reward, next_state, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

# 改良版プロット（移動平均を使用）
import matplotlib.pyplot as plt
import numpy as np

# 移動平均の計算関数
def moving_average(data, window_size):
    """Calculate moving average with specified window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# グラフ描画設定
plt.figure(figsize=(10, 6))

# 原データ（透明度を下げて背景に表示）
plt.plot(reward_history, alpha=0.2, color='gray', label='Raw Rewards')

# 移動平均を計算して表示
window_size = 50  # 平均化するウィンドウサイズ
if len(reward_history) > window_size:
    smoothed_rewards = moving_average(reward_history, window_size)
    plt.plot(np.arange(window_size-1, len(reward_history)), smoothed_rewards, 
             linewidth=2, color='blue', label=f'Moving Average ({window_size} episodes)')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Actor-Critic Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 平均報酬を表示（最後の100エピソードの平均）
if len(reward_history) >= 100:
    last_100_avg = np.mean(reward_history[-100:])
    print(f"Average reward over last 100 episodes: {last_100_avg:.1f}")

# === 学習したエージェントのデモンストレーション ===
# プレイ用の環境（可視化あり）
try:
    # 新しいGym API用
    play_env = gym.make('CartPole-v1', render_mode="human")  # 人間が見るためのモード
except TypeError:
    # 古いGym API用
    play_env = gym.make('CartPole-v1')

agent.epsilon = 0  # 評価時は常に最良の行動を選択（greedy policy）
state = play_env.reset()
if isinstance(state, tuple):
    state = state[0]
done = False
total_reward = 0

# 学習したエージェントでのシミュレーション実行
while not done:
    # 修正前: action = agent.get_action(state)
    # 修正後: アクションと確率を適切に分離
    action, prob = agent.get_action(state)
    
    # 新しいGym APIに対応
    try:
        # 新しいGym API (v0.26.0以降)
        next_state, reward, terminated, truncated, info = play_env.step(action)
        done = terminated or truncated
    except ValueError:
        # 古いGym API
        next_state, reward, done, info = play_env.step(action)
    state = next_state
    total_reward += reward
    
    # レンダリング処理
    try:
        if not hasattr(play_env, 'render_mode') or play_env.render_mode is None:
            play_env.render()  # 古いAPIの場合は明示的にレンダリング
    except Exception:
        pass
        
print('Total Reward:', total_reward)  # 最終スコアの表示
