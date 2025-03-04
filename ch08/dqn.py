# DQN (Deep Q-Network) アルゴリズムの実装
# CartPole環境でDQNを使った強化学習を行うコード
#
# 主な特徴:
# - Experience Replay: 経験を保存し、ランダムにサンプリングして学習安定性を高める
# - Target Network: 学習安定化のため、定期的に同期する別のネットワークを利用
# - Deep Neural Network: Q関数を近似するためのニューラルネットワーク

import copy
from collections import deque  # 効率的なリスト操作のためのデータ構造
import random
import matplotlib.pyplot as plt  # 結果の可視化に使用
import numpy as np
import gym  # 強化学習環境
from dezero import Model  # ディープラーニングフレームワーク
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class ReplayBuffer:
    """経験再生バッファ - 学習の安定性向上のため過去の経験を保存"""
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)  # 最大サイズを超えると古いデータから削除
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """新しい経験をバッファに追加"""
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        """バッファに保存されているデータ数を返す"""
        return len(self.buffer)

    def get_batch(self):
        """バッファからランダムにバッチサイズ分のデータを取り出す"""
        data = random.sample(self.buffer, self.batch_size)

        # データを種類ごとに分けて配列化
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


class QNet(Model):
    """Q関数をニューラルネットワークで近似するクラス"""
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)  # 第1層 - 128ユニット
        self.l2 = L.Linear(128)  # 第2層 - 128ユニット
        self.l3 = L.Linear(action_size)  # 出力層 - 行動数分のユニット

    def forward(self, x):
        """順伝播計算 - 状態から各行動のQ値を計算"""
        x = F.relu(self.l1(x))  # 活性化関数ReLU
        x = F.relu(self.l2(x))
        x = self.l3(x)  # 出力層は活性化関数なし
        return x


class DQNAgent:
    """DQNによる強化学習エージェント"""
    def __init__(self):
        # ハイパーパラメータ設定
        self.gamma = 0.98  # 割引率
        self.lr = 0.0005  # 学習率
        self.epsilon = 0.1  # ε-greedy法のランダム行動確率
        self.buffer_size = 10000  # リプレイバッファのサイズ
        self.batch_size = 32  # バッチサイズ
        self.action_size = 2  # 行動の種類数（CartPoleでは左右の2種類）

        # 各コンポーネントの初期化
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)  # メインのQ-Network
        self.qnet_target = QNet(self.action_size)  # ターゲットQ-Network
        self.optimizer = optimizers.Adam(self.lr)  # オプティマイザ
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        """状態に基づいて行動を選択（ε-greedy法）"""
        if np.random.rand() < self.epsilon:
            # εの確率でランダム行動
            return np.random.choice(self.action_size)
        else:
            # 1-εの確率でQ値最大の行動
            state = state[np.newaxis, :]  # バッチ次元を追加
            qs = self.qnet(state)  # Q値を計算
            return qs.data.argmax()  # Q値が最大の行動を返す

    def update(self, state, action, reward, next_state, done):
        """経験に基づいてQ関数を更新"""
        # 経験をリプレイバッファに追加
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return  # バッファのデータが不足していれば更新しない

        # リプレイバッファからデータをサンプリング
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        
        # 現在のQ値を計算
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]  # 実際に選んだ行動のQ値

        # ターゲットQ値を計算（TD誤差の目標値）
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)  # 次の状態で最大のQ値を取得
        next_q.unchain()  # 勾配計算の停止（ターゲットは固定）
        target = reward + (1 - done) * self.gamma * next_q  # ベルマン方程式

        # 損失関数の計算と最適化
        loss = F.mean_squared_error(q, target)  # MSE損失
        self.qnet.cleargrads()  # 勾配のリセット
        loss.backward()  # 誤差逆伝播
        self.optimizer.update()  # パラメータ更新

    def sync_qnet(self):
        """ターゲットネットワークを現在のネットワークで同期"""
        self.qnet_target = copy.deepcopy(self.qnet)


# トレーニングのパラメータ
episodes = 500  # 学習エピソード数
sync_interval = 20  # ターゲットネットワーク同期間隔

# 学習用の環境初期化
try:
    # 新しいGym API用
    env = gym.make('CartPole-v1', render_mode=None)  # 学習中はレンダリング不要
except TypeError:
    # 古いGym API用
    env = gym.make('CartPole-v1')

agent = DQNAgent()  # エージェントの初期化
reward_history = []  # 報酬履歴の記録用

# 学習ループ
for episode in range(episodes):
    state = env.reset()
    # 新しいAPIでは(state, info)のタプルを返す場合があるため
    if isinstance(state, tuple):
        state = state[0]
    done = False
    total_reward = 0

    # エピソード内のステップループ
    while not done:
        action = agent.get_action(state)  # 行動選択
        
        # 新しいGym APIに対応
        try:
            # 新しいGym API (v0.26.0以降)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except ValueError:
            # 古いGym API
            next_state, reward, done, info = env.step(action)

        # エージェントの学習
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    # ターゲットネットワークの定期的な同期
    if episode % sync_interval == 0:
        agent.sync_qnet()

    # 結果の記録と表示
    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))


# === 学習結果のグラフ化 ===
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()


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
    action = agent.get_action(state)
    # 新しいGym APIに対応
    try:
        next_state, reward, terminated, truncated, info = play_env.step(action)
        done = terminated or truncated
    except ValueError:
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
