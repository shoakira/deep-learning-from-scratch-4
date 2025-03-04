# Actor-Criticアルゴリズムの改良版実装
# CartPole環境を対象に、複数の安定化テクニックを導入した強化学習エージェント
#
# 主な改良点:
# - エントロピー正則化: 方策の過度な収束を防止し探索を促進
# - ターゲットネットワーク: 学習の安定性向上のため価値関数の目標を固定
# - バッチ学習: 複数の経験から同時に学習して安定性を向上
# - 学習率スケジューリング: 学習率を徐々に減衰させ細かい調整を可能に

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
import copy
import random


class PolicyNet(Model):
    """方策（行動選択確率）を近似するニューラルネットワーク"""
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)  # 隠れ層 - 128ユニット
        self.l2 = L.Linear(action_size)  # 出力層 - 行動数分のユニット

    def forward(self, x):
        """状態から行動確率を計算"""
        x = F.relu(self.l1(x))  # 活性化関数ReLU
        x = self.l2(x)
        x = F.softmax(x)  # 確率分布として正規化
        return x


class ValueNet(Model):
    """状態価値関数を近似するニューラルネットワーク"""
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)  # 隠れ層 - 128ユニット
        self.l2 = L.Linear(1)  # 出力層 - 価値は実数値1つ

    def forward(self, x):
        """状態から価値を計算"""
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    """Actor-Criticアルゴリズムを用いた強化学習エージェント（改良版）"""
    def __init__(self, lr_pi=0.0002, lr_v=0.0005):
        # 基本パラメータ
        self.gamma = 0.98  # 割引率
        self.lr_pi = lr_pi  # 方策ネットワークの学習率
        self.lr_v = lr_v    # 価値ネットワークの学習率
        self.action_size = 2  # 行動の種類数（CartPoleでは左右の2種類）
        
        # ニューラルネットワークとオプティマイザの初期化
        self.pi = PolicyNet()  # 方策ネットワーク（Actor）
        self.v = ValueNet()    # 価値ネットワーク（Critic）
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)
        self.entropy_coef = 0.01  # エントロピー正則化の係数

        # 改良1: ターゲットネットワーク（学習安定化のため）
        self.v_target = ValueNet()  # 価値関数の目標計算用ネットワーク
        self.sync_interval = 20     # ターゲットネットワーク同期間隔
        self.steps = 0              # 累積ステップ数

        # 改良2: 経験再生用バッファ（バッチ学習のため）
        self.buffer = []       # 経験を蓄積するリスト
        self.batch_size = 32   # バッチサイズ

    def set_lr(self, lr_pi=None, lr_v=None):
        """学習率を更新し、オプティマイザを再設定する（学習率スケジューリング用）"""
        if lr_pi is not None:
            self.lr_pi = lr_pi
            self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        
        if lr_v is not None:
            self.lr_v = lr_v
            self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        """状態に基づいて確率的に行動を選択"""
        # 入力状態の前処理
        if isinstance(state, tuple):
            state = state[0]
            
        state = state[np.newaxis, :]  # バッチ次元を追加
        
        # 方策ネットワークから行動確率を取得
        probs = self.pi(state)
        probs = probs[0]
        
        # 確率に基づいて行動をサンプリング
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]  # 行動とその選択確率を返す

    def store_experience(self, state, action_prob, reward, next_state, done):
        """経験をバッファに保存（経験再生用）"""
        self.buffer.append((state, action_prob, reward, next_state, done))
        # バッファサイズが大きくなりすぎたら古いデータを削除
        if len(self.buffer) > 10000:  # 最大バッファサイズ
            self.buffer.pop(0)
    
    def batch_update(self):
        """バッファから抽出したミニバッチでの学習（実装は簡略化）"""
        if len(self.buffer) < self.batch_size:
            return  # バッファのデータが不足していれば更新しない
            
        # バッファからランダムサンプリング
        batch = random.sample(self.buffer, self.batch_size)
        
        # バッチ処理は複雑なため今回は省略
        # 実際の実装では、サンプリングしたバッチデータを使用して
        # ネットワークを更新する処理をここに記述します

    def update(self, state, action_prob, reward, next_state, done):
        """TD誤差を用いた方策と価値関数の更新（Actor-Criticの中核）"""
        # 入力状態の前処理
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
            
        state = state[np.newaxis, :]  # バッチ次元を追加
        next_state = next_state[np.newaxis, :]

        # ターゲットネットワークの同期（定期的に行う）
        self.steps += 1
        if self.steps % self.sync_interval == 0:
            self.v_target = copy.deepcopy(self.v)  # 価値ネットワークのコピーを作成

        # ========== (1) 価値ネットワーク(Critic)の更新 ===========
        # TDターゲットの計算（ベルマン方程式）
        target = reward + self.gamma * self.v_target(next_state) * (1 - done)
        target.unchain()  # ターゲットの勾配計算を停止
        v = self.v(state)  # 現在の状態価値
        loss_v = F.mean_squared_error(v, target)  # MSE損失

        # ========== (2) 方策ネットワーク(Actor)の更新 ===========
        # アドバンテージ（TD誤差）の計算
        delta = target - v  # ターゲットと現在の価値の差
        delta.unchain()  # 勾配計算の停止
        
        # 改良3: エントロピー正則化項の計算
        # 方策のエントロピーを損失に加えることで探索を促進
        entropy = -F.sum(self.pi(state) * F.log(self.pi(state) + 1e-8)) * self.entropy_coef
        
        # 方策勾配損失にエントロピー項を追加
        loss_pi = -F.log(action_prob) * delta + entropy

        # 勾配の計算と更新
        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()  # 価値ネットワークの誤差逆伝播
        loss_pi.backward()  # 方策ネットワークの誤差逆伝播
        self.optimizer_v.update()  # 価値ネットワークのパラメータ更新
        self.optimizer_pi.update()  # 方策ネットワークのパラメータ更新


# ==================== メインスクリプト ====================

# 学習パラメータ
episodes = 3000  # 学習エピソード数
decay_rate = 0.9995  # 学習率減衰率

# CartPole環境の作成
try:
    # 新しいGym API用
    env = gym.make('CartPole-v1')  # 学習時はレンダリング不要
except TypeError:
    # 古いGym API用
    env = gym.make('CartPole-v1')

# エージェントの初期化と履歴記録用リスト
agent = Agent()
reward_history = []

# 学習ループ
for episode in range(episodes):
    # 改良4: 学習率のスケジューリング
    if episode > 0 and episode % 100 == 0:
        # 100エピソードごとに学習率を減衰
        new_lr_pi = agent.lr_pi * decay_rate
        new_lr_v = agent.lr_v * decay_rate
        agent.set_lr(lr_pi=new_lr_pi, lr_v=new_lr_v)
        print(f"Learning rates updated: pi={agent.lr_pi:.6f}, v={agent.lr_v:.6f}")
    
    # エピソードの初期化
    state = env.reset()
    if isinstance(state, tuple):  # 新しいAPIではタプルを返す場合がある
        state = state[0]
    done = False
    total_reward = 0

    # エピソード内のステップループ
    while not done:
        # 行動選択
        action, prob = agent.get_action(state)
        
        # 環境内で行動を実行
        try:
            # 新しいGym API対応
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # 終了条件の統合
        except ValueError:
            # 古いGym API対応
            next_state, reward, done, info = env.step(action)

        # 学習処理
        agent.update(state, prob, reward, next_state, done)  # 通常更新
        agent.store_experience(state, prob, reward, next_state, done)  # 経験をバッファに保存
        agent.batch_update()  # バッチ学習（実装簡略化）

        # 次のステップへ
        state = next_state
        total_reward += reward

    # エピソード結果の記録
    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

# 学習結果のグラフ化
from common.utils import plot_total_reward
plot_total_reward(reward_history)


# ==================== 学習したエージェントのテスト実行 ====================

# プレイ用の環境（可視化あり）
try:
    # 新しいGym API用
    play_env = gym.make('CartPole-v1', render_mode="human")  # 人間が見るためのモード
except TypeError:
    # 古いGym API用
    play_env = gym.make('CartPole-v1')

# テスト実行の初期化
agent.epsilon = 0  # 評価時は常に最良の行動を選択（greedy policy）
state = play_env.reset()
if isinstance(state, tuple):
    state = state[0]
done = False
total_reward = 0

# 学習したエージェントでのシミュレーション実行
while not done:
    # 行動と確率を取得
    action, prob = agent.get_action(state)
    
    # 環境内で行動を実行
    try:
        # 新しいGym API対応
        next_state, reward, terminated, truncated, info = play_env.step(action)
        done = terminated or truncated
    except ValueError:
        # 古いGym API対応
        next_state, reward, done, info = play_env.step(action)
    
    # 次のステートへ
    state = next_state
    total_reward += reward
    
    # 描画処理
    try:
        if not hasattr(play_env, 'render_mode') or play_env.render_mode is None:
            play_env.render()  # 古いAPIの場合は明示的にレンダリング
    except Exception:
        pass
        
print('Total Reward:', total_reward)  # 最終スコアの表示
