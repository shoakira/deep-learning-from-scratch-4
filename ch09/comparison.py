import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt
import copy
import random
from collections import deque  # DQN用に追加


# DQN用のReplayBufferクラス
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


# DQN用のQNetクラス
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


# Policy Network (共通): 方策（行動選択確率）を表現するニューラルネットワーク
class PolicyNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        # 隠れ層: 128ユニットの全結合層
        self.l1 = L.Linear(128)
        # 出力層: 行動数（例：CartPoleでは2）のユニットを持つ全結合層
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        # 隠れ層の活性化関数としてReLUを使用
        x = F.relu(self.l1(x))
        # 出力層の活性化関数としてSoftmaxを使用し、行動選択確率を計算
        x = F.softmax(self.l2(x))
        return x


# Value Network (共通): 状態価値関数を表現するニューラルネットワーク
class ValueNet(Model):
    def __init__(self):
        super().__init__()
        # 隠れ層: 128ユニットの全結合層
        self.l1 = L.Linear(128)
        # 出力層: 1ユニット（状態価値）の全結合層
        self.l2 = L.Linear(1)

    def forward(self, x):
        # 隠れ層の活性化関数としてReLUを使用
        x = F.relu(self.l1(x))
        # 出力層はそのまま（状態価値）
        x = self.l2(x)
        return x


# 新たに追加するDQNエージェントクラス
class DQNAgent:
    """DQNによる強化学習エージェント"""
    def __init__(self, lr=0.0005, gamma=0.98, action_size=2):
        # ハイパーパラメータ設定
        self.gamma = gamma  # 割引率
        self.lr = lr  # 学習率
        self.buffer_size = 10000  # リプレイバッファのサイズ
        self.batch_size = 32  # バッチサイズ
        self.action_size = action_size  # 行動の種類数（CartPoleでは左右の2種類）

        # エプシロンを初期値から徐々に減少させる
        self.epsilon_start = 1.0  # 初期値を1.0に変更
        self.epsilon_end = 0.05   # 最終的に0.05まで減少
        self.epsilon_decay = 0.999  # 0.995から緩やかに
        self.epsilon = self.epsilon_start

        # 各コンポーネントの初期化
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)  # メインのQ-Network
        self.qnet_target = QNet(self.action_size)  # ターゲットQ-Network
        # 初期時点でターゲットネットワークを同期
        self.qnet_target = copy.deepcopy(self.qnet)
        self.optimizer = optimizers.Adam(self.lr)  # オプティマイザ
        self.optimizer.setup(self.qnet)
        
        # 他のエージェントと統一するためのステップカウンター
        self.total_steps = 0

    def get_action(self, state):
        """状態に基づいて行動を選択（ε-greedy法）"""
        # Gym環境からの状態がタプルで返される場合、最初の要素のみを使用
        if isinstance(state, tuple):
            state = state[0]
            
        if np.random.rand() < self.epsilon:
            # εの確率でランダム行動
            action = np.random.choice(self.action_size)
            # 他のエージェントと同じインターフェースのために、ダミーの確率値を返す
            prob = 1.0 / self.action_size  # 一様分布の確率
            return action, prob
        else:
            # 1-εの確率でQ値最大の行動
            state = state[np.newaxis, :]  # バッチ次元を追加
            qs = self.qnet(state)  # Q値を計算
            action = qs.data.argmax()  # Q値が最大の行動を返す
            # 他のエージェントと同じインターフェースのために、ダミーの確率値を返す
            prob = 1.0  # 確定的な選択を表す確率
            return action, prob

    def update(self, state, action, reward, next_state, done):
        """経験に基づいてQ関数を更新"""
        # ステップカウンタの更新
        self.total_steps += 1
        
        # エプシロン減衰
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
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
        
    def reset(self):
        """エピソード間の処理（何もしない）"""
        pass

    def add(self, reward, prob):
        """REINFORCEなどと互換性を持たせるためのダミーメソッド"""
        pass


# REINFORCE Agent: REINFORCEアルゴリズムの実装
class REINFORCEAgent:
    def __init__(self, lr=0.0002, gamma=0.98, action_size=2):
        # 割引率
        self.gamma = gamma
        # 学習率
        self.lr = lr
        # 行動の数
        self.action_size = action_size
        # メモリ（報酬と行動確率を保存）
        self.memory = []
        # 方策ネットワーク
        self.pi = PolicyNet(self.action_size)
        # オプティマイザ
        self.optimizer = optimizers.Adam(self.lr)
        # オプティマイザをセットアップ
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        # Gym環境からの状態がタプルで返される場合、最初の要素のみを使用
        if isinstance(state, tuple):
            state = state[0]
        # バッチ次元を追加
        state = state[np.newaxis, :]
        # 方策ネットワークを使用して、各行動の確率を取得
        probs = self.pi(state)
        # 確率の形状を調整
        probs = probs[0]
        # 確率に基づいて行動を選択
        action = np.random.choice(len(probs), p=probs.data)
        # 行動と、その選択確率を返す
        return action, probs[action]

    def add(self, reward, prob):
        # エピソード中の報酬と選択確率をメモリに追加
        self.memory.append((reward, prob))

    def update(self):
        self.pi.cleargrads()
        G, loss = 0, 0
        rewards = []
        probs = []
        for reward, prob in self.memory:
            rewards.append(reward)
            probs.append(prob)
        
        # 各ステップでの正しいリターンを計算
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)  # 元の順序に戻す
        
        # 各ステップの報酬に対応する損失を計算
        for prob, G in zip(probs, returns):
            loss += -F.log(prob) * G
        # 勾配を計算
        loss.backward()
        # パラメータを更新
        self.optimizer.update()
        # メモリをクリア
        self.memory = []
        
    def reset(self):
        # メモリをリセット
        self.memory = []

# Simple PG Agent (modified from simple_pg.py): Simple Policy Gradientアルゴリズムの実装
class SimplePGAgent:
    def __init__(self, lr=0.0002, gamma=0.98, action_size=2):
        # 割引率
        self.gamma = gamma
        # 学習率
        self.lr = lr
        # 行動の数
        self.action_size = action_size
        # メモリ（報酬と行動確率を保存）
        self.memory = []
        # 方策ネットワーク
        self.pi = PolicyNet(self.action_size)
        # オプティマイザ
        self.optimizer = optimizers.Adam(self.lr)
        # オプティマイザをセットアップ
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        # Gym環境からの状態がタプルで返される場合、最初の要素のみを使用
        if isinstance(state, tuple):
            state = state[0]
        # バッチ次元を追加
        state = state[np.newaxis, :]
        # 方策ネットワークを使用して、各行動の確率を取得
        probs = self.pi(state)
        # 確率の形状を調整
        probs = probs[0]
        # 確率に基づいて行動を選択
        action = np.random.choice(len(probs), p=probs.data)
        # 行動と、その選択確率を返す
        return action, probs[action]

    def add(self, reward, prob):
        # エピソード中の報酬と選択確率をメモリに追加
        self.memory.append((reward, prob))

    def update(self):
        # パラメータの勾配をリセット
        self.pi.cleargrads()
        # 累積報酬を初期化
        G = 0
        loss = 0
        # エピソードを逆順に処理
        for reward, prob in reversed(self.memory):
            # 累積報酬を計算
            G = reward + self.gamma * G
        # エピソードを順順に処理
        for reward, prob in self.memory:
            # 損失を計算（方策勾配法）
             loss += -F.log(prob) * G
        # 勾配を計算
        loss.backward()
        # パラメータを更新
        self.optimizer.update()
        # メモリをクリア
        self.memory = []
    def reset(self):
        # メモリをリセット
        self.memory = []


# Actor-Critic Agent: Actor-Criticアルゴリズムの実装
class ActorCriticAgent:
    def __init__(self, lr_pi=0.0002, lr_v=0.0005, gamma=0.98, action_size=2):
        # 割引率
        self.gamma = gamma
        # 方策ネットワークの学習率
        self.lr_pi = lr_pi
        # 価値ネットワークの学習率
        self.lr_v = lr_v
        # 行動の数
        self.action_size = action_size
        # 方策ネットワーク
        self.pi = PolicyNet()
        # 価値ネットワーク
        self.v = ValueNet()
        # 方策ネットワークのオプティマイザ
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        # 価値ネットワークのオプティマイザ
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        # Gym環境からの状態がタプルで返される場合、最初の要素のみを使用
        if isinstance(state, tuple):
            state = state[0]
        # バッチ次元を追加
        state = state[np.newaxis, :]
        # 方策ネットワークを使用して、各行動の確率を取得
        probs = self.pi(state)
        # 確率の形状を調整
        probs = probs[0]
        # 確率に基づいて行動を選択
        action = np.random.choice(len(probs), p=probs.data)
        # 行動と、その選択確率を返す
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        # Gym環境からの状態がタプルで返される場合、最初の要素のみを使用
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        # バッチ次元を追加
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]
        # 価値ネットワークを更新する為のターゲットを計算
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        # ターゲットの勾配計算を停止
        target.unchain()
        # 現在の状態価値を計算
        v = self.v(state)
        # 価値ネットワークの損失を計算
        loss_v = F.mean_squared_error(v, target)
        # TD誤差を計算
        delta = target - v
        # TD誤差の勾配計算を停止
        delta.unchain()
        # 方策ネットワークの損失を計算
        loss_pi = -F.log(action_prob) * delta
        # 勾配をリセット
        self.v.cleargrads()
        self.pi.cleargrads()
        # 価値ネットワークの勾配を計算
        loss_v.backward()
        # 方策ネットワークの勾配を計算
        loss_pi.backward()
        # パラメータを更新
        self.optimizer_v.update()
        self.optimizer_pi.update()
    def reset(self):
        # 何もしない
        pass
    


# Improved Actor-Critic Agent: 安定化技術が追加された改良版Actor-Criticアルゴリズムの実装
class ImprovedActorCriticAgent:
    def __init__(self, lr_pi=0.0002, lr_v=0.0005, gamma=0.98, action_size=2):
        # 基本パラメータ設定
        self.gamma = gamma
        self.lr_pi = lr_pi
        self.lr_v = lr_v
        self.action_size = action_size
        
        # ネットワーク初期化
        self.pi = PolicyNet()
        self.v = ValueNet()
        
        # オプティマイザ設定
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)
        
        # エントロピー正則化係数（小さくする）
        self.entropy_coef = 0.001  # 0.01から0.001に変更
        
        # ターゲットネットワークを価値ネットワークで初期化
        self.v_target = ValueNet()
        # 重要: 初期状態でターゲットネットワークを同期
        self.v_target = copy.deepcopy(self.v)
        
        self.sync_interval = 100  # 20から100に変更
        self.total_steps = 0  # 累積ステップ数（リセットしない）
        self.episode_steps = 0  # エピソード内ステップ数（リセットする）
        
        # 経験バッファ
        self.buffer = []
        self.batch_size = 32
        self.buffer_max_size = 5000  # 10000から5000に変更

    def set_lr(self, lr_pi=None, lr_v=None):
        # 学習率を更新し、オプティマイザを再設定する（学習率スケジューリング用）
        if lr_pi is not None:
            self.lr_pi = lr_pi
            self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        if lr_v is not None:
            self.lr_v = lr_v
            self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        # Gym環境からの状態がタプルで返される場合、最初の要素のみを使用
        if isinstance(state, tuple):
            state = state[0]
        # バッチ次元を追加
        state = state[np.newaxis, :]
        # 方策ネットワークを使用して、各行動の確率を取得
        probs = self.pi(state)
        # 確率の形状を調整
        probs = probs[0]
        # 確率に基づいて行動を選択
        action = np.random.choice(len(probs), p=probs.data)
        # 行動と、その選択確率を返す
        return action, probs[action]

    def store_experience(self, state, action_prob, reward, next_state, done):
        # 経験をバッファに保存
        self.buffer.append((state, action_prob, reward, next_state, done))
        # バッファサイズが大きくなりすぎたら古いデータを削除
        if len(self.buffer) > 10000:
            self.buffer.pop(0)

    def batch_update(self):
        # バッチサイズチェック
        if len(self.buffer) < self.batch_size:
            return
        
        # 個別処理に変更（バッチ処理の複雑さを回避）
        batch = random.sample(self.buffer, self.batch_size)
        
        # グラディエントの累積ではなく、損失の累積に変更
        total_loss_v = 0
        total_loss_pi = 0
        
        self.v.cleargrads()
        self.pi.cleargrads()
        
        # サンプルごとに個別に損失を計算して累積
        for state, action_prob, reward, next_state, done in batch:
            # 通常の更新と同じロジックで個別計算
            if isinstance(state, tuple):
                state = state[0]
            if isinstance(next_state, tuple):
                next_state = next_state[0]
                
            state = state[np.newaxis, :]
            next_state = next_state[np.newaxis, :]
            
            # 価値計算
            target = reward + self.gamma * self.v_target(next_state) * (1 - done)
            target.unchain()
            v = self.v(state)
            loss_v = F.mean_squared_error(v, target)
            
            # TD誤差計算
            delta = target - v
            delta.unchain()
            
            # エントロピー計算
            pi_out = self.pi(state)
            entropy = -F.sum(pi_out * F.log(pi_out + 1e-8)) * self.entropy_coef
            
            # 方策損失
            loss_pi = -F.log(action_prob) * delta + entropy
            
            # 損失を累積（バックワードはまだ行わない）
            total_loss_v += loss_v
            total_loss_pi += loss_pi
        
        # バッチ全体の平均損失
        total_loss_v /= self.batch_size
        total_loss_pi /= self.batch_size
        
        # 一度だけバックワードを実行
        total_loss_v.backward()
        total_loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()

    def update(self, state, action_prob, reward, next_state, done):
        # 前処理部分は同じ
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]
        
        # ステップ数を更新（両方のカウンターを更新）
        self.total_steps += 1
        self.episode_steps += 1
        
        # 累積ステップ数に基づいてターゲットネットワークを同期
        if self.total_steps % self.sync_interval == 0:
            self.v_target = copy.deepcopy(self.v)
        
        # 以下は同じ
        target = reward + self.gamma * self.v_target(next_state) * (1 - done)
        target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(v, target)
        delta = target - v
        delta.unchain()
        
        # エントロピー計算（自己エントロピーのみ簡単に計算）
        pi_out = self.pi(state)
        current_entropy_coef = max(0.0005, self.entropy_coef * (0.99995 ** self.total_steps))
        entropy = -F.sum(pi_out * F.log(pi_out + 1e-8)) * current_entropy_coef
        
        loss_pi = -F.log(action_prob) * delta + entropy
        
        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()

    def reset(self):
        # エピソード内ステップ数のみリセット
        self.episode_steps = 0
        # 注意: total_stepsとバッファはリセットしない！

# Training Function: エージェントの学習を行う関数
def train(agent, env, episodes, decay_rate=0.9995):
    reward_history = []
    
    for episode in range(episodes):
        # ImprovedActorCriticAgentの場合の学習率スケジューリング
        if isinstance(agent, ImprovedActorCriticAgent) and episode > 0 and episode % 100 == 0:
            new_lr_pi = agent.lr_pi * decay_rate
            new_lr_v = agent.lr_v * decay_rate
            agent.set_lr(lr_pi=new_lr_pi, lr_v=new_lr_v)
            
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        total_reward = 0
        
        while not done:
            action, prob = agent.get_action(state)
            
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, info = env.step(action)
                
            if isinstance(agent, (REINFORCEAgent, SimplePGAgent)):
                agent.add(reward, prob)
            elif isinstance(agent, ActorCriticAgent):
                agent.update(state, prob, reward, next_state, done)
            # train関数内のDQNの処理部分
            elif isinstance(agent, DQNAgent):
                # DQNの更新
                agent.update(state, action, reward, next_state, done)
                # ステップ数に基づいてターゲットネットワークを同期（定期的に）
                if agent.total_steps % 20 == 0:  # 20ステップごとに同期
                    agent.sync_qnet()
            elif isinstance(agent, ImprovedActorCriticAgent):
                # 経験バッファに保存
                agent.store_experience(state, prob, reward, next_state, done)
                # 通常の更新
                agent.update(state, prob, reward, next_state, done)
                # バッチ更新（累積ステップ数に基づく）
                if agent.total_steps % 10 == 0:  # 10ステップごと
                    agent.batch_update()
                    
            state = next_state
            total_reward += reward
            
        if isinstance(agent, (REINFORCEAgent, SimplePGAgent)):
            agent.update()
            
        reward_history.append(total_reward)
        if episode % 100 == 0:
            print(f"episode: {episode}, total reward: {total_reward:.1f}")
            
        # エージェントごとに適切なリセット処理
        if isinstance(agent, ImprovedActorCriticAgent):
            # ImprovedActorCriticAgentはエピソード内ステップ数のみリセット
            agent.reset()
        else:
            # その他のエージェントは通常どおりリセット
            agent.reset()
            
    return reward_history


# DQN用の訓練関数
def train_dqn(agent, env, episodes, sync_interval=20):
    reward_history = []
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action, _ = agent.get_action(state)  # DQNはprobを返さないのでアンダースコア
            
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, info = env.step(action)
                
            # DQNの更新
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step += 1
            
            # ターゲットネットワークの同期
            if step % sync_interval == 0:
                agent.sync_qnet()
                
        reward_history.append(total_reward)
        if episode % 100 == 0:
            print(f"episode: {episode}, total reward: {total_reward:.1f}")
            
    return reward_history

# 移動平均の計算関数
def moving_average(data, window_size):
    """指定されたウィンドウサイズでの移動平均を計算"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# メイン関数修正
def main(episodes=1000):
    """各アルゴリズムを実行して比較"""
    
    # 環境の作成
    try:
        env = gym.make('CartPole-v1', render_mode=None)
    except TypeError:
        env = gym.make('CartPole-v1')
    
    # 各エージェントのインスタンス化
    reinforce_agent = REINFORCEAgent()
    simple_pg_agent = SimplePGAgent()
    actor_critic_agent = ActorCriticAgent()
    improved_ac_agent = ImprovedActorCriticAgent()
    dqn_agent = DQNAgent()  # DQNエージェントを追加
    
    # 各エージェントのトレーニング
    print("トレーニング開始: REINFORCE")
    reinforce_rewards = train(reinforce_agent, env, episodes)
    
    print("トレーニング開始: Simple PG")
    simple_pg_rewards = train(simple_pg_agent, env, episodes)
    
    print("トレーニング開始: Actor-Critic")
    ac_rewards = train(actor_critic_agent, env, episodes)
    
    print("トレーニング開始: Improved Actor-Critic")
    improved_ac_rewards = train(improved_ac_agent, env, episodes)
    
    print("トレーニング開始: DQN")
    # train_dqnではなくtrainを使用する（競合を避ける）
    dqn_rewards = train(dqn_agent, env, episodes)
    
    # 結果のプロット
    plt.figure(figsize=(12, 8))
    
    algorithms = {
        "REINFORCE": reinforce_rewards,
        "Simple-PG": simple_pg_rewards,
        "Actor-Critic": ac_rewards,
        "Improved Actor-Critic": improved_ac_rewards,
        "DQN": dqn_rewards  # DQNの結果を追加
    }
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # 色を1つ追加
    window_size = 50
    
    # 各アルゴリズムの移動平均をプロット
    for (name, rewards), color in zip(algorithms.items(), colors):
        if len(rewards) > window_size:
            smoothed = moving_average(rewards, window_size)
            plt.plot(np.arange(window_size-1, len(rewards)), smoothed, 
                     label=name, linewidth=2, color=color)
            
            # 最終的な性能評価
            if len(rewards) >= 100:
                last_100_avg = np.mean(rewards[-100:])
                print(f"{name} - 最後の100エピソードの平均: {last_100_avg:.1f}")
    
    plt.title('Comparison of Reinforcement Learning Algorithms')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('algorithm_comparison.png')
    plt.show()

# スクリプトを実行
if __name__ == "__main__":
    main(episodes=2000)