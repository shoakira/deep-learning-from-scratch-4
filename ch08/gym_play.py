# CartPoleのシミュレーション環境実行スクリプト
#
# このスクリプト（gym_play.py）はOpenAI Gym（現Gymnasium）を使用して、
# CartPole強化学習環境でランダムな行動を実行するシンプルなシミュレーションです。
#
# 概要:
# CartPoleは「倒立振子」問題として知られる強化学習の基本的なタスクです。
# 動くカートの上にポールを立てて、カートを左右に動かすことでポールをバランスさせるのが目標です。
#
# 入力項目:
# - アクション: 0（左に動かす）または1（右に動かす）のランダム選択
#
# 環境情報:
# - 環境名: CartPole-v1
# - 状態空間: 4次元ベクトル（カートの位置、速度、ポールの角度、角速度）
# - 行動空間: 離散的（左または右への力）
# - 報酬: 各ステップでポールが倒れなければ+1
# - 終了条件: ポールが一定角度以上傾くか、カートが一定距離以上移動した場合

import numpy as np
import gym


try:
    # 新しいGym API用 - render_modeを指定
    env = gym.make('CartPole-v1', render_mode="human")  # v1にアップグレード推奨
except TypeError:
    # 古いGym API用
    env = gym.make('CartPole-v1')  # v1にアップグレード推奨

state = env.reset()
# 新しいAPIでは(state, info)のタプルを返す場合に対応
if isinstance(state, tuple):
    state = state[0]
done = False

while not done:
    # 新APIではrender_mode="human"を指定した場合、自動的にレンダリングされる
    if not hasattr(env, 'render_mode') or env.render_mode is None:
        env.render()  # 古いAPIでのみ必要
    
    action = np.random.choice([0, 1])
    
    try:
        # 新しいGym API (v0.26.0以降)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # 両方のフラグを考慮
    except ValueError:
        # 古いGym API
        next_state, reward, done, info = env.step(action)
    
    state = next_state

env.close()