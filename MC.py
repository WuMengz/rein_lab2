import gym
import numpy as np

env = gym.make("FrozenLake-v0")  # 创建环境
env.reset()

def compute_qpi_MC(pi, env, gamma, epsilon, num_episodes=1000):
    """
    使用蒙特卡洛方法来估计动作价值函数Q_pi。
    参数：
        pi -- 在环境env中使用的确定性策略，是一个大小为状态数的numpy数组，输入状态，输出动作。
        env -- OpenAI Gym环境对象。
        gamma -- 折扣因子，一个0到1之间的浮点数。
        epsilon -- epsilon-贪心策略中的参数。
        num_episodes -- 进行采样的回合数。

    返回值：
        Q -- 动作价值函数的估计值。
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    N = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.int64)
    for _ in range(num_episodes):
        # 生成新的回合
        state = env.reset()
        episode = []
        # 对于该回合中的每个时间步
        while True:
            # 根据策略选择动作
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = pi[state]
            # 执行动作，获得新状态和回报值
            next_state, reward, done, _ = env.step(action)
            # 记录状态、动作、回报值
            episode.append((state, action, reward))
            # 如果回合结束，则退出循环
            if done:
                break
            # 转换到下一个状态
            state = next_state
        # 对于该回合中的每个状态-动作对
        G = 0
        for i in reversed(range(0, len(episode))):
            state, action, reward = episode[i]
            G = gamma * G + reward
            if not (state, action) in [(x[0], x[1]) for x in episode[:i]]:
                state = int(state)
                action = int(action)
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]

    return Q

# Qpi = compute_qpi_MC(np.ones(16), env, gamma=0.95)
# print("Qpi:\n", Qpi)

def policy_iteration_MC(env, gamma, eps0=0.5, decay=0.1, num_episodes=1000):
    """
    使用蒙特卡洛方法来实现策略迭代。
    参数：
        env -- OpenAI Gym环境对象。
        gamma -- 折扣因子，一个0到1之间的浮点数。
        eps0 -- 初始的探索概率。
        decay – 衰减速率。
        num_episodes -- 进行采样的回合数。

    返回值：
        pi -- 最终策略。
    """

    pi = np.zeros(env.observation_space.n)
    iteration = 1
    while True:
        epsilon = eps0/(1+decay*iteration)
        Q = compute_qpi_MC(pi, env, gamma, epsilon, num_episodes)
        new_pi = Q.argmax(axis=1)
        if (pi != new_pi).sum() == 0: # 策略不再改变，作为收敛判定条件
            return new_pi            
        print(f"iteration: {iteration}, eps: {epsilon}, change actions: {(pi != new_pi).sum()}")
        pi = new_pi
        iteration = iteration + 1

def test_pi(env, pi, num_episodes=1000):
    """
    测试策略。
    参数：
        env -- OpenAI Gym环境对象。
        pi -- 需要测试的策略。
        num_episodes -- 进行测试的回合数。

    返回值：
        成功到达终点的频率。
    """

    count = 0
    for e in range(num_episodes):
        ob = env.reset()
        while True:
            a = pi[ob]
            ob, rew, done, _ = env.step(a)
            if done:
                count += 1 if rew == 1 else 0
                break
    return count / num_episodes

pi = policy_iteration_MC(env, gamma=0.99, num_episodes=5000)
result = test_pi(env, pi)
print(result)
