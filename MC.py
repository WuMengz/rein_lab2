import gym
import time
import numpy as np

env = gym.make("FrozenLake-v1")  # 创建环境
env.reset()

def exp_decay(eps0, decay, iteration):
    return eps0 * (decay ** iteration)

def linear_decay(eps0, decay, iteration):
    eps = max(eps0 - decay * iteration, 0.05)
    return eps

def inverse_decay(eps0, decay, iteration):
    return eps0 / (1 + decay * iteration)

def fix_decay(eps0, decay, iteration):
    return eps0

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

def policy_iteration_MC(env, gamma, fn = inverse_decay, eps0=0.5, decay=0.1, num_episodes=1000):
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
    start_time = time.time()
    pi = np.zeros(env.observation_space.n)
    iteration = 1
    while True:
        epsilon = fn(eps0, decay, iteration)
        Q = compute_qpi_MC(pi, env, gamma, epsilon, num_episodes)
        new_pi = Q.argmax(axis=1)
        if (pi != new_pi).sum() == 0: # 策略不再改变，作为收敛判定条件
            end_time = time.time()
            return new_pi, iteration, end_time - start_time  
        # print(f"iteration: {iteration}, eps: {epsilon}, change actions: {(pi != new_pi).sum()}")
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

def PI_average(env, gamma, fn = inverse_decay, eps0=0.5, decay=0.1, num_episodes=1000):
    sumits = 0
    cnt = 0
    sumtime = 0
    sumcorr = 0
    for i in range(100):
        pi, its, timeconsume = policy_iteration_MC(env, gamma, fn, eps0, decay, num_episodes)
        corr = test_pi(env, pi)
        sumcorr += corr
        sumits += its
        sumtime += timeconsume
        cnt += 1
        print(f"Round {i + 1}: average_iterations = {sumits / cnt}, average_time = {sumtime / cnt}, average_reward = {sumcorr / cnt}, time = {timeconsume}, reward = {corr}", flush=True)
    return sumits / cnt

# PI_average(env, fn = fix_decay, eps0=0.2, gamma=0.99, num_episodes=5000)
# PI_average(env, fn = exp_decay, decay=0.965, gamma=0.99, num_episodes=5000)
# PI_average(env, fn = inverse_decay, decay=0.1, gamma=0.99, num_episodes=5000)
# PI_average(env, fn = linear_decay, decay=0.01, gamma=0.99, num_episodes=5000)

# pi = policy_iteration_MC(env, gamma=0.99, num_episodes=5000)
# print(pi[1], pi[2])
# pi = pi[0]
# result = test_pi(env, pi)
# print(result)

def plot_figure():
    import matplotlib.pyplot as plt
    episodes = ["500", "1000", "2000", "3000", "4000", "5000", "10000", "20000", "30000"]
    rewards = [0.00037999999999999997, 0.00607, 0.18220000000000003, 0.5209499999999998, 0.6850999999999998, 0.7008999999999999, 0.7238999999999998, 0.7252499999999997, 0.7285499999999999]
    its = [31.85, 71.17, 108.11, 96.62, 60.74, 51.7, 20.12, 10.85, 8.06]
    act_its = [int(a) * b for a, b in zip(episodes, its)]
    seconds = [2.858803651332855, 16.446815905570983, 69.13822311878204, 100.28371448755264, 92.6004074883461, 99.46874690771102, 70.86445260763168, 65.79808039665222, 70.0034040594101]
    # total_seconds = [100 * sec for sec in seconds]

    plt.plot(episodes, its)
    plt.xlabel("episodes")
    plt.ylabel("average iterations")
    plt.show()
    plt.plot(episodes, rewards)
    plt.xlabel("episodes")
    plt.ylabel("average reward")
    plt.show()

    # 创建图形和子图对象
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # 绘制第一个数据集到第一个子图对象上
    ax1.plot(episodes, act_its, 'b-', label='average total episodes')
    # 绘制第二个数据集到第二个子图对象上
    ax2.plot(episodes, seconds, 'r-', label='average time consumed(seconds)')

    # 设置标签和标题
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('average total episode', color='b')
    ax2.set_ylabel('total time consumed(seconds)', color='r')
    # ax1.set_title('D')

    # 添加图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 显示图形
    plt.show()

        # plt.plot(episodes, act_its)
    # plt.plot(episodes, seconds)
    # plt.show()
    methods = ["inverse decay", "exp decay", "linear decay", "fix epsilon"]
    rewards = [0.7008999999999999, 0.6008499999999999, 0.7128299999999999, 0.6992700000000001]
    its = [51.7, 49.1, 48.19, 54.65]
    act_its = [int(a) * b for a, b in zip(episodes, its)]
    seconds = [99.46874690771102, 64.7895920419693, 83.60290706157684, 67.51970574617386]
    
    plt.bar(methods, its)
    plt.xlabel("decay method")
    plt.ylabel("average iterations")
    plt.show()
    plt.bar(methods, rewards)
    plt.xlabel("decay method")
    plt.ylabel("average reward")
    plt.show()


plot_figure()