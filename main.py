import gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

""" Discrete Continous State-Space into 100 Bins """
car_pos = np.linspace(-1.20, 0.60, 100)
car_vel = np.linspace(-0.07, 0.07, 100)

def epsgreedy_action(Q,s,eps):
    """Epsilon Greedy Action"""
    if np.random.random() < eps:
        values = np.array([Q[s, a] for a in range(3)])
        action = np.argmax(values)
    else:
        action = env.action_space.sample()
    return action

def discritise_state(obs):
    """ Binning the State-Space"""
    x, xdot = obs
    x = int(np.digitize(x, car_pos))
    xdot = int(np.digitize(xdot, car_vel))
    return (x,xdot)

def build_states():
    """ Create a Q-Table for all possible State-Space Actions"""
    states = []
    for i in range(len(car_pos+1)):
        for j in range(len(car_vel+1)):
            states.append((i,j))
    return(states)

def plot_performance(s_avgepisodes, s_avgrewards, l_avgepisodes, l_avgrewards):

    plt.figure(1)
    plt.plot(s_avgepisodes, label='Sarsa(lambda)')
    plt.plot(l_avgepisodes, label='Q(lambda)')
    plt.xlabel('n_episodes')
    plt.ylabel('Avg_Episodes')
    plt.legend()
    #plt.show()
    plt.title('Performance: Avg_Episodes/n_Episodes')
    plt.savefig('P1.png')

    plt.figure(2)
    plt.plot(s_avgrewards, label='Sarsa(lambda)')
    plt.plot(l_avgrewards, label='Q(lambda)')
    plt.xlabel('Avg_Rewards-Episode')
    plt.ylabel('n_episodes')
    plt.legend()
    #plt.show()
    plt.title('Performance: Avg_Rewards-Episode/ n_Episodes')
    plt.savefig('P2.png')

def slambda (env, states, n_episodes, epsilon, alpha, gamma, _lambda):
    """ Sarsa-lambda Implementation"""
    Q = {}
    e = {}
    for s in states:
        for a in range(3):
            Q[s,a] = 0
            e[s,a] = 0

    rewards =[]
    episodes =[]
    avg_rewards =[]
    avg_episodes =[]
    
    for i in range(n_episodes):
        print ('Sarsa(lambda) Episode = %d of %d'%(i,n_ep))
        obs = env.reset()
        s = discritise_state(obs)
        a = epsgreedy_action(Q,s,epsilon)
        c_episode = 0
        c_reward =0
        done = False

        while not done:
            if i%1000 == 0:
                env.render()
                pass

            _obs, reward, done, _ = env.step(a)
            _s = discritise_state(_obs)
            _a = epsgreedy_action(Q,_s,epsilon)

            #print (_s, reward)
            delta = reward + gamma * np.max(Q[_s,_a]) - Q[s,a]
            e[s,a] += 1

            for s in states:
                for a in range(3):
                    Q[s,a] +=  alpha * delta * e[s,a]
                    e[s,a] *= _lambda * gamma
            
            s, a = _s, _a

            c_episode += 1
            c_reward += reward

            if done:
                rewards.append(c_reward)
                episodes.append(c_episode)

        avg_rewards.append(np.mean(rewards))
        avg_episodes.append(np.mean(episodes))

    
    return(avg_episodes, avg_rewards)

def qlambda (env, states, n_episodes, epsilon, alpha, gamma, _lambda):
    """ Q-lambda Implementation"""
    Q = {}
    e = {}
    for s in states:
        for a in range(3):
            Q[s,a] = 0
            e[s,a] = 0

    rewards =[]
    episodes =[]
    avg_rewards =[]
    avg_episodes =[]
    
    for i in range(n_episodes):
        print ('Q(lambda) Episode = %d of %d'%(i,n_ep))
        obs = env.reset()
        s = discritise_state(obs)
        a = epsgreedy_action(Q,s,epsilon)
        c_episode = 0
        c_reward = 0
        done = False

        while not done:
            if i%1000 == 0:
                env.render()
                pass

            _obs, reward, done, _ = env.step(a)
            _s = discritise_state(_obs)
            _a = epsgreedy_action(Q,_s,epsilon)
            
            _a_ = np.argmax(np.array([Q[_s, a] for a in range(3)]))

            if _a is _a_:
                _a_ = _a

            delta = reward + gamma * np.max(Q[_s,_a]) - Q[s,a]
            e[s,a] += 1

            for s in states:
                for a in range(3):
                    Q[s,a] +=  alpha * delta * e[s,a]
                    if _a is _a_:
                        e[s,a] *= _lambda * gamma
                    else:
                        e[s,a] = 0
            
            s, a = _s, _a

            c_episode += 1
            c_reward += reward

            if done:
                rewards.append(c_reward)
                episodes.append(c_episode)
    
        avg_rewards.append(np.mean(rewards))
        avg_episodes.append(np.mean(episodes))
    
    env.close()
    return(avg_episodes, avg_rewards)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    states = build_states()
    n_ep = 2500

    s_avgepisodes, s_avgrewards = slambda(env, states, n_ep, 1, 0.05, 0.8, 0.9)
    l_avgepisodes, l_avgrewards = qlambda(env, states, n_ep, 1, 0.05, 0.8, 0.9)

    plot_performance(s_avgepisodes, s_avgrewards, l_avgepisodes, l_avgrewards)