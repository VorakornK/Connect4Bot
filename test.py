from module import *

def RandomVsAgent(agent, random_agent, env, AI_number):
    env.reset()
    done = False

    while not done:
        state = env.get_state()

        if env.current_player == AI_number:
            action = agent.select_action(state, env, False)
        else:
            action = random_agent.select_action(env)

        next_state, _, done = env.step(action)
        
    winner = env.get_winner()
    return winner == AI_number

def TestCase(agent, num:int = 1000):
    env = ConnectFourEnvironment()
    random_agent = RandomAgent(action_size=7)
    AI_win_first, AI_win_second = 0, 0
    for i in range(num):
        if i % 2 == 0:
            AI_win_first += RandomVsAgent(agent, random_agent, env, 1)
        else:
            AI_win_second += RandomVsAgent(agent, random_agent, env, 2)
    return 2 * AI_win_first / num, 2 * AI_win_second / num

