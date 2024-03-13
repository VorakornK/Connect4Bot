from module import *

def AgentVsAgent(agent_01, agent_test, env, AI_number):
    env.reset()
    done = False

    while not done:
        state = env.get_state()

        if env.current_player == AI_number:
            action = agent_01.select_action(state, env, 0.0)
        else:
            action = agent_test.select_action(env)

        next_state, _, done = env.step(action)
        
    winner = env.get_winner()
    return winner == AI_number

def TestCase01(agent, num:int = 1000):
    env = ConnectFourEnvironment()
    random_agent = RandomAgent(action_size=7)
    AI_win_first, AI_win_second = 0, 0
    for i in range(num):
        if i % 2 == 0:
            AI_win_first += AgentVsAgent(agent, random_agent, env, 1)
        else:
            AI_win_second += AgentVsAgent(agent, random_agent, env, 2)
    return 2 * AI_win_first / num, 2 * AI_win_second / num

def TestCase02(agent, num:int = 1000):
    env = ConnectFourEnvironment()
    one_step_agent = NStep_Agent()
    AI_win_first, AI_win_second = 0, 0
    for i in range(num):
        if i % 2 == 0:
            AI_win_first += AgentVsAgent(agent, one_step_agent, env, 1)
        else:
            AI_win_second += AgentVsAgent(agent, one_step_agent, env, 2)
    return 2 * AI_win_first / num, 2 * AI_win_second / num

