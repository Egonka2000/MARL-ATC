from plugins.atc_gym_stable_baselines import Agent

def init_plugin():
    train_mode = True
    agents = ["{}".format(r) for r in range(2)]
    agent = Agent(train_mode, agents)

    config = {
        'plugin_name': 'test',
        'plugin_type': 'sim',
        'update_interval': 12.0,
        'update': agent.update_plugin
    }

    return config