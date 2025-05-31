import argparse
import gym
from argument import dqn_arguments, pg_arguments
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='gym')
# 忽略现有的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
# 忽略 old step API 警告
warnings.filterwarnings("ignore", message=".*old step API which returns one bool.*")


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')

    parser = dqn_arguments(parser)  # Use DQN arguments
    # parser = pg_arguments(parser)  # Comment out PG arguments
    args = parser.parse_args()
    return args

def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.run() 

    if args.train_dqn:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.run()


if __name__ == '__main__':
    args = parse()
    run(args)
