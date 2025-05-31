def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=128, type=int) 
    parser.add_argument("--lr", default=1e-3, type=float) 
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=1.0, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(100000), type=int) 

    # DQN specific arguments
    parser.add_argument("--input_size", type=int, default=4, help='input_size for training')
    parser.add_argument("--convergence_threshold", default=190, type=int, help='convergence threshold for rewards')
    parser.add_argument("--consecutive_episodes", default=20, type=int, help='consecutive episodes for convergence')
    parser.add_argument('--target_update_freq', type=int, default=500, help='frequency to update target network')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--buffer_size', type=int, default=100000, help='replay buffer size')
    parser.add_argument('--epsilon_start', type=float, default=0.9, help='start value of epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.02, help='end value of epsilon') 
    parser.add_argument('--epsilon_decay', type=int, default=10000, help='epsilon decay rate') 

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    return parser
