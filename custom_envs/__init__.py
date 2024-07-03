from gymnasium.envs.registration import register
from custom_envs.continuous_maze import Mazes

# Maze
register(
    id='Maze-Easy-v1',
    entry_point='custom_envs.continuous_maze:Maze', 
    kwargs={'name': 'Easy', 'max_episode_steps': 200, 'render': False, 
            'x_init': Mazes['Easy']['x_init'], 'y_init': Mazes['Easy']['y_init'],
            'target': Mazes['Easy']['target'], 'seed': 0}
)