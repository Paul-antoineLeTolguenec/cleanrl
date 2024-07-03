import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt
import numpy as np
import copy


class Maze(gym.Env):
        def __init__(self, name = None, target = [0.0,-0.5],
                     max_episode_steps = 200, seed = 0, 
                        x_init = 0.0, y_init = 0.0,
                        render_mode = None,
                     render = False):
            super(Maze, self).__init__()
            self.x=0
            self.y=0
            self.dx=0
            self.dy=0
            self.dt=2e-2
            self.max_x=1
            self.max_y=1
            self.min_x=-1
            self.min_y=-1
            self.observation_space=Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.action_space=Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.walls = [(-1,-1,-1,1),(-1,-1,1,-1),(-1,1,1,1),(1,-1,1,1)]
            self.dangerous_point =[]
            self.max_episode_steps = max_episode_steps
            self.x_init= x_init
            self.y_init= y_init
            self.dx_init=0
            self.dy_init=0
            self.name = name
            self.target = target
            self.episode_length = 0 
            self.episode_reward = 0
            self.seed = seed
            self.d = 0
            if name!=None:
                self.x_init = Mazes[name]['x_init']
                self.y_init = Mazes[name]['y_init']
                for wall in Mazes[name]['walls'] : self.walls.append(wall)
                target = Mazes[name]['target']
            self.fig, self.ax = plt.subplots() if render_mode == 'human' else (None, None)
            

        def reward(self):
            d_next = np.sqrt((self.x-self.target[0])**2+(self.y-self.target[1])**2)
            r = self.d-d_next
            self.d = d_next.copy()
            return r*10.0
            # return 0.0

        def reset(self, options = None, seed =0):
            self.x=self.x_init
            self.y=self.y_init
            self.dx=self.dx_init
            self.dy=self.dy_init
            self.episode_length = 0
            self.episode_reward = 0
            self.d = np.sqrt((self.x-self.target[0])**2+(self.y-self.target[1])**2).copy()
            return np.array([self.x,self.y], dtype=np.float32).copy(),{'pos' :copy.deepcopy(np.array([self.x,self.y], dtype=np.float32)), 'target' : self.target, 'l' : 0}
        
        def step(self,v):
            self.dx=v[0]
            self.dy=v[1]
            # clip the velocity
            self.dx = np.clip(self.dx, -1, 1)
            self.dy = np.clip(self.dy, -1, 1)
            # walls 
            self.x, self.y = self.new_position(self.x,self.y,self.dx,self.dy)
            self.x = np.clip(self.x, -self.max_x, self.max_x)
            self.y = np.clip(self.y, -self.max_y, self.max_y)
            self.episode_length += 1
            reward = self.reward()
            self.episode_reward += reward
            if self.episode_length >= self.max_episode_steps : 
                return np.array([self.x,self.y], dtype=np.float32).copy(), reward, True, False, {'pos' : copy.deepcopy(np.array([self.x,self.y], dtype=np.float32)), 'l' : self.episode_length, 'episode' : {'r' : self.episode_reward, 
                                                                                                                                                                           'l' : self.episode_length, 
                                                                                                                                                                           'target' : self.target}}
            else:
                return np.array([self.x,self.y], dtype=np.float32).copy(), reward, False, False, {'pos' : copy.deepcopy(np.array([self.x,self.y], dtype=np.float32)), 'l' : self.episode_length, 'r' : self.episode_reward, 'target' : self.target}
        
    
        
        def point_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
            # Calculer le déterminant
            det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            # Si le déterminant est 0, les lignes sont parallèles (ou confondues)
            if det == 0:
                return None
            # Autrement, calculons les coordonnées du point d'intersection
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

            # Vérifier si le point d'intersection (px, py) se situe sur les deux segments
            if (min(x1, x2) <= px <= max(x1, x2) and
                    min(y1, y2) <= py <= max(y1, y2) and
                    min(x3, x4) <= px <= max(x3, x4) and
                    min(y3, y4) <= py <= max(y3, y4)):
                return px, py
            return None
        def get_normals(self, A, B):
            # Vecteur directeur du segment
            AB = B - A
            # Calcul des vecteurs normaux
            n1 = np.array([AB[1], -AB[0]])
            n2 = np.array([-AB[1], AB[0]])
            # Normalisation (si nécessaire)
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)
            return n1, n2

        def new_position(self,x,y,vx,vy):
        
            new_x = x + self.dt*vx
            new_y = y + self.dt*vy
            for wall in self.walls: 
                intersection = self.point_intersection(x, y, new_x, new_y, wall[0], wall[1], wall[2], wall[3])
                if intersection != None:
                    # for d_p in self.dangerous_point : 
                    #     if d_p == intersection: 
                    #         new_x = x - self.dt*vx/2.0
                    #         new_y = y - self.dt*vy/2.0
                    n1, n2 = self.get_normals(np.array([wall[0],wall[1]]), np.array([wall[2],wall[3]]))
                    v_n = np.dot(np.array([vx, vy]), n1)
                    v_proj = v_n * n1
                    v_after = np.array([vx, vy]) - 2 * v_proj
                    # new_x = x + self.dt*v_after[0]
                    # new_y = y + self.dt*v_after[1]
                    new_x = x
                    new_y = y 
                    return np.array([new_x,new_y])

            return (new_x, new_y)

        def render(self):
            self.ax.clear()
            self.ax.set_xlim(self.min_x, self.max_x)
            self.ax.set_ylim(self.min_y, self.max_y)
            for wall in self.walls:
                self.ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-')
            self.ax.plot(self.target[0], self.target[1], 'ro')
            self.ax.plot(self.x, self.y, 'bo')
            plt.pause(0.001)
            plt.draw()
        def close(self):
            plt.close()


Mazes = {
    'Easy' : { 
        'walls':[],
        'x_init':-0.75,
        'y_init':-0.75, 
        'target': [0.75,0.75]
    },
    'Ur' : { 
        'walls':[
            (0,-10,0,0.0)
        ],
        'x_init':-0.5,
        'y_init':-0.5,
        'target': [0.75, -0.75]
    },
    'Hard' : { 
        'walls':[(-0.25,-2,-0.25,-0.75),
                 (-2,0.0,-0.25,0.0),
                 (-0.25,0.0,-0.25,-0.25),
                 (-0.25,-0.25,0.25,-0.25),
                 (0.5,-0.50,2,-0.50),
                 (0.50,0.0,0.50,2)],
        'x_init':-0.5,
        'y_init':-0.5, 
        'target': [-0.75, 0.75]
    }
}
if __name__ ==  '__main__' : 
    env = Maze(name ='Hard', render=True)
    s=env.reset()
    for k in range(env.max_steps):
        env.step(np.array([-1,10]))
        env.render()
    env.close()

        