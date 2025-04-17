import highway_env
import gym
import numpy as np
from gym.envs.registration import register

def register_highway_envs():
    register(
        id='intention-v0',
        entry_point='highway_env.envs:IntentionEnv'
    )

    register(
        id='intention-v1',
        entry_point='highway_env.envs:IntentionEnvHard',
    )

# Simple class to hold environment parameters
class Args:
    def __init__(self):
        self.difficulty = "easy"
        self.n_obs_vehicles = 10
        self.episode_limit = 40
        self.n_lane = 4
        self.n_agents = 1
        self.vehicles_density = 1
        self.scaling = 5.5
        self.screen_height = 150
        self.screen_width = 600
        self.n_other_vehicles = 50

# env wrapper for highway_env
def env_wrapper(args=None):
    if args is None:
        args = Args()
        
    if args.difficulty == "easy":
        env = gym.make("intention-v0")
    else:
        env = gym.make("intention-v1")

    env.configure({
      "action": {
        "type": "MultiAgentAction",
        "action_config": {
          "type": "DiscreteMetaAction",
        }
      }
    })

    env.configure({
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": args.n_obs_vehicles,  # Number of observed vehicles
                "features": ["id", "presence", "x", "y", "vx", "vy",
                             "ax", "ay", "type"],  # added features
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted"
            }
        }
    })
    # State definition
    env.configure({"features": ["id", "presence", "x", "y", "vx", "vy", 
                                "ax", "ay", "type"]})  # added features

    env.configure({"duration": args.episode_limit})
    env.configure({"lanes_count": args.n_lane})
    env.configure({"controlled_vehicles": args.n_agents})
    env.configure({"vehicles_density": args.vehicles_density})
    env.configure({"simulation_frequency": 5})
    # Display
    env.configure({'scaling': args.scaling, 'screen_height': args.screen_height, 'screen_width': args.screen_width})
    env.configure({"vehicles_count": args.n_other_vehicles})
    env = highway_env.envs.MultiAgentWrapper(env)
    return env

if __name__ == "__main__":
    register_highway_envs()
    env = env_wrapper()
    obs = env.reset()
    print("Environment reset successfully!")
    print(f"Observation shape: {np.shape(obs)}")
    
    # Render the environment
    env.render()
    
    # Keep the window open until user closes it
    import time
    try:
        while True:
            env.render()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Simulation ended by user")
