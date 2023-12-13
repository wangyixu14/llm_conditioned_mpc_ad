import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
import cv2
import time

from scenario.scenario import Scenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser
from LLMDriver.customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe,
)

OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'azure':
    os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
    os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['AZURE_API_VERSION']
    os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['AZURE_API_BASE']
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['AZURE_API_KEY']
    llm = AzureChatOpenAI(
        deployment_name=OPENAI_CONFIG['AZURE_MODEL'],
        temperature=0,
        max_tokens=1024,
        request_timeout=60
    )
elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-1106', # or any other model with 8k+ context
        max_tokens=1024,
        request_timeout=120
    )


# base setting
vehicleCount = 30

# environment setting
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
        # "vehicles_count": vehicleCount,
        # "see_behind": True,
        "order": "sorted"
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 40, 10),
    },
    "lanes_count": 3,
    "vehicles_count": 30,
    "controlled_vehicles": 1,
    "initial_lane_id": -2,
    # "simulation_frequency": 10,
    # "policy_frequency": 10,
    # "duration": 60,
    "vehicles_density": 0.5,
    "collision_reward": -1,     # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.1,   # The reward received when driving on the right-most lanes, linearly mapped to
                                # zero for other lanes.
    "high_speed_reward": 0.4,   # The reward received when driving at full speed, linearly mapped to zero for
                                # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,    # The reward received at each lane change action.
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "show_trajectories": True,
    "render_agent": True,
}


env = gym.make('highway-v0', render_mode="rgb_array")
env.configure(config)
env = RecordVideo(
    env, './results-video',
    name_prefix=f"highwayv0"
)
env.unwrapped.set_record_video_wrapper(env)
obs, info = env.reset()
env.render()

# scenario and driver agent setting
if not os.path.exists('results-db/'):
    os.mkdir('results-db')
database = f"results-db/highwayv0.db"
sce = Scenario(vehicleCount, database)
toolModels = [
    getAvailableActions(env),
    getAvailableLanes(sce),
    getLaneInvolvedCar(sce),
    isChangeLaneConflictWithCar(sce),
    isAccelerationConflictWithCar(sce),
    isKeepSpeedConflictWithCar(sce),
    isDecelerationSafe(sce),
    isActionSafe(),
]
DA = DriverAgent(llm, toolModels, sce, verbose=True)
outputParser = OutputParser(sce, llm)
output = None
done = truncated = False
frame = 0
MAX_STEPS = 600
TRIAL_NUM = 5

velocity_collection = []
timing = []

def filter_obs(obs):
    # Remove irrelevant agents
    relevant_obs = obs[obs[:, 0] != 0]
    ego_obs = relevant_obs[0]
    other_obs = relevant_obs[1:]
    return ego_obs, other_obs

try:
    while not (done or truncated):
        now = time.time()
        sce.upateVehicles(obs, frame)
        DA.agentRun(output)
        da_output = DA.exportThoughts()
        output = outputParser.agentRun(da_output)
        env.render()
        env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()
        obs, reward, done, info, _ = env.step(output["action_id"])
        ego_observation, _ = filter_obs(obs)
        _, _, ego_vx, ego_vy = ego_observation[1:]
        timing.append(time.time() - now)
        velocity_collection.append([ego_vx, ego_vy])
        np.save('velocity_trace_'+str(TRIAL_NUM)+'.npy', np.array(velocity_collection))
        np.save('timing_'+str(TRIAL_NUM)+'.npy', np.array(timing))
        print(output)
        frame += 1
        render_img = env.render()
        # # Save image    
        cv2.imwrite(f"/mnt/d/highway-env-mpc/DriveLikeAHuman/render_images_{TRIAL_NUM}/img_{frame:04}.png", render_img)
        if frame >= MAX_STEPS:
            done = True
finally:
    env.close()