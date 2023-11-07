# from https://github.com/MartinEthier/highway-env-mpc

import gymnasium as gym
import pprint, copy, math
import numpy as np
import casadi as csd
from matplotlib import pyplot as plt
import pygame
import cv2
import openai


def filter_obs(obs):
    # Remove irrelevant agents
    relevant_obs = obs[obs[:, 0] != 0]
    ego_obs = relevant_obs[0]
    other_obs = relevant_obs[1:]
    return ego_obs, other_obs

def mpc(env, obs, target="lane 0"):
    ego_obs, _ = filter_obs(obs)

    N = 25 # horizon length
    ego_actions = np.zeros((N, 2))
    dt = 1.0 / env.config['policy_frequency'] # s

    vehicle_length = 5.0 # m
    vehicle_width = 2.0 # m
    lane_width = 4.0 # m

    # Setup MPC optimization
    opti = csd.Opti()
    # Control variables
    acc = opti.variable(N) # Longitudinal acceleration
    delta = opti.variable(N) # Steering angle
    # State variables
    x = opti.variable(N+1)
    y = opti.variable(N+1)
    speed = opti.variable(N+1)
    heading = opti.variable(N+1)
    
    # Objective: Maximize x position at the end of the horizon
    c_acc = 0 # Doesn't make sense to put a penalty on acceleration since our goal is to go fast
    c_delta = 4
    c_jerk = 0
    opti.minimize(-x[N] + c_acc * csd.sum1(acc) + c_delta * csd.sum1(csd.diff(delta)) + c_jerk * csd.sum1(csd.diff(acc)))
    
    # Kinematic bicycle model
    for k in range(N):
        beta = csd.arctan(0.5 * csd.tan(delta[k]))
        vx = speed[k] * csd.cos(heading[k] + beta)
        vy = speed[k] * csd.sin(heading[k] + beta)
        x_next = x[k] + vx * dt
        y_next = y[k] + vy * dt
        heading_next = heading[k] + speed[k] * csd.sin(beta) / (vehicle_length / 2) * dt
        speed_next = speed[k] + acc[k] * dt
        opti.subject_to(x[k+1] == x_next)
        opti.subject_to(y[k+1] == y_next)
        opti.subject_to(heading[k+1] == heading_next)
        opti.subject_to(speed[k+1] == speed_next)
    
    # Input constraints
    opti.subject_to(opti.bounded(-10, acc, 5))
    opti.subject_to(opti.bounded(-np.pi/3, delta, np.pi/3))
    
    # Speed constraint
    opti.subject_to(opti.bounded(0, speed, 45))
    
    # Heading constraint
    opti.subject_to(opti.bounded(-np.pi/2, heading, np.pi/2))
    
    # Off-road constraint
    opti.subject_to(opti.bounded(-lane_width/2, y, (env.config['lanes_count'] - 1/2) * lane_width))
    
    # Simulate future trajectory of all vehicles in the scene for collision constraint
    env_copy = copy.deepcopy(env)
    if env.viewer is not None:
        env.viewer.other_traj = []
    for k in range(N):
        obs_next, _, _, _, _ = env_copy.step(ego_actions[k])
        _, other_obs_next = filter_obs(obs_next)
        if env.viewer is not None:
            env.viewer.other_traj.append(other_obs_next[:, 1:3])

        other_x = other_obs_next[:, 1] + vehicle_length/2 * np.cos(other_obs_next[:, 5])
        other_y = other_obs_next[:, 2] + vehicle_length/2 * np.sin(other_obs_next[:, 5])
        other_x_noise = np.array([0.0]*2*len(other_x))
        for i in range(len(other_x_noise)):
            random_noise = np.random.uniform(0, 1.5, 1)[0]
            if i % 2 == 0:
                other_x_noise[i] = other_x[i//2] - random_noise
            else:
                other_x_noise[i] = other_x[i//2] + random_noise

        other_y = np.clip(-2, 10, other_y)

        # Compare to xy at k+1
        x_adj = x[k+1] + vehicle_length/2 * csd.cos(heading[k+1])
        y_adj = y[k+1] + vehicle_length/2 * csd.sin(heading[k+1])
        for i in range(other_x.shape[0]):
            
            # condition_range1 = (y_adj >= -2) * (y_adj <= 2) * (other_y[i] >= -2) * (other_y[i] <= 2)
            # condition_range2 = (y_adj >= 2) * (y_adj <= 6) * (other_y[i] >= 2) * (other_y[i] <= 6)
            # condition_range3 = (y_adj >= 6) * (y_adj <= 10) * (other_y[i] >= 6) * (other_y[i] <= 10)
            # condition_range4 = (y_adj >= 10) * (y_adj <= 14) * (other_y[i] >= 10) * (other_y[i] <= 14)
            # # Define the constraint using if_else to cover both ranges
            # safety_constraint = csd.if_else(condition_range1 + condition_range2 + condition_range3 + condition_range4,
            #                                 csd.sqrt((x_adj - other_x[i])**2) - vehicle_length, 0)
            # opti.subject_to(safety_constraint >= 0)

            # condition_range1 = (y_adj >= -2) * (y_adj <= 2) * (other_y[i] >= -2) * (other_y[i] <= 2)
            # condition_range2 = (y_adj >= 2) * (y_adj <= 6) * (other_y[i] >= 2) * (other_y[i] <= 6)
            # condition_range3 = (y_adj >= 6) * (y_adj <= 10) * (other_y[i] >= 6) * (other_y[i] <= 10)
            # # Define the constraint using if_else to cover both ranges
            # safety = csd.sqrt((x_adj - other_x[i])**2) - vehicle_length
            # safety_constraint = csd.if_else(condition_range1, safety, (csd.if_else(condition_range2, safety, (csd.if_else(condition_range3, safety, 0)))))
            # opti.subject_to(safety_constraint >= 0)
            # 
            # if k > 18:
            # current_y = ego_obs[2]
            # if behavior == "Lane Keep":
            #     target_y = current_y
            # elif behavior == "Lane Left":
            #     target_y = current_y - lane_width
            # elif behavior == "Lane Right":
            #     target_y = current_y + lane_width
            # else:
            #     raise ValueError("inappropriate behavior action")
            # target_y = np.clip(target_y, -2, 10)
            # if -2 <= target_y and target_y <= 2:
            #     lane_left, lane_right = -2, 2
            # elif 2 < target_y and target_y <= 6:
            #     lane_left, lane_right = 2, 6
            # elif 6 < target_y and target_y <= 10:
            #     lane_left, lane_right = 6, 10
            # else:
            #     raise ValueError("Out of lane boundary")
            if target == "Left Lane":
                lane_left, lane_right = -2, 2
            elif target == "Middle Lane":
                lane_left, lane_right = 2, 6
            elif target == "Right Lane":
                lane_left, lane_right = 6, 10
            else:
                raise ValueError("Undefined lane")
  
            if k < 5:
                opti.subject_to(opti.bounded(lane_left + 1, y_adj, lane_right-1))
            else:
                opti.subject_to(opti.bounded(lane_left+1.5, y_adj, lane_right-1.5))
            if other_x[0] < ego_obs[1] + 10 or other_x[0] > ego_obs[1] - 10:  
                if other_y[i] >= lane_left + 1 and other_y[i] <= lane_right - 1:
                    # safety_constraint = csd.sqrt((x_adj - other_x[i])**2) - vehicle_length
                    safety_constraint1 = csd.fabs(other_x_noise[2*i] - x_adj) - vehicle_length
                    # safety_constraint = -0.5 * 
                    opti.subject_to(safety_constraint1 >= 0)
                    safety_constraint2 = csd.fabs(other_x_noise[2*i+1] - x_adj) - vehicle_length
                    opti.subject_to(safety_constraint2 >= 0)
            # if k < 5:
            # opti.subject_to(csd.fabs(y_adj - other_y[i]) >= 2.)
            # if k < 10:
            #     opti.subject_to(csd.sqrt((x_adj - other_x[i])**2 + (y_adj - other_y[i])**2) >= 5.5)
            # pass

    # Set state values for k=0
    opti.subject_to(x[0] == ego_obs[1])
    opti.subject_to(y[0] == ego_obs[2])
    opti.subject_to(speed[0] == math.sqrt(ego_obs[3]**2 + ego_obs[4]**2))
    opti.subject_to(heading[0] == ego_obs[5])

    # Warm start to previous solution
    opti.set_initial(acc, ego_actions[:, 0])
    opti.set_initial(delta, ego_actions[:, 1])
    
    # Solve the optimization
    # opts = {"ipopt.print_level": 0}
    # nlp = {"x":cs.vertcat(*w), "p":par, "f":cost, "g":cs.vertcat(*G)}
    opts = {'ipopt.print_level':0, 'print_time':0}
    opti.solver('ipopt', opts)
    sol = opti.solve()
    
    if env.viewer is not None:
        env.viewer.ego_traj = np.stack((sol.value(x), sol.value(y)), axis=1)
    
    # Save full open-loop trajectory for behaviour prediction step
    ego_actions = np.stack((sol.value(acc), sol.value(delta)), axis=1)
    action = ego_actions[0]
    return action

def LLM_setup():
    question = "You are an autonomous vehicle driving on a three-lane highway, including a Left lane, a Middle lane, and a Right lane. There are \
    other cars on the highway. You have to choose which lane to drive according to your location, speed and surrounding information. The objective \
        is to drive as far as possible but you have to be cautious about the collsion. "
    question += "The rule is that if you are on the middle lane, you can choose driving on the Middle Lane, Left Lane, or Right Lane. \
                If you are on the left lane, you can only choose from Left Lane and Middle Lane. \
                If you are on the right lane, you can only choose from Right Lane and Middle Lane. "
    answer_format = "Please reply your action by choosing one lane from {Left Lane, Middle Lane, Right Lane} with reasons in a format of [target lane]: reasons. For example you reply format could be: \
    [Middle Lane], Because the Middle Lane has largest value of TTC."
    completion = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": question + answer_format}]
            )
    print(completion["choices"][0]["message"]["content"].split("\n")[0])

    split_answer = completion["choices"][0]["message"]["content"].split("\n")[0]
    # print(split_answer.split("[")[1].split("]")[0])
    # assert False


def ask_LLM(obs):
    ego, surroundings = filter_obs(obs)
    x, y, vx, vy, _ = ego[1:]
    TTC = [100, 100, 100]
    if -2 <= y and y < 2:
        Current_lane = "Left"
    elif 2 <= y and y < 6:
        Current_lane = "Middle"
    else:
        Current_lane = "Right"

    text = ""
    text += f"Currently you are driving on the {Current_lane} lane."

    LANES = ["Left Lane", "Middle Lane", "Right Lane"]
    text = ""
    for s in surroundings:
        sx, sy, svx, svy, _ = s[1:]
        idx = int(np.clip((sy + 2) // 4, 0, 2))
        text += f"There is a car driving on the {LANES[idx]}."
        if sx > x:
            if vx > svx:
                text += "It is in front of you and driving slower than you. Therefore, if you choose to merge into its lane, you should be careful." 
            else:
                text += "it is in front of you and faster than you. You can merge into its lane without too much caution."
        else:
            if vx > svx:
                text += "It is behind you and driving slower than you. Therefore, it is safe to merge into ites lane if you want to."
            else:
                text += "It is behind you and driving faster than you. You have to be cautious if you want to merge to its lane. Acceleration or wait until it passes you might be good choices."
        if sx - x > 0 and vx - svx > 0: 
            TTC[idx] = min(TTC[idx], (sx - x)/(vx - svx))
    
    if Current_lane == "Middle":
        text = f"If choose the Left Lane, your TTC is {TTC[0]:.2f} seconds; \
                If choose the Middle Lane, your TTC is {TTC[1]:.2f} seconds; \
                If choose the Right Lane, your is {TTC[2]:.2f} seconds"
    elif Current_lane == "Left":
        text = f"If choose the Left Lane,  is {TTC[0]:.2f} seconds; \
                If choose the Middle Lane, TTC is {TTC[1]:.2f} seconds; \
                You cannot choose the Right Lane, because you are on the Left Lane."
    elif Current_lane == "Right":
        text = f"If choose the Right Lane, TTC is {TTC[2]:.2f} seconds; \
                If choose the Middle Lane, TTC is {TTC[1]:.2f} seconds; \
                You cannot choose the Left Lane, because you are on the Right Lane."
        
    text += "You should prefer the action that brings LARGER TTC value. \
                because this means you have lower collision risk after lane changing. \
                for example, if left lane has TTC as 20 and right lane has TTC as 19, you should consider to prefer left lane because 20 > 19. \
                You should encourage lane change if the target lane has a larger TTC, no need to worry about lane change risk, the low level MPC will ensure safety, \
                I will provide TTC in each lane for you to make a decision. However, you should also consider the whole scene I described to you, and \
                choose the optimal selection based on your understanding. For example, besides the TTC value, you may choose the middle lane as it gives more operational space."

    answer_format = "Please reply your action by choosing one lane from {Left Lane, Middle Lane, Right Lane} with reasons in a format of [target lane]: reasons. For example you reply format could be: \
    [Middle Lane], Because the Middle Lane has largest value of TTC."
    completion = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": text + answer_format}]
            )
    answer = completion["choices"][0]["message"]["content"].split("\n")[0]
    behavior = answer.split("[")[1].split("]")[0]
    reason = answer.split("[")[1].split("]")[-1]

    if behavior not in ["Left Lane", "Middle Lane", "Right Lane"]:
        completion = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "Your reply format is not correct, " + answer_format}]
                )
        answer = completion["choices"][0]["message"]["content"].split("\n")[0]
        behavior = answer.split("[")[1].split("]")[0]
        reason = answer.split("[")[1].split("]")[-1]
    print(behavior, reason)        
    return behavior, Current_lane

def mpc_fail_feedback_llm(old_behavior, Current_lane):
    prompt = f"You are driving on the {Current_lane} and the "
    prompt += old_behavior + f"is not feasible for the low-level control optimization in the current time step. \
        Please propose another possible action, keep in mind that it is highly likely to be infeasible if your new lane requires the ego to cross the {old_behavior}. \
            Please also keep in mind that you can keep the {Current_lane} if there is no better choice."
    
    answer_format = "Please reply your action by choosing one lane from {Left Lane, Middle Lane, Right Lane} with reasons in a format of [target lane]: reasons. For example you reply format could be: \
    [Middle Lane], Because the Middle Lane has largest value of TTC."
    completion = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt + answer_format}]
            )
    answer = completion["choices"][0]["message"]["content"].split("\n")[0]
    behavior = answer.split("[")[1].split("]")[0]
    reason = answer.split("[")[1].split("]")[-1]

    if behavior not in ["Left Lane", "Middle Lane", "Right Lane"] or behavior == old_behavior:
        completion = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": f"Your reply format is not correct, " + answer_format + " please also avoid " + old_behavior}]
                )
        answer = completion["choices"][0]["message"]["content"].split("\n")[0]
        behavior = answer.split("[")[1].split("]")[0]
        reason = answer.split("[")[1].split("]")[-1]
    print("")
    print("new proposed action is: " + behavior + " the reason is: " + reason)        
    return behavior

def fail_safe(obs, Current_lane):
    # prompt = f"You are driving on the {Current_lane} and {behavior1} and {behavior2} are infeasible for the low-level MPC."
    # prompt += "The ego is entering fail safe mode and slow down the speed without steering."
    # prompt += f"It is possible that {behavior1} and/or {behavior2} are infeasible in the near future, \
    # if there is another decision choice, you should prefer it,  but you should try {behavior1} and/or {behavior2} if no better choice."
    # answer_format = "You don't have to answer anything for this."
    # completion = openai.ChatCompletion.create(
    #                 # model="gpt-3.5-turbo",
    #                 model = "gpt-4",
    #                 messages=[{"role": "user", "content": prompt + answer_format}]
    #                 )
    
    ego, surroundings = filter_obs(obs)
    x, y, vx, vy, heading = ego[1:]
    ego_idx = int(np.clip((y + 2) // 4, 0, 2))
    acc, ttc = 5, 100
    for s in surroundings:
        sx, sy, svx, svy, _ = s[1:]
        s_idx = int(np.clip((sy + 2) // 4, 0, 2))
        if s_idx == ego_idx and sx > x and vx > svx:
            ttc = min(ttc, max(0, (sx- x) / (vx - svx)))
            # acc = min(acc, 0.8*(svx - vx))
            acc = min(acc, -vx / ttc)
            print(sx, x, svx, vx, ttc, acc)    
    action = np.array([acc, -heading*0.4])
    return action

# LLM_MODEL = "gpt-3.5-turbo"
LLM_MODEL = "gpt-4"
MAX_STEPS = 600
env = gym.make("highway-env-mpc-v0", render_mode='rgb_array')
done, cnt = False, 0
obs, _ = env.reset()
action_periods = 5
all_actions = ["Lane Keep", "Lane Right", "Lane Left"]
behavior = "Lane Keep"
LLM_setup()
velocity_collection = []

while not done:
    print(f"step {cnt} starts: ")
    if cnt % action_periods == 0:
        attempt = 0
        while attempt < 5:
            try:
                behavior, Current_lane = ask_LLM(obs)
                attempt = 5
            except:
                print("first round error happens when asking LLM")
            attempt += 1
    try:     
        action = mpc(env, obs, target=behavior)
    except:
        print(f"{behavior} is infeasible, will ask LLM again")
        # if cnt % action_periods == 0:
        attempt = 0
        while attempt < 5:
            try:
                behavior = mpc_fail_feedback_llm(behavior, Current_lane) 
                attempt = 5
            except:
                print("second round error happens when asking LLM")
            attempt += 1
        try:
            action = mpc(env, obs, target=behavior)
        except:
            print("retried llm action is also infeasible, use failsafe lane keep deaccelerate instead.")         
            try:
                action = mpc(env, obs, target=Current_lane + " Lane")
            except:
                print("all MPCs failed, turns to backup")
                action = fail_safe(obs, Current_lane)

    obs, reward, done, truncated, _ = env.step(action)
    ego_observation, _ = filter_obs(obs)
    _, _, ego_vx, ego_vy, _ = ego_observation[1:]
    velocity_collection.append([ego_vx, ego_vy])  
    render_img = env.render()
    # # Save image    
    cv2.imwrite(f"/mnt/d/highway-env-mpc/render_images/1107/img_{cnt:04}.png", render_img)
    cnt += 1
    if cnt >= MAX_STEPS:
        done = True
    np.save('velocity_trace.npy', np.array(velocity_collection))