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

def mpc(env, obs, behavior="Lane Right"):
    ego_obs, _ = filter_obs(obs)

    N = 30 # horizon length
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
    c_delta = 0.5
    c_jerk = 0
    opti.minimize(-x[N] + c_acc * csd.sum1(acc) + c_delta * csd.sum1(delta) + c_jerk * csd.sum1(csd.diff(acc)))
    
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
    opti.subject_to(opti.bounded(-5, acc, 5))
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
        # other_y = np.clip(-2, 10, other_y)

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
            current_y = ego_obs[2]
            if behavior == "Lane Keep":
                target_y = current_y
            elif behavior == "Lane Left":
                target_y = current_y - lane_width
            elif behavior == "Lane Right":
                target_y = current_y + lane_width
            else:
                raise ValueError("inappropriate behavior action")
            target_y = np.clip(target_y, -2, 10)
            if -2 <= target_y and target_y <= 2:
                lane_left, lane_right = -2, 2
            elif 2 < target_y and target_y <= 6:
                lane_left, lane_right = 2, 6
            elif 6 < target_y and target_y <= 10:
                lane_left, lane_right = 6, 10
            else:
                raise ValueError("Out of lane boundary")
            # opti.subject_to(opti.bounded(lane_left+1, y_adj, lane_right-1))    
            if k < 4:
                opti.subject_to(opti.bounded(lane_left+0.5, y_adj, lane_right-0.5))
            else:
                opti.subject_to(opti.bounded(lane_left+1.7, y_adj, lane_right-1.7))

            if other_y[i] >= lane_left + 1 and other_y[i] <= lane_right - 1:
                # safety_constraint = csd.sqrt((x_adj - other_x[i])**2) - vehicle_length
                safety_constraint = csd.fabs(other_x[i] - x_adj) - vehicle_length
                opti.subject_to(safety_constraint >= 0)
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
        is to drive as far as possible but you have to cautious about the collsion. "
    question += "The rule is that if you are on the middle lane, you can choose Lane Keeping, Change Lane to Left, or Change Lane to Right. \
                If you are on the left lane, you can only choose from Lane Keeping or Change Lane to Right. \
                If you are on the right lane, you can only choose from Lane Keeping or Change Lane to Left. "
    # question += "You should prefer the action that brings LARGER TTC in the target lane. \
                # because this means you have lower collision risk after lane changing. I will provide TTC in each lane for you to make a decision."
    
    # answer_format = "Please reply your action by choosing one from {Lane Left, Lane Keep, Lane Right} without saying anything else."
    answer_format = "Please reply your action by choosing one from {Lane Left, Lane Keep, Lane Right} with reasons in a format of [action]: reasons. For example you reply format could be: \
    [Lane Keep], Because Lane Keep has largest value of TTC."
    completion = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model = "gpt-4",
            messages=[{"role": "user", "content": question + answer_format}]
            )
    print(completion["choices"][0]["message"]["content"].split("\n")[0])

    split_answer = completion["choices"][0]["message"]["content"].split("\n")[0]
    print(split_answer.split("[")[1].split("]")[0], split_answer.split("[")[1].split("]")[-1])
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

    for s in surroundings:
        sx, sy, svx, svy, _ = s[1:]
        idx = int(np.clip((sy + 2) // 4, 0, 2))
        if sx - x > 0 and vx - svx > 0: 
            TTC[idx] = min(TTC[idx], (sx - x)/(vx - svx))
    
    if Current_lane == "Middle":
        text = f"If choose Lane Left, your TTC is {TTC[0]:.2f} seconds; \
                If choose Lane Keep, your TTC is {TTC[1]:.2f} seconds; \
                If choose Lane Right, your is {TTC[2]:.2f} seconds"
    elif Current_lane == "Left":
        text = f"If choose Lane Keep,  is {TTC[0]:.2f} seconds; \
                If choose Lane Right, TTC is {TTC[1]:.2f} seconds; \
                You cannot choose Lane Left, because you are on the left lane already."
    elif Current_lane == "Right":
        text = f"If choose Lane Keep, TTC is {TTC[2]:.2f} seconds; \
                If choose Lane Left, TTC is {TTC[1]:.2f} seconds; \
                You cannot choose Lane Right, because you are on the right lane already."
        
    text += "You should prefer the action that brings LARGER TTC value. \
                because this means you have lower collision risk after lane changing. \
                for example, if left lane has TTC as 20 and right lane has TTC as 19, you should consider to prefer left lane because 20 > 19. \
                I will provide TTC in each lane for you to make a decision."

    question = "You are an autonomous vehicle driving on a three-lane highway, including a Left lane, a Middle lane, and a Right lane. There are \
    other cars on the highway. You have to choose which lane to drive according to your location, speed and surrounding information. The objective \
        is to drive as far as possible but you have to cautious about the collsion. "
    question += "The rule is that if you are on the middle lane, you can choose Lane Keeping, Change Lane to Left, or Change Lane to Right. \
                If you are on the left lane, you can only choose from Lane Keeping or Change Lane to Right. \
                If you are on the right lane, you can only choose from Lane Keeping or Change Lane to Left. "

    question += f"Currently, you are driving on the {Current_lane} lane."
    # question += "You should consider to lane change, if the value of TTC (time to collision) in the target lane is lower than the TTC in the current lane \
                # because this means you have lower collision risk after lane changing. I will provide TTC in each lane for you to make a decision."
    
    # answer_format = "Please reply your action by choosing one from {Lane Left, Lane Keep, Lane Right} without saying anything else."
    answer_format = "Please reply your action by choosing one from {Lane Left, Lane Keep, Lane Right} with reasons in a format of [action]: reasons. For example you reply format could be: \
    [Lane Keep], Because Lane Keep has largest value of TTC."
    completion = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model = "gpt-4",
            messages=[{"role": "user", "content": question + text + answer_format}]
            )
    answer = completion["choices"][0]["message"]["content"].split("\n")[0]
    behavior = answer.split("[")[1].split("]")[0]
    reason = answer.split("[")[1].split("]")[-1]

    if behavior not in ["Lane Keep", "Lane Left", "Lane Right"]:
        completion = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model = "gpt-4",
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
        Please propose another possible action, keep in mind that this action should be feasible given the {Current_lane} and should have a fair LARGE TTC value. \
            "
    answer_format = "Please reply your action by choosing one from {Lane Left, Lane Keep, Lane Right} with reasons in a format of [action]: reasons. For example you reply format could be: \
    [Lane Keep], Because Lane Keep has largest value of TTC."
    completion = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model = "gpt-4",
            messages=[{"role": "user", "content": prompt + answer_format}]
            )
    answer = completion["choices"][0]["message"]["content"].split("\n")[0]
    behavior = answer.split("[")[1].split("]")[0]
    reason = answer.split("[")[1].split("]")[-1]

    if behavior not in ["Lane Keep", "Lane Left", "Lane Right"] or behavior == old_behavior:
        completion = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model = "gpt-4",
                messages=[{"role": "user", "content": f"Your reply format is not correct, " + answer_format + " please also avoid " + old_behavior}]
                )
        answer = completion["choices"][0]["message"]["content"].split("\n")[0]
        behavior = answer.split("[")[1].split("]")[0]
        reason = answer.split("[")[1].split("]")[-1]
    print("new proposed action is: " + behavior + " the reason is: " + reason)        
    return behavior

env = gym.make("highway-env-mpc-v0", render_mode='rgb_array')
done, cnt = False, 0
obs, _ = env.reset()
action_periods = 5
all_actions = ["Lane Keep", "Lane Right", "Lane Left"]
behavior = "Lane Keep"
LLM_setup()

while not done:
    if cnt % action_periods == 0:
        attempt = 0
        while attempt < 5:
            try:
                behavior, Current_lane = ask_LLM(obs)
                attempt = 5
            except:
                print("error happens when asking LLM")
            attempt += 1
    try:
        action = mpc(env, obs, behavior=behavior)
    except:
        try:
            behavior = mpc_fail_feedback_llm(behavior, Current_lane)
            action = mpc(env, obs, behavior=behavior)
        except:
            print("retried llm action is also infeasible, use failsafe lane keep instead.")         
            action = mpc(env, obs, behavior="Lane Keep")
    obs, reward, done, truncated, _ = env.step(action)   
    render_img = env.render()
    # # Save image    
    cv2.imwrite(f"/mnt/d/highway-env-mpc/render_images/img_{cnt:04}.png", render_img)
    cnt += 1