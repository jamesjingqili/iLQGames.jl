import pybullet as p
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import gym
from gym import Env
from gym.spaces import Discrete, Box

from utils.learned_feature import LearnedFeature
from utils.environment_utils import *
from utils.plot_utils import *

import sys
import pygame
from pygame.locals import *
from itertools  import permutations, product

import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are
import time

from DRE_class import *
from LQR_util import *
import copy


from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


# Simulated human internal reward dynamics
def update_human_internal_reward(traj_snippet,\
                              robot_action_snippet,\
                              human_action_snippet,\
                              Q_int_np, w_g_int_np,\
                              human_mode, learning_rate):
  """Update the simulated human's internal robot model using gradient decent"""

  Q_int_tensor   = nn.parameter.Parameter(torch.tensor(Q_int_np, device = device))
  w_g_int_tensor = nn.parameter.Parameter(torch.tensor(w_g_int_np, device = device))
  human_int_loss = 0
  for i in range(1):
    x_0 = traj_snippet[i]
    u_R_0 = robot_action_snippet[i]
    # compute the ARE solution under the current Q estimate
    P_R = Riccati.apply(A_tensor, B_tensor, Q_int_tensor, R_tensor)
    R_inv_tensor = torch.tensor(R_inv, device = device).double()
    temp = torch.matmul(R_inv_tensor, torch.transpose(B_tensor,0,1))
    K = -torch.matmul(temp, P_R)
    # compute human's control induced by the human's reward estimate
    u_H = torch.matmul(K, torch.tensor(x_0, device = device) - torch.matmul(X_g_tensor, w_g_int_tensor))
    human_int_loss += torch.norm(u_H - torch.tensor(u_R_0, device = device)) ** 2
  human_int_loss.backward()

  # use the Q grad to update the Q estimate
  Q_int_grad = Q_int_tensor.grad.cpu().numpy() * Q_grad_mask
  #print(Q_int_grad)
  d_Q = 0.1 * Q_int_grad
  if human_mode == 'gradient_decent_threshold':
    if abs(d_Q[0,0]) < 0.02:
      d_Q[0,0] = 0.0
    if abs(d_Q[1,1]) < 0.02:
      d_Q[1,1] = 0.0
    if abs(d_Q[2,2]) < 0.02:
      d_Q[2,2] = 0.0
  # clamp the max change of the Q matrix
  d_Q[0,0] = max(min(d_Q[0,0], 0.3), -0.3)
  d_Q[1,1] = max(min(d_Q[1,1], 0.3), -0.3)
  d_Q[2,2] = max(min(d_Q[2,2], 0.3), -0.3)
 
  Q_int_np_new = Q_int_tensor.detach().cpu().numpy() - d_Q
  Q_int_np_new[1,1] = max(min(Q_int_np_new[1,1], 1.0), 0.01)
  Q_int_np_new[2,2] = max(min(Q_int_np_new[2,2], 1.0), 0.01)
  #Q_int_tensor = nn.parameter.Parameter(torch.tensor(Q_int_np_new,device = device))
  #np.random.multivariate_normal(np.zeros((3)), np.eye(3,3) * 0.001)
  
  # use the w grad to update the goal weight matrix
  w_g_int_grad = w_g_int_tensor.grad.cpu().numpy() * w_grad_mask
  w_g_int_np_new = w_g_int_tensor.detach().cpu().numpy() -    learning_rate * w_g_int_grad
  #w_g_int_np_new = np.exp(3. * w_g_int_np_new)/sum(np.exp(3. * w_g_int_np_new))
  w_g_int_np_new[0,0] = max(min(w_g_int_np_new[0,0], 1.0), 0.01)
  w_g_int_np_new[1,0] = max(min(w_g_int_np_new[1,0], 1.0), 0.01)
  w_g_int_np_new = w_g_int_np_new / sum(w_g_int_np_new)
  #print(w_g_int_np_new)
  #w_g_int_tensor = nn.parameter.Parameter(torch.tensor(w_g_int_np_new,device = device))
  #print(w_int_tensor)
  return Q_int_np_new, w_g_int_np_new

# Gym human class
class HumanRobotEnv(Env):
    def __init__(self, robot_mode, alpha, sim_mode, human_mode) :
        # the mode of the robot (active or passive)
        self.robot_mode = robot_mode
        # the blending policy of the robot
        self.alpha = alpha
        self.sim_mode = sim_mode
        self.human_mode = human_mode
        # ground truth environment dynamics (linear dynamics with LQR control)
        self.A_t = None
        self.B_t = None
        self.Q_t = None
        self.R_t = None
        # robot goal set
        self.X_g = None
        self.nX = None
        self.nU = None
        self.episode_length = None
        # actions that the robot can take
        self.action_space = None
        self.robot_action_set = None
        # observation recieved by the robot
        self.observation_space = Box(low=np.array([-2, -2, -2, 0.000]), \
                                     high=np.array([2, 2, 2, 1.0]), dtype=np.float)
        self.human_internal_model = None
        # Initialize the human state
        self.physical_state = None
        self.mental_state = None
        self.state = None
        
        # If in gym env, the action is an index of the action set.
        self.in_gym_env = True
        self.robot_action_mode = 'addon'
        self.modeled_human_lr =  6.0

        self.step_count = 1
        # Initialize the traj, state list to save the data
        self.current_demo_state_traj              = []
        self.current_demo_human_action_traj       = []
        self.current_demo_human_obs_traj          = []
        self.current_demo_robot_action_traj       = []
        self.current_demo_human_mental_state_traj = []
        self.current_demo_reward_traj             = []
        self.current_demo_task_reward_traj        = []
        self.current_demo_action_reward_traj      = []
  
    def set_environment(self, A_t, B_t, Q_t, R_t, X_g, w_g_t, sequence_length):
        """
        Set the environment dynamics and the goal set
        """
        self.A_t = A_t
        self.B_t = B_t
        self.Q_t = Q_t
        self.R_t = R_t
        self.X_g = X_g
        self.w_g_t = w_g_t
        self.nX = A_t.shape[0]
        self.nU = B_t.shape[1]
        self.episode_length = sequence_length
    
    def set_robot_action_set(self, robot_action_set):
        """
        Set the action set of the robot
        """
        self.action_space = Discrete(len(robot_action_set))
        self.robot_action_set = robot_action_set
    
    def set_human_internal_model(self, dynamics_model):
        """
        Set the learned human model
        """
        self.human_internal_model = dynamics_model
    
    def set_human_state(self, physical_state, mental_state):
        """
        Set the human state
        """
        self.physical_state = physical_state
        self.mental_state = mental_state
        self.state = \
        np.concatenate((self.physical_state, self.mental_state), 0).reshape(physical_state.shape[0] + mental_state.shape[0])

    def step(self, action):

        u_t0_R = get_LQR_control(self.physical_state, self.A_t, self.B_t,\
                                 self.Q_t, self.R_t, np.matmul(self.X_g, self.w_g_t))
        if action is not None:
            #query the robot action from the action set (note that here the action is an index)        
            if self.in_gym_env:
                u_t0_R_aug = np.array(self.robot_action_set[action]).reshape(self.nU,1)
            else:
                u_t0_R_aug = action
        else:
            u_t0_R_aug = 0
        if self.robot_action_mode == 'addon':
            u_t0_R += u_t0_R_aug
        else:
            u_t0_R = u_t0_R_aug
        #u_t0_R = action # for eval the function only
        # estimate the human action
        w_g_hat = np.zeros((2,1))
        w_g_hat[0,0] = self.mental_state[0,0]
        w_g_hat[1,0] = 1 - w_g_hat[0,0]

        u_t0_H = get_LQR_control(self.physical_state, self.A_t, self.B_t,\
                                 self.Q_t, self.R_t, np.matmul(self.X_g, w_g_hat))

        # fuse the action and update state
        u_t0 = self.alpha * u_t0_H + (1- self.alpha) * u_t0_R
        #u_t0 = action# for eval the function only
        x_t1 = np.matmul(self.A_t, self.physical_state) \
              + np.matmul(self.B_t, u_t0)      
        
        self.current_demo_state_traj.append([self.physical_state])
        self.current_demo_human_action_traj.append([u_t0_H])
        self.current_demo_human_obs_traj.append([u_t0])
        self.current_demo_robot_action_traj.append([u_t0_R])
        self.current_demo_human_mental_state_traj.append(copy.deepcopy(self.mental_state))
        # update the mental state
        if self.sim_mode == 'use_nn_human' and self.human_mode == 'dynamic': 
            current_demo_state_traj_copy = copy.deepcopy(self.current_demo_state_traj)
            current_demo_human_action_traj_copy = copy.deepcopy(self.current_demo_human_action_traj)
            current_demo_human_obs_traj_copy = copy.deepcopy(self.current_demo_human_obs_traj)
            f_hat_batch_pred = InternalModelPred(self.human_internal_model, 
                                                current_demo_state_traj_copy,
                                                current_demo_human_action_traj_copy,
                                                current_demo_human_obs_traj_copy)
            self.mental_state[0,0] = f_hat_batch_pred[0,-1,0]
        if self.sim_mode == 'use_model_human'and self.human_mode == 'dynamic':
            # use the ground truth model to test the RL algorithm
            Q_int_t0, w_g_int_t0 = update_human_internal_reward([self.physical_state], \
                                                             [u_t0], [u_t0_H],\
                                                             self.Q_t, w_g_hat,\
                                                             'grad', self.modeled_human_lr)
            self.mental_state[0,0] = copy.deepcopy(w_g_int_t0[0,0])
            #print(w_g_int_t0)
        # Calculate reward
        if self.robot_mode == 'active_teaching':
            reward = -np.linalg.norm(self.mental_state[0,0] - self.w_g_t[0,0])
        if self.robot_mode == 'active_assisting':
            reward = -get_LQR_cost(self.physical_state,\
                                    self.A_t, self.B_t,\
                                    self.Q_t, self.R_t,\
                                    np.matmul(X_g, self.w_g_t),\
                                    u_t0)\
                     - 1.5 * np.linalg.norm(u_t0_R-u_t0_H)
            reward = reward[0,0]

        self.current_demo_reward_traj.append(reward)
        # add the initial state cost to the list so they start from the same
        # if len(self.current_demo_task_reward_traj)==0:
        #   self.current_demo_task_reward_traj.append(-getLQRCost(self.physical_state, u_t0*0)[0,0])
        self.current_demo_task_reward_traj.append(\
                                                  -get_LQR_cost(self.physical_state,\
                                                                self.A_t, self.B_t,\
                                                                self.Q_t, self.R_t,\
                                                                np.matmul(X_g, self.w_g_t),\
                                                                u_t0)[0,0])
        self.current_demo_action_reward_traj.append(- 1.5 * np.linalg.norm(u_t0_R-u_t0_H))

        # check if simulation ends
        self.step_count += 1
        if self.step_count == self.episode_length:
            done = 1
        else:
            done = 0
        info = {}

        # Return step information
        # update the physical_state
        self.physical_state = x_t1
        self.state = \
        np.concatenate((self.physical_state, self.mental_state), 0).reshape(self.physical_state.shape[0] + self.mental_state.shape[0],)
        return self.state, reward, done, info
    
    def reset(self):
        self.physical_state = np.array([[-0.3], [0.0], [1.3]])
        self.mental_state   = np.array([[0.]])
        self.step_count = 1
        self.current_demo_state_traj = []
        self.current_demo_human_action_traj = []
        self.current_demo_human_obs_traj = []
        self.current_demo_robot_action_traj = []
        self.current_demo_human_mental_state_traj = []
        self.current_demo_reward_traj = []
        self.current_demo_task_reward_traj = []
        self.current_demo_action_reward_traj = []
        return self.state

# g1 = [0.1, 0.3, 0.6]
# g2 = [0.1, -0.3, 0.6]
def setup_environment():
    objectID = {}
    # Add the first goal.
    # objectID["goal_1"] = p.loadURDF("tray/tray_yellow.urdf", [0.1, 0.3, 0.65], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
    # # Add the second goal.
    # objectID["goal_2"] = p.loadURDF("tray/tray.urdf", [0.1, -0.3, 0.65], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
    
    objectID["goal_3"] = p.loadURDF("objects/mug.urdf", [0.1, -0, 0.63], p.getQuaternionFromEuler([0, 0, -11]), useFixedBase=True)

    # Add the floor.
    objectID["plane"] = p.loadURDF("plane.urdf")

    # Add a table.
    pos = [0, 0, 0]
    orientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["table"] = p.loadURDF("table/table.urdf", pos, orientation, useFixedBase=True)

    # Add a robot support.
    pos = [-0.65, 0, 0.65]
    orientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["stand"] = p.loadURDF("support.urdf", pos, orientation, useFixedBase=True)

    # Add the Jaco robot and initialize it.
    pos = [-0.65, 0, 0.675]
    orientation = p.getQuaternionFromEuler([0, 0, 0.2])
    objectID["robot"] = p.loadURDF("jaco.urdf", pos, orientation, useFixedBase=True)
    move_robot(objectID["robot"])
    
   
    return objectID

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument(
        "--feature",
        type=str,
        default='table',
        help="Feature to be taught",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="../data/user_data/",
        help="Path to dir where traces should be saved",
    )

    parser.add_argument(
        "--resources-dir",
        type=str,
        default="../data/resources",
        help="Path to dir where environment resources are stored.",
    )

    args = parser.parse_args()
    return args

def envsetup(args, direct=False):
    # Connect to physics simulator.
    if direct:
        physicsClient = p.connect(p.DIRECT)
    else:
        physicsClient = p.connect(p.GUI)

    # Add path to data resources for the environment.
    p.setAdditionalSearchPath(args.resources_dir)

    # Setup the environment.
    objectID = setup_environment()

    # Get rid of gravity and make simulation happen in real time.
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(1)

    return objectID

# Move the robot's end-effector to the target pose
def move_end_point(robotID, targetPose):
    n_ite = 10
    dist2 = -1.
    newPos = None
    for itr in range(n_ite):
        jointPoses = p.calculateInverseKinematics(robotID, 7, targetPose,1)
        for jointIndex in range(p.getNumJoints(robotID)-1):
            p.resetJointState(robotID, jointIndex+1, jointPoses[jointIndex])
        ls = p.getLinkState(robotID, 7)
        newPos = ls[4]
        diff = [targetPose[0] - newPos[0], targetPose[1] - newPos[1], targetPose[2] - newPos[2]]
        dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
    newPos = np.reshape(np.array(newPos), (3,1))
    return newPos, np.sqrt(dist2)


def robot_LQR_policy(x_robot_np):
    
    u_t0_R = get_LQR_control(x_robot_np, A, B, Q_t, R, robot_goal)
    return u_t0_R

def human_policy(x_robot_np, w_g_human):
    u_t0_H = get_LQR_control(x_robot_np, A, B, Q_t, R, np.matmul(X_g, w_g_human))
    return u_t0_H

def goal_reached(x_robot_np, robot_goal):
    return np.linalg.norm(x_robot_np - robot_goal) < 0.1

# Build the human-robot environment.
def human_env_builder(task_name, x_robot_np):
    robot_policy = None
    eval_env = None
    if task_name == 'active_assisting_dynamic_human':
        eval_env = HumanRobotEnv('active_assisting',0.6,'use_model_human', 'dynamic')
        eval_env.set_environment(A, B, Q_t, R, X_g, w_g_robot, task_duration)
        eval_env.set_robot_action_set(robot_action_set)
        eval_env.set_human_internal_model(None)
        eval_env.set_human_state(x_robot_np, np.array([[0.]]))
        eval_env.modeled_human_lr = 6.0
        obs = eval_env.reset()
        robot_policy= PPO.load('/Users/rantian/Dropbox/ICRA23_Tian_InfluenceHumanInternalModel/code/active_assisting_dynamic_8_17.zip', env = eval_env)
    if task_name == 'active_assisting_static_human':
        eval_env = HumanRobotEnv('active_assisting',0.6,'use_model_human', 'dynamic')
        eval_env.set_environment(A, B, Q_t, R, X_g, w_g_robot, task_duration)
        eval_env.set_robot_action_set(robot_action_set)
        eval_env.set_human_internal_model(None)
        eval_env.set_human_state(x_robot_np, np.array([[0.]]))
        eval_env.modeled_human_lr = 6.0
        eval_env.robot_action_mode = 'replace'
        obs = eval_env.reset()
        robot_policy= PPO.load('/Users/rantian/Dropbox/ICRA23_Tian_InfluenceHumanInternalModel/code/active_assisting_static_8_17.zip', env = eval_env)
    if task_name == 'active_teaching':
        eval_env = HumanRobotEnv('active_teaching',0.8,'use_model_human', 'dynamic')
        eval_env.set_environment(A, B, Q_t, R, X_g, w_g_robot, task_duration)
        eval_env.set_robot_action_set(robot_action_set)
        eval_env.set_human_internal_model(None)
        eval_env.set_human_state(x_robot_np, np.array([[0.]]))
        eval_env.modeled_human_lr = 4.0
        eval_env.robot_action_mode = 'addon'
        obs = eval_env.reset()
        robot_policy= PPO.load('/Users/rantian/Dropbox/ICRA23_Tian_InfluenceHumanInternalModel/code/active_teaching_8_17.zip', env = eval_env)
    if task_name == 'passive_teaching':
        eval_env = HumanRobotEnv('active_teaching',0.8,'use_model_human', 'dynamic')
        eval_env.set_environment(A, B, Q_t, R, X_g, w_g_robot, task_duration)
        eval_env.set_robot_action_set(robot_action_set)
        eval_env.set_human_internal_model(None)
        eval_env.set_human_state(x_robot_np, np.array([[0.]]))
        eval_env.modeled_human_lr = 4.0
        eval_env.robot_action_mode = 'addon'
        eval_env.in_gym_env = False
        obs = eval_env.reset()
    return eval_env, robot_policy

def get_robot_human_actions(task_name, x_robot_np, robot_policy, eval_env):
    # Compute the optimal robot action if robot acts along.
    u_t0_R = robot_LQR_policy(x_robot_np)
    # Compute the robot action if robot tries to teach the human.
    if task_name == 'active_assisting_dynamic_human' or task == 'active_teaching':
        action, _state = robot_policy.predict(eval_env.state, deterministic = True)
        u_t0_R_aug =  np.array(eval_env.robot_action_set[action]).reshape(eval_env.nU,1)
        u_t0_R += u_t0_R_aug
        obs, reward, done, info = eval_env.step(action)
    if task_name == 'active_assisting_static_human':
        action, _state = robot_policy.predict(eval_env.state, deterministic = True)
        u_t0_R =  np.array(eval_env.robot_action_set[action]).reshape(eval_env.nU,1)
        obs, reward, done, info = eval_env.step(action)
    if task_name == 'passive_teaching':
        obs, reward, done, info = eval_env.step(None)
    u_t0_H = eval_env.current_demo_human_action_traj[-1][0]
    return u_t0_R, u_t0_H

############################### Define the environment parameters ##############################################
device = 'cpu'
sampling_time = 0.2

# robot dynamics matrix
A        = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
A        = np.array(A)
A_tensor = torch.from_numpy(A).to(device).double()
nX       = A.shape[0]

# robot control matrix
B        = [[sampling_time, 0., 0.], [0., sampling_time, 0.], [0., 0., sampling_time]]
B        = np.array(B)
B_tensor = torch.from_numpy(B).to(device).double()
nU       = B.shape[1]

# Optimal LQR cost function
Q_t        = [[1., 0., 0.], [0., 1., 0.], [0., 0., 0.1]]
Q_t        = np.array(Q_t)
Q_t_tensor = torch.from_numpy(Q_t).to(device).double()

R        = 10 * np.eye(3,3)
R_tensor = torch.from_numpy(R).to(device).double()
R_inv    = np.linalg.inv(R)

# Human's internal cost function
Q_int = [[1.0, 0., 0.], [0.0, 0.05, 0.], [0., 0., 1.0]]
Q_int = [[1.0, 0., 0.], [0.0, 1.0, 0.], [0., 0., 0.1]]
Q_int = np.array(Q_int)

# Human's goal set
g1 = [0.1, 0.3, 0.6]
g2 = [0.1, -0.3, 0.6]
goal_set = [g1, g2]
X_g = np.array([g1, g2]).T
X_g_tensor = torch.tensor(X_g, requires_grad=False, device = device)

# Belief distribution over the goal set
w_g_human = np.array([[0.], [1.]])
w_g_robot = np.array([[1.], [0.]])

# Mask for the Q gradient
Q_grad_mask = np.array(Q_int != Q_t) * 1.0
#print("Q grad mask:",Q_grad_mask)
w_grad_mask = np.array(w_g_human != w_g_robot) * 1.0
#print("Goal weight mask:", w_grad_mask)


action_set       = [0, 0.05, 0.1, 0.15, 0.2, -0.05, -0.1, -0.15, -0.2]
robot_action_set = list(product(action_set, repeat = 3))
robot_action_set = np.array(robot_action_set).tolist()

########################### Start the GUI and Joystick ########################################

# Parse experimental arguments.
args = parse_arguments()
args.feature = 'table'
# Start GUI
objectID = envsetup(args)
p.resetDebugVisualizerCamera(cameraDistance=1.22, cameraYaw=-450.82, cameraPitch=-22.38,
                                cameraTargetPosition=[-0.2,0,0.61])
pos, _ = p.getBasePositionAndOrientation(objectID["robot"])
#[(-0.65, 0.85), (-0.5, 0.5), (0.7250000000000001, 1.675)]
limits = [(pos[0], pos[0]+1.5), (pos[1]-1.0/2, pos[1]+1.0/2), (pos[2]+0.05, pos[2]+1.0)]
# Initialize the Joystick controller
pygame.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
for joystick in joysticks:
    print(joystick.get_name())



########################### Select the task ########################################
task = 'passive_teaching'

# If true, use real human input.
real_human_flag = False
manual_mode = False

########################### Initialize the human class ########################################

x_robot = [-0.3, 0.0, 1.3]
x_robot_np =  np.array(x_robot).reshape(3,1)
task_duration = 65
eval_env, robot_policy = human_env_builder(task, x_robot_np)

robot_goal = np.matmul(X_g, w_g_robot)
robot_goal[2,0] += 0.2

u_t0_R = None
u_t0_H = None
Delta_t = 0.0
frequency = 12

if manual_mode:
    eval_env.alpha = 1.0
    eval_env.episode_length = 10000
########################### Start the simulation ########################################
# used to track the real human control
u_human = [0.0, 0.0, 0.0]

model_preedicted_state = None
move_end_point(objectID["robot"], x_robot)

#p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, task+".mp4")

while True:

    start_time = time.time()
    # Compute the first robot action and hunan action for the first time interval.
    if u_t0_R is None:
        u_t0_R, u_t0_H = get_robot_human_actions(task, x_robot_np, robot_policy, eval_env)

    # Overwrite the human action if the real human flag is true.  
    if real_human_flag:
        for event in pygame.event.get():
            if event.type == JOYAXISMOTION:    
                # motion signals
                if event.axis == 3:
                    u_human[1] = -event.value
                if event.axis == 4:
                    u_human[0] = -event.value
                if event.axis == 1:
                    u_human[2] = event.value
            if event.type == JOYDEVICEADDED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
                for joystick in joysticks:
                    print(joystick.get_name())
            if event.type == JOYDEVICEREMOVED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        u_t0_H = np.array(u_human).reshape((3,1)) * 0.1

    x_robot[0] += (eval_env.alpha * u_t0_H[0,0] + (1-eval_env.alpha) * u_t0_R[0,0]) * sampling_time /frequency
    x_robot[1] += (eval_env.alpha * u_t0_H[1,0] + (1-eval_env.alpha) * u_t0_R[1,0]) * sampling_time / frequency
    x_robot[2] += (eval_env.alpha * u_t0_H[2,0] + (1-eval_env.alpha) * u_t0_R[2,0]) * sampling_time / frequency
    #print('Desired robot position', x_robot)    
    x_robot_np[0,0] = x_robot[0]
    x_robot_np[1,0] = x_robot[1]
    x_robot_np[2,0] = x_robot[2]

    # Cap the robot's position to present constraints violation.
    for i in range(3):
        x_robot[i] = min(max(x_robot[i], limits[i][0]+0.02), limits[i][1]-0.02)

    # Update the robot's state in the physcis environment.
    if not goal_reached(x_robot_np, robot_goal):
        x_robot_np, pose_error = move_end_point(objectID["robot"], x_robot)
        time.sleep(0.01)
    else:
        break
    # Update elapsed time.
    end_time = time.time()
    Delta_t += end_time - start_time
    
    # Reached the sampling time
    if Delta_t >= sampling_time:
        print('Step:', eval_env.step_count)
        print('Time elapsed:', Delta_t)
        
        # Difference between the predicted state from discrete model and the real state.
        print('State diff.:', np.linalg.norm(eval_env.physical_state - x_robot_np))
        
        # Zero out the time
        Delta_t = 0.0
        eval_env.physical_state = x_robot_np

        # Compute the robot and human control for the next time interval
        u_t0_R, u_t0_H = get_robot_human_actions(task, x_robot_np, robot_policy, eval_env)
          
        # Reset the sampling time
        Delta_t = 0.0
        if eval_env.step_count >= eval_env.episode_length:
           break

print('Goal reached!')
print(eval_env.current_demo_human_mental_state_traj)
p.disconnect()
