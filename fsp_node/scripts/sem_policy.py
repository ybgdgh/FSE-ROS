#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import random
from typing import Dict, Optional
import math

n_dir = os.path.dirname(__file__)

import numba
import numpy as np
import torch
import torch.nn as nn
import gym
from torchvision import transforms

import habitat
from habitat.config import Config
from habitat.core.agent import Agent
from habitat.core.simulator import Observations

from skimage import measure
import skimage.morphology
from PIL import Image

import cv2

from model import RL_Policy, Semantic_Mapping, RL_Policy_IAM
import utils.pose as pu
from utils.fmm_planner import FMMPlanner
from utils.semantic_prediction import SemanticPredMaskRCNN
import utils.visualization as vu
from arguments import get_args
from habitat_sim.utils.common import d3_40_colors_rgb


from constants import coco_categories_mapping, coco_categories, color_palette, category_to_id

from RedNet.RedNet_model import load_rednet
from constants import mp_categories_mapping



@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def find_big_connect(image):
    img_label, num = measure.label(image, neighbors=8, return_num=True)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
    # print("img_label.shape: ", img_label.shape) # 480*480
    resMatrix = np.zeros(img_label.shape)
    tmp_area = 0
    for i in range(0, len(props)):
        if props[i].area > tmp_area:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix = tmp
            tmp_area = props[i].area 
    
    return resMatrix



class SLAMTrainer(Agent):
    def __init__(self, args) -> None:
        self.args = args
        print("args: ", args)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        self.device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

        # Semantic detect
        self.sem_pred = SemanticPredMaskRCNN(args)
        # ckpt_path = os.path.join(n_dir, 'RedNet/model/rednet_semmap_mp3d_40.pth')
        # self.red_sem_pred = load_rednet(
        #     self.device, ckpt=ckpt_path, resize=True, # since we train on half-vision
        # )
        # self.red_sem_pred.eval()
        # self.red_sem_pred.to(self.device)

        # initializations for planning:
        self.selem = skimage.morphology.disk(4)

        self.last_sim_location = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.last_action = None
        self.col_width = None
        self.l_step = 0
        self.episode_n = 0
        self.collision_n = 0

        self.collision_s = 0
        self.replan_count = 0
        self.replan_flag = 0


        if args.visualize or args.print_images:
            legend_path = os.path.join(n_dir, 'docs/legend.png')
            self.legend = cv2.imread(legend_path)
            self.vis_image = None
            self.rgb_vis = None

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        
        # Initialize map variables:
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        nc = args.num_sem_categories + 4  # num channels

        # Calculating full and local map sizes
        self.map_size = args.map_size_cm // args.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w = int(self.full_w / args.global_downscaling)
        self.local_h = int(self.full_h / args.global_downscaling)

        # Initializing full and local map
        self.full_map = torch.zeros(nc, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(nc, self.local_w,
                                self.local_h).float().to(self.device)

        self.local_ob_map = np.zeros((self.local_w,
                                self.local_h))

        self.local_ex_map = np.zeros((self.local_w,
                                self.local_h))

        self.target_edge_map = np.zeros((4, self.local_w,
                                self.local_h))

        self.target_point_map = np.zeros((self.local_w,
                                self.local_h))

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        self.tv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
        

        # Initial full and local pose
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.local_pose = torch.zeros(3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros(3)

        # Local Map Boundaries
        self.lmb = np.zeros(4).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros(7)

        self.init_map_and_pose()

        # Global policy observation space
        ngc = 8 + args.num_sem_categories
        es = 2
        self.g_observation_space = gym.spaces.Box(0, 1,
                                            (ngc,
                                            int(self.local_w/2.0),
                                            int(self.local_h/2.0)), dtype='uint8')


        # Global policy action space
        self.g_action_space = gym.spaces.Discrete(4)

        # Invalid action masking space
        self.invalid_action_masks = torch.zeros(self.g_action_space.n).float().to(self.device)

        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(args).to(self.device)
        self.sem_map_module.eval()


        self.g_policy = RL_Policy_IAM(self.g_observation_space.shape, self.g_action_space,
                         model_type=1,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': args.global_hidden_size,
                                      'num_sem_categories': ngc - 8
                                      }).to(self.device)
        self.g_action = torch.zeros(1)
        self.global_input = torch.zeros(ngc, int(self.local_w/2), int(self.local_h/2)).to(self.device)
        self.global_orientation = torch.zeros(1).long()
        self.extras = torch.zeros(2, dtype=torch.long).to(self.device)


        if args.load != "0":
            print("Loading model {}".format(args.load))
            load_path = os.path.join(n_dir, args.load)
            state_dict = torch.load(load_path,
                                    map_location=lambda storage, loc: storage)
            self.g_policy.load_state_dict(state_dict)
        self.g_policy.eval()

        self.reset()

        
    def remove_small_points(self, local_ob_map, image, threshold_point, pose):
        # print("goal_cat_id: ", goal_cat_id)
        # print("sem: ", sem.shape)
        selem = skimage.morphology.disk(1)
        traversible = skimage.morphology.binary_dilation(
            local_ob_map, selem) != True
        # traversible = 1 - traversible
        planner = FMMPlanner(traversible)
        goal_pose_map = np.zeros((local_ob_map.shape))
        goal_pose_map[int(pose[0].cpu()), int(pose[1].cpu())] = 1
        # goal_map = skimage.morphology.binary_dilation(
        #     goal_pose_map, selem) != True
        # goal_map = 1 - goal_map
        planner.set_multi_goal(goal_pose_map)

        img_label, num = measure.label(image, connectivity=2, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        # print("img_label.dtype: ", img_label.dtype) # 480*480
        Goal_edge = np.zeros((4, img_label.shape[0], img_label.shape[1]))
        Goal_point = np.zeros(img_label.shape)
        dict_cost = {}
        for i in range(1, len(props)):
            # print("area: ", props[i].area)
            # dist = pu.get_l2_distance(props[i].centroid[0], pose[0], props[i].centroid[1], pose[1])
            dist = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])] * 5
            dist_s = 8 if dist < 300 else 0
            
            cost = props[i].area + dist_s

            if props[i].area > threshold_point and dist > 180 and dist < np.max(planner.fmm_dist-4):
                dict_cost[i] = cost
        
        if dict_cost:
            dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)
            
            # print(dict_cost)
            for i, (key, value) in enumerate(dict_cost):
                # print(i, key)
                Goal_edge[i][img_label == key + 1] = 1
                Goal_point[int(props[key].centroid[0]), int(props[key].centroid[1])] = i+1 #
                if i == 3:
                    break

        return Goal_edge, Goal_point

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.full_map[2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0


        self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
                                            (self.local_w, self.local_h),
                                            (self.full_w, self.full_h))

        self.planner_pose_inputs[3:] = self.lmb
        self.origins = [self.lmb[2] * self.args.map_resolution / 100.0,
                    self.lmb[0] * self.args.map_resolution / 100.0, 0.]

        self.local_map = self.full_map[:,
                                self.lmb[0]:self.lmb[1],
                                self.lmb[2]:self.lmb[3]]
        self.local_pose = self.full_pose - \
            torch.from_numpy(np.array(self.origins)).to(self.device).float()




    def reset(self) -> None:
        self.init_map_and_pose()

        self.local_ob_map = np.zeros((self.local_w,
                                self.local_h))

        self.local_ex_map = np.zeros((self.local_w,
                                self.local_h))

        self.target_edge_map = np.zeros((4, self.local_w,
                                self.local_h))

        self.target_point_map = np.zeros((self.local_w,
                                self.local_h))

        self.l_step = 0
        self.last_action = None
        self.col_width = 1

        self.episode_n += 1
        self.collision_n = 0
        self.replan_count = 0
        self.replan_flag = 0
        self.stop = 0

        self.stair_flag = 0
        self.stop_flag = 0

        self.curr_loc = [self.args.map_size_cm / 100.0 / 2.0,
                         self.args.map_size_cm / 100.0 / 2.0, 0.]

        map_shape = (self.map_size, self.map_size)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        print(self.episode_n)
        

    def act(self, observations: Observations) -> Dict[str, int]:

        # observations['gps'] = [int(observations['gps'][1] * 100.0 / self.args.map_resolution),
        #                 int(observations['gps'][0] * 100.0 / self.args.map_resolution)]
        # observations['gps'][0] += self.args.map_size_cm / 100 / 2.0
        # observations['gps'][1] += self.args.map_size_cm / 100 / 2.0

        if self.l_step == 0:
            self.local_pose[2] = observations['compass'][0] * 57.29577951308232
            self.last_sim_location = [observations['gps'][0], observations['gps'][1], observations['compass'][0]]

            actions = torch.randn(1, 2)*6
            cpu_actions = nn.Sigmoid()(actions).numpy().squeeze()
            global_goals = [int(cpu_actions[0] * self.local_w),
                             int(cpu_actions[1] * self.local_h)]
            self.global_goals = [min(global_goals[0], int(self.local_w - 1)),
                             min(global_goals[1], int(self.local_h - 1))] 

            if self.args.visualize or self.args.print_images:
                self.vis_image = vu.init_vis_image(category_to_id[observations['objectgoal'][0]], self.legend)
                self.goal_name = category_to_id[observations['objectgoal'][0]]
            # print("objectgoal: ", observations['objectgoal'])


        rgb = observations['rgb'].astype(np.uint8)
        depth = observations['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)



        # print("rgb_size: ", rgb.shape)
        # print("depth_size: ", depth.shape)

        obs = self._preprocess_obs(state) 

        rgb_img = Image.fromarray(rgb, mode="RGB")

        rgb_img = cv2.cvtColor(np.asarray(rgb_img),cv2.COLOR_RGB2BGR)
        
        
        # cv2.imshow("rgb", rgb_img)
        # cv2.imshow("depth", depth)
        obs = torch.from_numpy(obs).float().to(self.device)

        poses = torch.from_numpy(np.asarray(self.get_pose_change(observations['gps'], observations['compass']))).float().to(self.device)
        
        _, self.local_map, self.local_map_stair, self.local_pose = \
            self.sem_map_module(obs.unsqueeze(0), poses.unsqueeze(0), self.local_map.unsqueeze(0), self.local_pose.unsqueeze(0))

        # self.local_map[0, :, :][self.local_map[13, :, :] > 0.5] = 0
        # print("local_map shape: ", local_map.shape)
        # print("local_pose: ", self.local_pose)
        # print("poses: ", poses)

        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs + self.origins
        self.local_map[2, :, :].fill_(0.)  # Resetting current location channel
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]
        self.local_map[2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.


        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if self.l_step % self.args.num_local_steps == self.args.num_local_steps - 1:
            # For every global step, update the full and local maps

            # self.full_map[:, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]] = \
            #     self.local_map
            # self.full_pose = self.local_pose + \
            #     torch.from_numpy(np.asarray(self.origins)).to(self.device).float()

            # locs = self.full_pose.cpu().numpy()
            # r, c = locs[1], locs[0]
            # loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
            #                 int(c * 100.0 / self.args.map_resolution)]

            # self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
            #                                 (self.local_w, self.local_h),
            #                                 (self.full_w, self.full_h))

            # self.planner_pose_inputs[3:] = self.lmb
            # self.origins = [self.lmb[2] * self.args.map_resolution / 100.0,
            #             self.lmb[0] * self.args.map_resolution / 100.0, 0.]

            # self.local_map = self.full_map[:,
            #                         self.lmb[0]:self.lmb[1],
            #                         self.lmb[2]:self.lmb[3]]
            # self.local_pose = self.full_pose - \
            #     torch.from_numpy(np.asarray(self.origins)).to(self.device).float()

      
            

            # if torch.all(self.local_map[18, loc_r-10:loc_r+10, loc_c-10:loc_c+10] < 0.1):
            #     self.stair_flag = 0
            #     self.replan_flag = 0
                
        # if self.stop or (self.l_step % self.args.num_local_steps == self.args.num_local_steps - 1 and self.l_step < 26):
            ############################ choose global goal map #############################
            # choose global goal map
            local_ob_map = self.local_map[0].cpu().numpy()
            self.local_ob_map = cv2.dilate(local_ob_map, self.kernel)

            show_ex = cv2.inRange(self.local_map[1].cpu().numpy(),0.1,1)
            
            contours,_=cv2.findContours(show_ex, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            if len(contours)>0:
                contour = max(contours, key = cv2.contourArea)
                cv2.drawContours(self.local_ex_map,contour,-1,1,1)

            # clear the boundary
            self.local_ex_map[0:2, 0:self.local_w]=0.0
            self.local_ex_map[self.local_w-2:self.local_w, 0:self.local_w-1]=0.0
            self.local_ex_map[0:self.local_w, 0:2]=0.0
            self.local_ex_map[0:self.local_w, self.local_w-2:self.local_w]=0.0
            

            target_edge = self.local_ex_map-self.local_ob_map
            # print("local_ob_map ", self.local_ob_map[200])
            # print("full_map ", self.full_map[0].cpu().numpy()[200])

            target_edge[target_edge>0.8]=1.0
            target_edge[target_edge!=1.0]=0.0
            
            # full_pose_map = [self.full_pose[1]*100/self.args.map_resolution, self.full_pose[0]*100/self.args.map_resolution]
            local_pose_map = [self.local_pose[1]*100/self.args.map_resolution, self.local_pose[0]*100/self.args.map_resolution]
            self.target_edge_map, self.target_point_map = self.remove_small_points(local_ob_map, target_edge, 4, local_pose_map) 
    
            # self.target_lo_map[int(full_pose_map[0])-20:int(full_pose_map[0])+20, int(full_pose_map[1])-20:int(full_pose_map[1])+20]=0.0


            self.local_ob_map=np.zeros((self.local_w,
                    self.local_h))
            self.local_ex_map=np.zeros((self.local_w,
                    self.local_h))

            # ------------------------------------------------------------------

            ##### Semantic Policy 
            # ------------------------------------------------------------------
            # print("self.target_edge_map: ", self.target_edge_map.shape)
            locs = self.local_pose.cpu().numpy()
            self.global_orientation = int((locs[2] + 180.0) / 5.)
            self.global_input[0:4, :, :] = nn.MaxPool2d(self.args.global_downscaling)(
                torch.from_numpy(self.target_edge_map).unsqueeze_(0).to(self.device).float())
            self.global_input[4:, :, :] = nn.MaxPool2d(self.args.global_downscaling)(self.local_map[0:, :, :])
            self.goal_cat_id = torch.tensor(coco_categories[observations['objectgoal'][0]])
            # print("self.global_orientation: ", self.global_orientation)
            self.extras[0] = self.global_orientation
            self.extras[1] = self.goal_cat_id

            # config invalid action mask
            self.invalid_action_masks.fill_(0.)
            tpm = len(list(set(self.target_point_map.ravel()))) -1
            for inv in range(tpm):
                self.invalid_action_masks[inv] = 1

            # print("extras: ", self.extras.unsqueeze(0))
            self.g_action = self.g_policy.act(
                    self.global_input.unsqueeze(0),
                    extras=self.extras.unsqueeze(0),
                    deterministic=False,
                    invalid_action_masks=self.invalid_action_masks
                    )

            actions = torch.randn(1, 2)*1
            cpu_actions = nn.Sigmoid()(actions).cpu().numpy().squeeze()
            global_goals = [int(cpu_actions[0] * self.local_w),
                            int(cpu_actions[1] * self.local_h)]
            self.global_goals = [min(global_goals[0], int(self.local_w - 1)),
                            min(global_goals[1], int(self.local_h - 1))] 
                # print("self.global_goals: ", self.global_goals)

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = 0
    
        local_goal_maps = np.zeros((self.local_w, self.local_h)) 

        # print("self.global_goals: ", self.global_goals)
        cn = coco_categories[observations['objectgoal'][0]] + 4

        tpm = len(list(set(self.target_point_map.ravel()))) -1
        # cpu_actions = nn.Sigmoid()(self.g_action).cpu().numpy()
        global_item = self.g_action.cpu().numpy()

        if np.any(self.target_point_map == global_item+1):
            local_goal_maps[self.target_point_map == global_item+1] = 1
            # print("Find the edge")
        else:
            local_goal_maps[self.global_goals[0], self.global_goals[1]] = 1



                # print("Don't Find the edge")
        if self.local_map[cn, :, :].sum() != 0.:
            cat_semantic_map = self.local_map[cn, :, :].cpu().numpy()
            cat_semantic_scores = cat_semantic_map
            cat_semantic_scores[cat_semantic_scores > 0] = 1.
            local_goal_maps = find_big_connect(cat_semantic_scores)
            found_goal = 1
     
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = {}
        # self.planner_pose_inputs[3:] = [0, self.local_w, 0, self.local_h]
        planner_inputs['map_pred'] = self.local_map[0, :, :].cpu().numpy()
        planner_inputs['exp_pred'] = self.local_map[1, :, :].cpu().numpy()
        planner_inputs['pose_pred'] = self.planner_pose_inputs
        planner_inputs['goal'] = local_goal_maps  # global_goals[e]
        planner_inputs['new_goal'] = (self.l_step % self.args.num_local_steps - 1) == 0
        planner_inputs['found_goal'] = found_goal
        if self.args.visualize or self.args.print_images:
            self.local_map[-1, :, :] = 1e-5
            planner_inputs['sem_map_pred'] = self.local_map[4:, :,
                                            :].argmax(0).cpu().numpy()
        
        action = self._plan(planner_inputs)
        # print("self.l_step: ", self.l_step)
        # print("action: ", action)
        # stg = np.nonzero(local_goal_maps)
        self.l_step += 1

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs, action)

        # (stg_x, stg_y) = stg
        # stg_x = stg_x / 100.0 * self.args.map_resolution
        # stg_y = stg_y / 100.0 * self.args.map_resolution
        # stg_x -= self.origins[0]
        # stg_y -= self.origins[1]

        # stg = [int(stg[1] / 100.0 * self.args.map_resolution),
        #                 int(stg[0] / 100.0 * self.args.map_resolution)]

        return action
    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        if planner_inputs["new_goal"]:
            self.collision_map = np.zeros(self.visited.shape)

        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        # if args.visualize or args.print_images:
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                            self.visited_vis[gx1:gx2, gy1:gy2])

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)

        # Deterministic Local Policy
        if stop and planner_inputs['found_goal'] == 1:
            self.stop_flag = 0  # Stop
            action = 0
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        # print("grid: ", grid.shape)

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        # traversible = grid[x1:x2, y1:y2] != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self.kernel) == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        if replan:
            self.replan_count += 1
            print("false: ", self.replan_count)
        else:
            self.replan_count = 0

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        # print("obs: ", obs.shape)
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        # print(depth.shape)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))   

        sem_seg_pred = self._get_sem_pred(
            rgb.astype(np.uint8), depth, use_seg=use_seg)


        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        # depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state


    def _preprocess_depth(self, depth, min_d, max_d):
        # print("depth origin: ", depth[100])
        depth = depth / 10.0
        depth = depth.astype(np.float32)

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 300
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 50000.0
        # depth = depth * 100.0
        # depth = depth*1000.
        # print(depth.shape)
        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
        # if use_seg:
            # image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            # depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            # red_semantic_pred = self.red_sem_pred(image, depth).squeeze().cpu().detach().numpy()
            # self.rgb_vis = rgb[:, :, ::-1]
        if use_seg:
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)

            # semantic_img = Image.new("P", (semantic_pred.shape[1], semantic_pred.shape[0]))
            # semantic_img.putpalette(d3_40_colors_rgb.flatten())
            # semantic_img.putdata((semantic_pred.flatten() % 40).astype(np.uint8))
            # semantic_img = semantic_img.convert("RGBA")
            # sem_img = cv2.cvtColor(np.asarray(semantic_img),cv2.COLOR_RGB2BGR)
            # fn = 'result_target/Vis-rgb-{}.png'.format(self.l_step)
            # fn_path = os.path.join(n_dir, fn)
            # cv2.imwrite(fn_path, sem_img)

        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return semantic_pred

    def get_pose_change(self, gps, compass):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        # print("gps: ", gps)
        # print("compass: ", compass)
        curr_sim_pose = [gps[0], gps[1], compass[0]]
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def _visualize(self, inputs, action):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/eps_{}/'.format(
            dump_dir, self.l_step)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)

        self.vis_image = vu.init_vis_image(self.goal_name, str(action))

        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("episode_n {}".format(self.episode_n), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            fn = '{}/episodes/eps_{}/Vis-{}.png'.format(
                dump_dir, self.episode_n,
                self.l_step)
            cv2.imwrite(fn, self.vis_image)



