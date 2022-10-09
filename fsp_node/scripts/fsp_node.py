#!/usr/bin/env python3

import sys
sys.path.append(".")


# ros package
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import tf
from ros_numpy import numpify

# habitat
from arguments import get_args
import numpy as np
from sem_policy import SLAMTrainer








def movebase_client(stg):

    (stg_x, stg_y) = stg
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = -stg_x
    goal.target_pose.pose.position.y = -stg_y
    goal.target_pose.pose.orientation.w = 1.0

    client.send_goal(goal)
    # wait = client.wait_for_result()
    # if not wait:
    #     rospy.logerr("Action server not available!")
    #     rospy.signal_shutdown("Action server not available!")
    # else:
    #     return client.get_result()

global obs
obs = {}


def odom_callback(msg):
    # print(msg)
    global obs
    # print(rs_rgb.type)
    obs['gps'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    obs['compass'] = [yaw]
    obs['objectgoal'] = [4]

    # print(obs['gps'])
    # print(yaw)
    
    
    # print("recevied odom!")

    # movebase_client(stg)


def rgb_callback(msg):
    global obs
    h = msg.height
    w = msg.width
    rs_rgb = numpify(msg)
    # print(rs_rgb.shape)
    # print(rs_rgb.type)
    rs_rgb= np.reshape(rs_rgb, (h, w, 3))
    obs['rgb'] = rs_rgb


    # print("recevied rgb!")

def depth_callback(msg):
    global obs
    h = msg.height
    w = msg.width
    # print(type(msg.data)) # <class 'bytes'>
    rs_depth = numpify(msg)

    rs_depth= np.reshape(rs_depth, (h, w, 1))
    # print(rs_depth[100][100])
    obs['depth'] = rs_depth

    # print("recevied depth!")


def main():
    rospy.init_node('fsp_node')

    # rospy.Subscriber("/jackal_velocity_controller/odom", Odometry, odom_callback)
    rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)
    listener = tf.TransformListener()

    velocity_publisher = rospy.Publisher('/fsp/cmd_vel', Twist, queue_size=10)

    # client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    # client.wait_for_server()

    # stg_x_old = 0
    # stg_y_old = 0
    
    time_count = 0
    action = 0

    args = get_args()
    print(args)

    agent = SLAMTrainer(args)


    rate = rospy.Rate(10) # 10hz
    global obs
    
    while not rospy.is_shutdown():
        if len(obs) > 0:

            if time_count % 50 == 0:
                (trans,rot) = listener.lookupTransform("/map", "/camera_link", rospy.Time(0))
                obs['gps'] = [trans[0], trans[1]]
                (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(rot)
                # first_yaw = -yaw
                obs['compass'] = [yaw]
                obs['objectgoal'] = [4]
                action = agent.act(obs)
                
                time_count = 1

            # (stg_x, stg_y) = stg
            # goal = MoveBaseGoal()
            # goal.target_pose.header.frame_id = "map"
            # goal.target_pose.header.stamp = rospy.Time.now()
            # goal.target_pose.pose.position.x = -stg_x
            # goal.target_pose.pose.position.y = -stg_y
            # goal.target_pose.pose.orientation.w = 1.0

            # if stg_x != stg_x_old:
            #     client.send_goal(goal)
            #     stg_x_old = stg_x
            # wait = client.wait_for_result()
            # if not wait:
            #     rospy.logerr("Action server not available!")
            #     rospy.signal_shutdown("Action server not available!")
            # else:
            #     return client.get_result()
            print("stg: ", action)
            vel_msg = Twist()

            # vel_msg.linear.x = 0
            # vel_msg.linear.y = 0
            # vel_msg.linear.z = 0
            # vel_msg.angular.x = 0
            # vel_msg.angular.y = 0
            # vel_msg.angular.z = 0 

            if action == 1:
                vel_msg.linear.x = 0.05
            elif action == 2:
                vel_msg.angular.z = 0.2
            elif action == 3:
                vel_msg.angular.z = -0.2
                

            velocity_publisher.publish(vel_msg)
            time_count +=1

        rate.sleep()

if __name__ == "__main__":
    main()
