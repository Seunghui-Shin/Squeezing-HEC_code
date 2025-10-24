import _init_paths
import argparse
import random
import numpy as np
import numpy.ma as ma
import yaml
import copy
import cv2 
import time
import random
from scipy.spatial.transform import Rotation as RR
from skimage.metrics import structural_similarity as ssim


import torch
import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd import Variable
from DenseFusion_Pytorch_1_0.lib.network import PoseNet, PoseRefineNet
from DenseFusion_Pytorch_1_0.lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from DenseFusion_Pytorch_1_0.vanilla_segmentation.segnet import SegNet 

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
import tf
from kortex_driver.srv import *
from kortex_driver.msg import *
from franka_cal_sim_single.srv import Dfaction, Dfnext, DfnextResponse

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
# Topic rgb, depth topic setting 
image_topic = "/camera/color/image_raw"
# depth_topic = "/camera/aligned_depth_to_color/image_raw"
depth_topic = "/camera/depth/image_raw"

# ROS topic for listening to camera intrinsics
camera_info_topic = "/camera/color/camera_info"

# ROS service for sending request to capture frame
capture_frame_service_topic = "/dense_fusion/capture_frame"

# ROS service for sending request to clear buffer
clear_buffer_service_topic = "/dense_fusion/clear_buffer"

# ROS topics for outputs
topic_out_net_input_image = "/dense_fusion/net_input_image"
topic_out_keypoint_frame_overlay = "/dense_fusion/keypoint_frame_overlay"
topic_out_segment_keypoint_frame_overlay = "/dense_fusion/2D_segment"

# ROS frames for the output of dense_fusion
tform_out_childname = "dense_fusion/camera_rgb_frame"


offset = 0
fail = False
done = False
action_pose = [0,0,0,0,0,0]
raw_action_pose = 100
restart_cnt = 0
reward = -100
next_state = [0,0,0,0,0,0,0,0,0,0,0,0,0]
initial_state_num = 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, rgb, depth, mask, noise_trans, refine):
        self.objlist = [16]
        self.mode = mode
        self.list_rgb = []
        self.list_depth = []
        self.list_label = [] # mask 
        self.list_obj = 16
        self.noise_trans = noise_trans
        self.refine = refine
        
        self.list_rgb.append(rgb)
        self.list_depth.append(depth)
        self.list_label.append(mask)

        self.length = len(self.list_rgb)

        self.cam_cx = 320.44989013671875
        self.cam_cy = 244.81730651855469
        self.cam_fx = 614.04742431640625
        self.cam_fy = 614.044677734375

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [0]

    def __getitem__(self, idx):
        img = self.list_rgb[0]
        
        depth = np.array(self.list_depth[0])
        label = np.array(self.list_label[0])
        obj = self.list_obj     
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0)) # 0 아닌 부분 --(True)

        
        if self.mode == 'eval':
            if obj == 16:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255))) # new_dr 102
            elif obj == 17:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(127)))
            else:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))

        mask = mask_label * mask_depth
        if self.add_noise:
            img = self.trancolor(img)
       
        img = np.array(img)[:, :, :3]
     
        img = np.transpose(img, (2, 0, 1))
     
        img_masked = img

        if self.mode == 'eval':
            rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
            


        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc)

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int) 
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])


        cam_scale = 1.0
        pt2 = depth_masked / cam_scale # z
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx # x
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy # y
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0 #(x,y,z)
        
        if self.add_noise:
            cloud = np.add(cloud, add_t)
    
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.LongTensor([self.objlist.index(obj)])

    def __len__(self):
        return self.length


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640



def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax



class DenseFusionInferenceRos:
    def __init__(self, args, single_frame_mode=True):
        """Initialize inference engine.

            single_frame_mode:  Set this to True.  (Appears to be some sort of future-proofing by Tim.)
        """
        self.is_there_segment_image = False 


        self.cv_image = None
        self.camera_K = None
        self.single_frame_mode = single_frame_mode

        self.kp_projs_raw_buffer = np.array([])
        self.kp_positions_buffer = np.array([])
        self.bounding_box_found = False
        self.capture_frame_max_kps = True

        self.K = np.array([[614.04742431640625,0,320.44989013671875],[0,614.044677734375,244.81730651855469],[0,0,1]])
        self.segment_image_org = np.zeros((480,640))

        self.segnet_make_image = False 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
        # Setting ros node 
        # Create subscribers
        self.bridge = CvBridge()

        # create image and depth subscriber
        self.image_sub = rospy.Subscriber(
            image_topic, Image, self.on_image, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(
            depth_topic, Image, self.on_depth, queue_size=1
        )

        # Define publishers
        self.net_input_image_pub = rospy.Publisher(
            topic_out_net_input_image, Image, queue_size=1
        )
        self.kp_frame_overlay_pub = rospy.Publisher(
            topic_out_keypoint_frame_overlay, Image, queue_size=1
        )

        self.segment_frame_overlay_pub = rospy.Publisher(
            topic_out_segment_keypoint_frame_overlay, Image, queue_size=1
        )

        # Subscriber for camera intrinsics topic
        self.camera_info_sub = rospy.Subscriber(
            camera_info_topic, CameraInfo, self.on_camera_info, queue_size=1
        )
        
        self.reward_pub = rospy.Publisher('reward', Float64MultiArray, queue_size=1)
        self.loss = 0.1
        self.c_loss = 0.1
        self.ee_rot_loss = 10
        self.ee_trans_loss = 0.1
        self.hand_rot_loss = 10
        self.hand_trans_loss = 0.1
        self.reward = -100
        self.next_state = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.done = 0
        self.action_pose = [0,0,0,0,0,0]
        self.raw_action_pose = 100
        s = rospy.Service('/next_state_reward', Dfnext, self.step)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
        
        # Setting evaluation network
        
        self.count = 0
        self.num_objects = 1 
        self.objlist = [16]
        self.num_points = 500
        self.iteration = 4
        self.bs = 1
        
        root_dir = '/home/robot_ws/src/code/pose/weights/dis'
        
        self.dataset_config_dir = '/home/robot_ws/src/code/pose/datasets/linemod/dataset_config'
        
        
        self.seg = SegNet()
        self.seg.cuda()
        self.seg.load_state_dict(torch.load(root_dir + '/segnet.pth'))
        self.seg.eval()
        
        self.estimator = PoseNet(num_points = self.num_points, num_obj = self.num_objects)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load(root_dir + '/pose.pth'))
        self.estimator.eval()
        
        self.refiner = PoseRefineNet(num_points = self.num_points, num_obj = self.num_objects)
        self.refiner.cuda()
        self.refiner.load_state_dict(torch.load(root_dir + '/refine.pth'))
        
        self.refiner.eval()
        
    

    def on_image(self, image):
        self.cv_image = self.bridge.imgmsg_to_cv2(image, "rgb8")

    def on_depth(self, depth):
        self.cv_depth = self.bridge.imgmsg_to_cv2(depth, "32SC1")
        

    # camera info message
    def on_camera_info(self, camera_info):
        # Create camera intrinsics matrix
        self.camera_K = np.array([[614.04742431640625,0,320.44989013671875],[0,614.044677734375,244.81730651855469],[0,0,1]])

    def semantic_image(self):
        if self.cv_image is None :
            return

        seg_rgb = np.transpose(self.cv_image, (2, 0, 1))
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        seg_rgb = self.norm(torch.from_numpy(seg_rgb.astype(np.float32)))
        
        seg_rgb = seg_rgb.unsqueeze_(0).cuda()
        predict = self.seg(seg_rgb)

        predict = predict.squeeze(dim=0)
        self.mask = torch.argmax(predict, dim=0)
        

        self.mask = self.mask * 255
        self.mask = self.mask.cpu().detach().numpy()

        if np.max(self.mask) == 0:
            self.is_there_segment_image = False
            
        else :
            self.segment_image = self.mask.astype(np.int8)
            self.is_there_segment_image = True
            

    def process_image(self):
        if self.is_there_segment_image == False :
            return 

        testdataset = PoseDataset('eval', self.num_points, False, self.cv_image, self.cv_depth, self.mask , 0.0, True)
        testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

        diameter = []
        meta_file = open('{0}/models_info.yml'.format(self.dataset_config_dir), 'r')
    
        meta = yaml.safe_load(meta_file)
        for obj in self.objlist:
            diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)

        num_count = [0 for i in range(self.num_objects)]
        temp = 0
        for i, datass in enumerate(testdataloader, 0):
            temp += 1
            points, choose, img, idx = datass
            if len(points.size()) == 2:
                print('No.{0} NOT Pass! Lost detection!'.format(i))
                continue
            points, choose, img, idx = Variable(points).cuda(), \
                                        Variable(choose).cuda(), \
                                        Variable(img).cuda(), \
                                        Variable(idx).cuda()

            pred_r, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)
            pred_c = pred_c.view(self.bs, self.num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(self.bs * self.num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points.view(self.bs * self.num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

            self.confidence = how_max[0].item()


            for _ in range(0, self.iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(self.num_points, 1).contiguous().view(1, self.num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_points = torch.bmm((points - T), R).contiguous()
                
                pred_r, pred_t = self.refiner(new_points, emb, idx)


                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)
                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_r = my_r_final
                my_t = my_t_final

            my_r = quaternion_matrix(my_r)[:3, :3]
            
            self.rot = my_r
            self.t = my_t 
            num_count[idx[0].item()] += 1

    def get_quaternion_rotation_matrix(self,Q_init, switch_w=True):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                This rotation matrix converts a point in the local reference
                frame to a point in the global reference frame.
        """
        # Extract the values from Q

        if switch_w:
            Q = np.insert(Q_init[:3], 0, Q_init[-1])  # put w to first place
        else:
            Q = Q_init  # w already at the first place

        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])

        return rot_matrix

    def step(self, req):
        response = DfnextResponse()
        response.next_state = np.array(self.next_state)
        response.reward = self.reward
        response.done = self.done
        response.action_srv = self.action_pose
        response.raw_action_srv = self.raw_action_pose
        response.loss = self.c_loss
        response.ee_rot_loss = self.ee_rot_loss
        response.ee_trans_loss = self.ee_trans_loss
        response.hand_rot_loss = self.hand_rot_loss
        response.hand_trans_loss = self.hand_trans_loss
        return response

          
    def adaptive_reward(self, loss, initial_loss, raw_action_pose):

        if raw_action_pose == 12:
            if loss >= initial_loss*4:
                reward = 1
            elif initial_loss*4 > loss and loss >= initial_loss*2:
                reward = 2
            elif initial_loss*2 > loss and loss >= initial_loss:
                reward = 4
            else:
                reward = 8

        else:
            reward = 0
        return reward  

    

    def draw_bounding_box(self, model_ply):
        global offset, fail, done, action_pose, raw_action_pose, reward, next_state, initial_state_num, file_num, initial_loss
        if self.is_there_segment_image is False or self.cv_image is None or self.camera_K is None:
            return

        draw_pro_points = False

        x= 33.19151000/1000
        y= 56.44355000/1000
        z= 79.01963800/1000
        min_z = 0.00194200/1000
        max_z = 159.121416/1000
 
        model_points = np.array([[x,y,max_z],
                         [x,y,min_z],
                         [x,-y,max_z],
                         [x,-y,min_z],
                         [-x,y,max_z],
                         [-x,y,min_z],
                         [-x,-y,max_z],
                         [-x,-y,min_z],
                         [0, 0, 0],
                         [0.3, 0, 0],
                         [0, 0.3, 0],
                         [0, 0, 0.3]])
        
        df_to_kinvoa_rot = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1],])
  
        rrot = (self.rot @ df_to_kinvoa_rot.T)
        tt = self.t.copy()
        tt = np.array([tt])
        tt = tt.T

        RT = np.column_stack((rrot, tt))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))

        # Pred cam to base
        listener = tf.TransformListener()
        listener.waitForTransform("/base_link", "/end_effector_link", rospy.Time(0), rospy.Duration(3.0))
        (trans,rot) = listener.lookupTransform('/base_link', '/end_effector_link', rospy.Time(0))
        tvec = np.array([trans[0],trans[1],trans[2]])
        r = np.array([rot[0],rot[1],rot[2],rot[3]])

        R = self.get_quaternion_rotation_matrix(r, switch_w=True)
        base_to_ee=np.column_stack((R,tvec))
        base_to_ee= np.row_stack((base_to_ee, np.array([0.0, 0.0, 0.0, 1.0])))
        cam_to_base = RT @ np.linalg.inv(base_to_ee)
        print("Prediction cam_to_base",cam_to_base)

         # GT cam to base
        listener.waitForTransform("/camera_depth_optical_frame", "/base_link", rospy.Time(0), rospy.Duration(3.0))
        (trans,rot) = listener.lookupTransform('/camera_depth_optical_frame', '/base_link', rospy.Time(0))
        tvec = np.array([trans[0],trans[1],trans[2]])
        r = np.array([rot[0],rot[1],rot[2],rot[3]])
        R = self.get_quaternion_rotation_matrix(r, switch_w=True)
        cam_to_base_gt = np.column_stack((R,tvec))
        cam_to_base_gt= np.row_stack((cam_to_base_gt, np.array([0.0, 0.0, 0.0, 1.0])))
        print("GT cam_to_base",cam_to_base_gt)

        # GT cam to ee
        listener.waitForTransform("/camera_depth_optical_frame", "/end_effector_link", rospy.Time(0), rospy.Duration(3.0))
        (trans,rot) = listener.lookupTransform('/camera_depth_optical_frame', '/end_effector_link', rospy.Time(0))
        tvec = np.array([trans[0],trans[1],trans[2]])
        r = np.array([rot[0],rot[1],rot[2],rot[3]])
        R = self.get_quaternion_rotation_matrix(r, switch_w=True)
        cam_to_ee_gt = np.column_stack((R,tvec))
        cam_to_ee_gt= np.row_stack((cam_to_ee_gt, np.array([0.0, 0.0, 0.0, 1.0])))

        pred_r = RR.from_matrix(RT[:3,:3])
        pred_r = pred_r.as_euler('xyz', degrees=True)
        pred_coord  = [RT[0,3],RT[1,3],RT[2,3],pred_r[0],pred_r[1],pred_r[2]]

        # ee pose result
        pred_loss = (model_ply) @ RT[:3,:3].T
        pred_loss = np.add(pred_loss, RT[:3,3])

        gt_loss = (model_ply) @ cam_to_ee_gt[:3,:3].T
        gt_loss = np.add(gt_loss, cam_to_ee_gt[:3,3])
        
        loss = np.mean(np.linalg.norm((pred_loss - gt_loss),axis=1))
        print("add loss", loss)


        cam_to_ee_t = np.transpose(RT[:3,:3])
        ee_rot_loss = cam_to_ee_gt[:3,:3] @ cam_to_ee_t
        ee_rot_loss = np.rad2deg(np.arccos((np.trace(ee_rot_loss) - 1) / 2))
        # print("EE Rotation loss",ee_rot_loss)

        ee_trans_loss = np.sqrt((RT[0,3]-cam_to_ee_gt[0,3])**2+(RT[1,3]-cam_to_ee_gt[1,3])**2+(RT[2,3]-cam_to_ee_gt[2,3])**2)
        # print("EE Translation loss",ee_trans_loss)


        # hand eye result
        cam_to_base_t = np.transpose(cam_to_base[:3,:3])
        hand_rot_loss = cam_to_base_gt[:3,:3] @ cam_to_base_t
        hand_rot_loss = np.rad2deg(np.arccos((np.trace(hand_rot_loss) - 1) / 2))
        # print("Hand Rotation loss",hand_rot_loss)

        hand_trans_loss = np.sqrt((cam_to_base[0,3]-cam_to_base_gt[0,3])**2+(cam_to_base[1,3]-cam_to_base_gt[1,3])**2+(cam_to_base[2,3]-cam_to_base_gt[2,3])**2)
        # print("Hand Translation loss",hand_trans_loss)
        print("raw_action", raw_action_pose)

        if offset == 1:
                initial_loss = loss
        print("initial_loss", initial_loss)
	
        self.next_state = [next_base_to_tool_rt[0],next_base_to_tool_rt[1],next_base_to_tool_rt[2],next_base_to_tool_rt[3],next_base_to_tool_rt[4],next_base_to_tool_rt[5], pred_coord[0],pred_coord[1],pred_coord[2],pred_coord[3],pred_coord[4],pred_coord[5], offset-1]
        

        self.c_loss = loss
        self.ee_rot_loss = ee_rot_loss
        self.ee_trans_loss = ee_trans_loss
        self.hand_rot_loss = hand_rot_loss
        self.hand_trans_loss = hand_trans_loss

        self.reward = self.adaptive_reward(loss, initial_loss, raw_action_pose)
        if offset == 2 and raw_action_pose == 12:
            self.reward = -1
        

        self.action_pose = action_pose
        self.raw_action_pose = raw_action_pose
        print("action", self.action_pose)
        print("raw action", self.raw_action_pose)
        print("reward", self.reward)
        print("next_state", self.next_state)
        

        if offset % 11 == 0 or raw_action_pose == 12 or (offset==2 and raw_action_pose == 12):# or self.loss <= 0.001: #or loss<=0.001:
            self.done = 1
            offset = 0
            self.loss = 0.1
            reward = -100
            file_num += 1

        else:
            self.done = 0
            reward = self.reward

        done = self.done
        
        next_state = self.next_state


        self.step(None)

        
        if offset % 11 == 0 or raw_action_pose == 12 or (offset==2 and raw_action_pose == 12):
            next_state = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            action_pose = [0,0,0,0,0,0]
            raw_action_pose = 100

        self.loss = loss

        
        pred = (model_points) @ rrot.T

        # dense fusion coordinate to kinova coordinate 
       
        pred = np.add(pred, self.t)
        
        imgpoints = self.K @ pred.T
        imgpoints = imgpoints.T

        # ~~~~~~~~~~~~ point cloud plot ~~~~~~~~~~~~#  
        if draw_pro_points == True:
            img_pro_points = self.K @ self.pro_points.squeeze().T
            img_pro_points = img_pro_points.T

            z = np.array([img_pro_points[:, 2]]).T
            img_pro_points_uv = img_pro_points / z 
        
    

        for i in range(len(model_points)):
            imgpoints[i] /= imgpoints[i][2]


        img1 = self.cv_image.copy()

        color = (255,0,0)

        img1 = cv2.circle(img1, (int(imgpoints[0][0]),int(imgpoints[0][1])),3,color,-1)
        img1 = cv2.circle(img1, (int(imgpoints[1][0]),int(imgpoints[1][1])),3,color,-1)
        img1 = cv2.circle(img1, (int(imgpoints[2][0]),int(imgpoints[2][1])),3,color,-1)
        img1 = cv2.circle(img1, (int(imgpoints[3][0]),int(imgpoints[3][1])),3,color,-1)
        img1 = cv2.circle(img1, (int(imgpoints[4][0]),int(imgpoints[4][1])),3,color,-1)
        img1 = cv2.circle(img1, (int(imgpoints[5][0]),int(imgpoints[5][1])),3,color,-1)
        img1 = cv2.circle(img1, (int(imgpoints[6][0]),int(imgpoints[6][1])),3,color,-1)
        img1 = cv2.circle(img1, (int(imgpoints[7][0]),int(imgpoints[7][1])),3,color,-1)

        img1 = cv2.line(img1, (int(imgpoints[0][0]),int(imgpoints[0][1])), (int(imgpoints[1][0]),int(imgpoints[1][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[2][0]),int(imgpoints[2][1])), (int(imgpoints[3][0]),int(imgpoints[3][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[4][0]),int(imgpoints[4][1])), (int(imgpoints[5][0]),int(imgpoints[5][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[6][0]),int(imgpoints[6][1])), (int(imgpoints[7][0]),int(imgpoints[7][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[2][0]),int(imgpoints[2][1])), (int(imgpoints[6][0]),int(imgpoints[6][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[3][0]),int(imgpoints[3][1])), (int(imgpoints[7][0]),int(imgpoints[7][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[0][0]),int(imgpoints[0][1])), (int(imgpoints[4][0]),int(imgpoints[4][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[1][0]),int(imgpoints[1][1])), (int(imgpoints[5][0]),int(imgpoints[5][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[0][0]),int(imgpoints[0][1])), (int(imgpoints[2][0]),int(imgpoints[2][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[4][0]),int(imgpoints[4][1])), (int(imgpoints[6][0]),int(imgpoints[6][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[1][0]),int(imgpoints[1][1])), (int(imgpoints[3][0]),int(imgpoints[3][1])), color)
        img1 = cv2.line(img1, (int(imgpoints[5][0]),int(imgpoints[5][1])), (int(imgpoints[7][0]),int(imgpoints[7][1])), color)

        self.cv_image_with_bbox = img1

    def publish_pose(self):
        global fail, offset, restart_cnt, done
        if self.is_there_segment_image is False or self.cv_image is None:
            print('realsense camera fail')
            fail = True
            offset -= 1
            return
        
        ssim_score = ssim(self.segment_image_org, self.segment_image, channel_axis=1, full=True)
        if ssim_score==1.0:
             return

        # self.cv_iamge_with_bbox_org = self.cv_image_with_bbox.copy()
        cv_image_overlay = self.cv_image_with_bbox.copy()
        cv_image_overlay = cv_image_overlay[:, :, ::-1]

        image_overlay_msg = self.bridge.cv2_to_imgmsg(
            cv_image_overlay, encoding="bgr8"
        )

        self.segment_image_org = self.segment_image.copy()
        cv_image_segment = self.segment_image.copy()     

        image_segment_msg = self.bridge.cv2_to_imgmsg(
            cv_image_segment, encoding="passthrough"
        ) # segment msg 

        self.segment_frame_overlay_pub.publish(image_segment_msg) #segment 
        self.kp_frame_overlay_pub.publish(image_overlay_msg)


class ExampleCartesianActionsWithNotifications:
    def __init__(self):
        try:
            # rospy.init_node('example_cartesian_poses_with_notifications_python')

            self.HOME_ACTION_IDENTIFIER = 2

            self.action_topic_sub = None
            self.all_notifs_succeeded = True

            self.all_notifs_succeeded = True

            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3_lite")

            rospy.loginfo("Using robot_name " + self.robot_name)

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            # Init the services
            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            set_cartesian_reference_frame_full_name = '/' + self.robot_name + '/control_config/set_cartesian_reference_frame'
            rospy.wait_for_service(set_cartesian_reference_frame_full_name)
            self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)
            
            self.listener = tf.TransformListener()
            self.listener.waitForTransform("/camera_depth_optical_frame", "/end_effector_link", rospy.Time(0), rospy.Duration(3.0))


            set_action_pose_name = '/action_pose'
            rospy.wait_for_service(set_action_pose_name)
            self.set_action_pose = rospy.ServiceProxy(set_action_pose_name, Dfaction)

        except:
            self.is_init_success = False
        else:
            self.is_init_success = True

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                self.all_notifs_succeeded = False
                return False
            else:
                time.sleep(0.01)

    def example_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True

    def example_home_the_robot(self):
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                return self.wait_for_action_end_or_abort()

    def example_set_cartesian_reference_frame(self):
        # Prepare the request with the frame we want to set
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED

        # Call the service
        try:
            self.set_cartesian_reference_frame()
        except rospy.ServiceException:
            rospy.logerr("Failed to call SetCartesianReferenceFrame")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")
            return True

        # Wait a bit
        rospy.sleep(0.25)

    def example_subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.sleep(1.0)

        return True
    
    def get_quaternion_rotation_matrix(self, Q_init, switch_w=True):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                This rotation matrix converts a point in the local reference
                frame to a point in the global reference frame.
        """
        # Extract the values from Q

        if switch_w:
            Q = np.insert(Q_init[:3], 0, Q_init[-1])  # put w to first place
        else:
            Q = Q_init  # w already at the first place

        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])

        return rot_matrix

    def step(self):
        try:
            resp1 = self.set_action_pose(False)
        except:
            return np.zeros((6,)), np.zeros((6,))
        return resp1.action, resp1.raw_action

    def main(self, action, raw_action, next_base_to_ee_rt):
        global fail, offset, done, action_pose, raw_action_pose, restart_cnt
        # For testing purposes
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python")
        except:
            pass

        if success:
            print("offset!!!!!!!!",offset)

            if fail == True or offset == 0 or restart_cnt==0:
                #*******************************************************************************
                # Make sure to clear the robot's faults else it won't move if it's already in fault
                success &= self.example_clear_faults()
                #*******************************************************************************
                
                #*******************************************************************************
                # Start the example from the Home position
                success &= self.example_home_the_robot()
                #*******************************************************************************

                #*******************************************************************************
                # Set the reference frame to "Mixed"
                success &= self.example_set_cartesian_reference_frame()

                #*******************************************************************************
                # Subscribe to ActionNotification's from the robot to know when a cartesian pose is finished
                success &= self.example_subscribe_to_a_robot_notification()
                fail = False
            #*******************************************************************************

            # Prepare and send pose 1
            my_cartesian_speed = CartesianSpeed()
            my_cartesian_speed.translation = 1000 # m/s
            my_cartesian_speed.orientation = 360  # deg/s

            my_constrained_pose = ConstrainedPose()
            my_constrained_pose.constraint.oneof_type.speed.append(my_cartesian_speed)

            self.move = action
            self.raw_move = raw_action
            my_constrained_pose.target_pose.x = next_base_to_ee_rt[0] #random.uniform(0.2,0.3)
            my_constrained_pose.target_pose.y = next_base_to_ee_rt[1] #random.uniform(-0.2,0.2)
            my_constrained_pose.target_pose.z = next_base_to_ee_rt[2] #random.uniform(0.2,0.3)
            my_constrained_pose.target_pose.theta_x = next_base_to_ee_rt[3] #random.randint(-30,30)
            my_constrained_pose.target_pose.theta_y = next_base_to_ee_rt[4] #random.randint(160,200)
            my_constrained_pose.target_pose.theta_z = next_base_to_ee_rt[5] #random.randint(-30,30)

            action_pose = self.move
            raw_action_pose = self.raw_move

            req = ExecuteActionRequest()
            req.input.oneof_action_parameters.reach_pose.append(my_constrained_pose)
            req.input.name = "pose"
            req.input.handle.action_type = ActionType.REACH_POSE
            req.input.handle.identifier = 1000

            rospy.loginfo("Sending pose ...")
            self.last_action_notif_type = None

            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to send pose")
                success = False
            else:
                rospy.loginfo("Waiting for pose to finish...")

            self.wait_for_action_end_or_abort()

            success &= self.all_notifs_succeeded
            success &= self.all_notifs_succeeded
            offset += 1
            
        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python", success)

        if not success:
            rospy.logerr("The example encountered an error.")
        restart_cnt += 1

def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

def get_quaternion_rotation_matrix(Q_init, switch_w=True):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                This rotation matrix converts a point in the local reference
                frame to a point in the global reference frame.
        """
        # Extract the values from Q

        if switch_w:
            Q = np.insert(Q_init[:3], 0, Q_init[-1])  # put w to first place
        else:
            Q = Q_init  # w already at the first place

        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])

        return rot_matrix


def step():
        set_action_pose_name = '/action_pose'
        rospy.wait_for_service(set_action_pose_name)
        set_action_pose = rospy.ServiceProxy(set_action_pose_name, Dfaction)
        try:
            resp1 = set_action_pose(1)
        except:
            return np.zeros((6,)), 100
        return resp1.action, resp1.raw_action

if __name__ == '__main__':
    print("restart!!!!!!!!!!!!!!")
    file_num = 0
    # Parse input arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--dataset_root', 
        type=str, 
        default = '', 
        help='dataset root dir',
    )
    parser.add_argument(
        '--model',
        type=str, 
        default = '',  
        help='resume PoseNet model',
    )
    parser.add_argument(
        '--refine_model', 
        type=str, 
        default = '',  
        help='resume PoseRefineNet model',
    )
    parser.add_argument(
        '--seg_model',
        type=str, 
        default = '',  
        help='resume SegNet model',
    )
    parser.add_argument(
        "-r",
        "--node-rate",
        type=float,
        default=1.0,
        help="The rate in Hz for this node to run.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="initial state num",
    )
    args = parser.parse_args()
    initial_state_num = args.num

    print("initial_state_num",initial_state_num)

    # Initialize ROS node
    rospy.init_node("densefusion")

    single_frame_mode = True
    mode_str = "single-frame mode" if single_frame_mode else "multi-frame mode"
    dense_fusion_ros = DenseFusionInferenceRos(
        args, single_frame_mode
    )

    print("Dense Fusion Online " + mode_str)

    rate = rospy.Rate(args.node_rate)

    pt = ply_vtx('/home/robot_ws/src/code/pose/datasets/linemod/Linemod_preprocessed/models/obj_16.ply')
    model_points = pt / 1000.0
    dellist = [j for j in range(0, len(model_points))]
    dellist = random.sample(dellist, len(model_points) - 500)
    model_points = np.delete(model_points, dellist, axis=0)

    ex = ExampleCartesianActionsWithNotifications()
    listener = tf.TransformListener()
    while not rospy.is_shutdown():
        if done == True and file_num == 15:
            exit()
        
        if offset == 0:
            print("file_num!!!!!!!!!!!", file_num)
            next_base_to_tool_rt = np.array([random.uniform(0.2,0.3),random.uniform(-0.2,0.2),random.uniform(0.2,0.3),random.uniform(-20,20),random.uniform(160,200),random.uniform(-20,20)])
            print("next_base_to_tool_rt",next_base_to_tool_rt)
            ex.main(action_pose, raw_action_pose, next_base_to_tool_rt)

        else:
            action, raw_action = step()
            
            # Synchronization
            if raw_action == 100:
                continue
            
            if raw_action != 12:
                if action_pose[0] == action[0] and action_pose[1] == action[1] and action_pose[2] == action[2] and action_pose[3] == action[3] and action_pose[4] == action[4] and action_pose[5] == action[5]:
                    continue

            if raw_action == 0:
                if action[0] != 0.002:
                    continue
            elif raw_action == 1:
                if action[0] != -0.002:
                    continue
            elif raw_action == 2:
                if action[1] != 0.002:
                    continue
            elif raw_action == 3:
                if action[1] != -0.002:
                    continue
            elif raw_action == 4:
                if action[2] != 0.002:
                    continue
            elif raw_action == 5:
                if action[2] != -0.002:
                    continue
            elif raw_action == 6:
                if action[3] != 2:
                    continue
            elif raw_action == 7:
                if action[3] != -2:
                    continue
            elif raw_action == 8:
                if action[4] != 2:
                    continue
            elif raw_action == 9:
                if action[4] != -2:
                    continue
            elif raw_action == 10:
                if action[5] != 2:
                    continue
            elif raw_action == 11:
                if action[5] != -2:
                    continue
            else:
                if action[0] != 0 or action[1] != 0 or action[2] != 0 or action[3] != 0 or action[4] != 0 or action[5] != 0:
                    continue
                    
            action = np.add(action, next_state[:6])
            next_base_to_tool_rt = action
            print(next_base_to_tool_rt)
            ex.main(action, raw_action, next_base_to_tool_rt)

        dense_fusion_ros.semantic_image() 
        
        # find R|t
        dense_fusion_ros.process_image()

        # draw bbox
        dense_fusion_ros.draw_bounding_box(model_points)

        # Publish Image 
        dense_fusion_ros.publish_pose()

        rate.sleep()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 

