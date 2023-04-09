import os
import argparse
import numpy as np
import torch
import cv2
import trimesh
from matplotlib import pyplot as plt
from fusion import TSDFVolumeTorch
from dataset.tum_rgbd import TUMDataset, TUMDatasetOnline
from tracker import ICPTracker
from utils import load_config, get_volume_setting, get_time
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument('--config', type=str, default="configs/fr1_desk.yaml", help='Path to config file.')
    parser.add_argument("--save_dir", type=str, default=None, help="Directory of saving results.")
    args = load_config(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    dataset = TUMDataset(os.path.join(args.data_root), device, near=args.near, far=args.far, img_scale=0.25, depth_scale= args.depth_scale)
    H, W = dataset.H, dataset.W

    vol_dims, vol_origin, voxel_size = get_volume_setting(args)
    tsdf_volume = TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True)
    icp_tracker = ICPTracker(args, device)

    t, poses, poses_gt = list(), list(), list()
    curr_pose, depth1, color1 = None, None, None

    # save all the T10 for debug
    T10_results = list()
    T10 = None

    for i in range(0, len(dataset), 1):
        t0 = get_time()
        sample = dataset[i]
        color0, depth0, pose_gt, K = sample  # use live image as template image (0)
        # depth0[depth0 <= 0.5] = 0.

        if i == 0:  # initialize
            # make the first pose (estimated) equal to the gt
            curr_pose = torch.tensor([[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]], device=device)
            T10 = curr_pose
        else:  # tracking
            # 1. render depth image (1) from tsdf volume
            depth1, color1, vertex01, normal1, mask1 = tsdf_volume.render_model(curr_pose, K, H, W, near=args.near, far=args.far, n_samples=args.n_steps)
            T10 = icp_tracker(depth0, depth1, K)  # transform from 0 to 1
            curr_pose = curr_pose @ T10

        # fusion
        tsdf_volume.integrate(depth0,
                              K,
                              curr_pose,
                              obs_weight=1.,
                              color_img=color0
                              )
        t1 = get_time()
        t += [t1 - t0]
        print("processed frame: {:d}, time taken: {:f}s".format(i, t1 - t0))
        poses += [curr_pose.cpu().numpy()]

    avg_time = np.array(t).mean()
    print("average processing time: {:f}s per frame, i.e. {:f} fps".format(avg_time, 1. / avg_time))

    # get the pose list output by KinectFusion
    poses = np.stack(poses, 0)

    # get the time stamp (suppose the same as depth image)
    assoc_info = np.loadtxt("{}/associations.txt".format(args.data_root), dtype=str)
    depth_timestamp_list = assoc_info.T[2]

    # # convert to TUM traj
    pose_txt = []

    for i in range(len(poses)):
        # get the pose
        T_pose = poses[i]

        # get the time stamp
        timestamp = float(depth_timestamp_list[i])
        
        # get the quat
        rotation = R.from_matrix(T_pose[:3,:3])
        quat = list(rotation.as_quat())
        # get the translation vector
        translation = list(T_pose[:3,3])

        # generate pose list (tum format)
        pose_tum = [timestamp] + translation + quat

        pose_txt.append(pose_tum)

    # for saving the pose
    np.savetxt("test.txt", pose_txt, fmt='%.06f')

    # save results
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        # save the mesh model
        verts, faces, norms, colors = tsdf_volume.get_mesh()
        partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, vertex_colors=colors)
        partial_tsdf.export(os.path.join(args.save_dir, "mesh.ply"))
