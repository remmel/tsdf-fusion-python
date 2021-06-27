"""Fuse RGB-D images from RGB + D(png) images. Poses (1 csv file) from 3DRecorder format (translation+quaternion)
"""



# pip3 install --user numpy-quaternion

import time

import cv2
import numpy as np
import quaternion

import fusion
import csv


def quaternion_multiply(q1, q0):
  w0, x0, y0, z0 = q0.w, q0.x, q0.y, q0.z
  w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
  arr=np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                   x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                   -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                   x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
  return np.quaternion(arr[0], arr[1], arr[2], arr[3])


def toTransformationMatrix4(p):
  q = np.quaternion(float(p['qw']), float(p['qx']), float(p['qy']), float(p['qz']))
  q3 = quaternion_multiply(q, np.quaternion(0,1,0,0))
  m3x3 = quaternion.as_rotation_matrix(q3)
  m4x3 = np.append(m3x3, [[float(p['tx'])], [float(p['ty'])], [float(p['tz'])]], axis=1)
  m4x4 = np.append(m4x3, [[0,0,0,1]], axis=0)
  return m4x4

if __name__ == "__main__":
  dir = "/home/remmel/workspace/dataset/2021-04-09_190810_myoffice/240x180" #using 640x480 or 240x180 provides same result (note that 640x480 depths pngs are upscaled from 240x180)
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 50
  cam_intr = np.loadtxt(dir + "/camera-intrinsics.txt", delimiter=' ')

  poses = []
  with open(dir + '/../poses.csv', newline='\n') as f:
    reader = csv.DictReader(f)
    for row in reader:
      poses.append(row)


  poses = poses[0:n_imgs] # limit number of poses

  vol_bnds = np.zeros((3,2))
  for p in poses:
    # Read depth image and camera pose
    depth_im = cv2.imread(dir + "/"+p['frame']+"_depth16.bin.png",-1).astype(float)
    depth_im[0:3, ]=0.0 # ignore first rows
    depth_im /= 5000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    # 4x4 ndarray
    # cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(1))  # 4x4 rigid transformation matrix
    cam_pose = toTransformationMatrix4(p)

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.03)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  i = 0
  for p in poses:
    print("Fusing frame %d/%d"%(i+1, len(poses)))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(dir + "/"+p['frame']+"_image.jpg"), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(dir + "/"+p['frame']+"_depth16.bin.png",-1).astype(float)
    depth_im /= 5000.
    depth_im[depth_im > 2.0] = 0
    # cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))
    cam_pose = toTransformationMatrix4(p)

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    i += 1

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)

