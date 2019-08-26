import os,sys
import numpy as np
import cv2
from obj_io import parse_obj_file
import sklearn.preprocessing

def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi)
    temp = param[3]*np.cos(phi)
    camX = temp * np.cos(theta)    
    camZ = temp * np.sin(theta)        
    cam_pos = np.array([camX, camY, camZ])        

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
    return cam_mat, cam_pos

def sample_triangle(v, n=None):
    if hasattr(n, 'dtype'):
        n = np.asscalar(n)
    if n is None:
        size = v.shape[:-2] + (2,)
    elif isinstance(n, int):
        size = (n, 2)
    elif isinstance(n, tuple):
        size = n + (2,)
    elif isinstance(n, list):
        size = tuple(n) + (2,)
    else:
        raise TypeError('n must be int, tuple or list, got %s' % str(n))
    assert(v.shape[-2] == 2)
    a = np.random.uniform(size=size)
    mask = np.sum(a, axis=-1) > 1
    a[mask] *= -1
    a[mask] += 1
    a = np.expand_dims(a, axis=-1)
    return np.sum(a*v, axis=-2)

def sample_faces(vertices, faces, n_total):
    if len(faces) == 0:
        raise ValueError('Cannot sample points from zero faces.')
    tris = vertices[faces]
    n_faces = len(faces)
    d0 = tris[..., 0:1, :]
    ds = tris[..., 1:, :] - d0
    assert(ds.shape[1:] == (2, 3))
    areas = 0.5 * np.sqrt(np.sum(np.cross(ds[:, 0], ds[:, 1])**2, axis=-1))
    cum_area = np.cumsum(areas)
    cum_area *= (n_total / cum_area[-1])
    cum_area = np.round(cum_area).astype(np.int32)

    positions = []
    last = 0
    for i in range(n_faces):
        n = cum_area[i] - last
        last = cum_area[i]
        if n > 0:
            positions.append(d0[i] + sample_triangle(ds[i], n))
    return np.concatenate(positions, axis=0)

def sample_single(obj_path, view_path, output_folder):
    # 1 sampling
    with open(obj_path,'r') as f:
        vertices, faces = parse_obj_file(f)[:2]
    sample = sample_faces(vertices, faces, 16384)

    # 2 tranform to camera view   
    position = sample * 0.57
    cam_params = np.loadtxt(view_path)

    for index, param in enumerate(cam_params):
        cam_mat, cam_pos = camera_info(param)
        pt_trans = np.dot(position-cam_pos, cam_mat.transpose())
        pt_trans = pt_trans.astype(np.float32)
        npy_path = os.path.join(output_folder,'{0:02d}.npy'.format(int(index)))
        np.save(npy_path, pt_trans)
        # np.savetxt(npy_path.replace('npy','xyz'), pt_trans)


if __name__ == '__main__':
    obj_path = '1a6f615e8b1b5ae4dbbc9440457e303e/model.obj'
    view_path = '1a6f615e8b1b5ae4dbbc9440457e303e/rendering/rendering_metadata.txt'
    output_folder = '1a6f615e8b1b5ae4dbbc9440457e303e/temp'
    sample_single(obj_path,view_path,output_folder)
    

    # # 1 sampling
    # obj_path = '1a6f615e8b1b5ae4dbbc9440457e303e/model.obj'
    # mesh_list = trimesh.load_mesh(obj_path)
    # if not isinstance(mesh_list, list):
    #     mesh_list = [mesh_list]

    # area_sum = 0
    # for mesh in mesh_list:
    #     area_sum += np.sum(mesh.area_faces)

    # # sample_16k = np.zeros((0,3), dtype=np.float32)
    # sample = np.zeros((0,3), dtype=np.float32)
    # # normal = np.zeros((0,3), dtype=np.float32)
    # total = 0
    # for mesh in mesh_list:
    #     # number_16k = int(round(16384*np.sum(mesh.area_faces)/area_sum))
    #     number = int(round(1024*np.sum(mesh.area_faces)/area_sum))
    #     if number < 1:
    #         continue
    #     # points_16k, _index_16k = trimesh.sample.sample_surface(mesh, number_16k)
    #     points, _index = trimesh.sample.sample_surface(mesh, number)

    #     # sample_16k = np.append(sample_16k, points_16k, axis=0)
    #     sample = np.append(sample, points, axis=0)

    #     # triangles = mesh.triangles[index]
    #     # pt1 = triangles[:,0,:]
    #     # pt2 = triangles[:,1,:]
    #     # pt3 = triangles[:,2,:]
    #     # norm = np.cross(pt3-pt1, pt2-pt1)
    #     # norm = sklearn.preprocessing.normalize(norm, axis=1)
    #     # normal = np.append(normal, norm, axis=0)

    # # 2 tranform to camera view
    # # position_16k = sample_16k * 0.57
    # position = sample * 0.57

    # view_path = '1a6f615e8b1b5ae4dbbc9440457e303e/rendering/rendering_metadata.txt'
    # cam_params = np.loadtxt(view_path)
    # for index, param in enumerate(cam_params):
    #     # camera tranform
    #     cam_mat, cam_pos = camera_info(param)

    #     # pt_trans_16k = np.dot(position_16k-cam_pos, cam_mat.transpose())
    #     pt_trans = np.dot(position-cam_pos, cam_mat.transpose())
    #     # nom_trans = np.dot(normal, cam_mat.transpose())
    #     # train_data = np.hstack((pt_trans, nom_trans))
    #     # train_data = pt_trans
        
    #     img_path = os.path.join(os.path.split(view_path)[0], '%02d.png'%index)
    #     # np.savetxt(img_path.replace('png','xyz'), train_data)

    #     # np.savetxt(img_path.replace('png','xyz'), pt_trans_16k)
    #     np.savetxt(img_path.replace('png','xyz'), pt_trans)
        
    #     # #### project for sure
    #     # img = cv2.imread(img_path)
        
    #     # X,Y,Z = pt_trans.T
    #     # F = 284
    #     # h = (-Y)/(-Z)*F + 256/2.0
    #     # w = X/(-Z)*F + 256/2.0
    #     # h = np.minimum(np.maximum(h, 0), 255)
    #     # w = np.minimum(np.maximum(w, 0), 255)
    #     # img[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
    #     # img[np.round(h).astype(int), np.round(w).astype(int), 1] = 255
    #     # cv2.imwrite(img_path.replace('.png','_prj.png'), img)
