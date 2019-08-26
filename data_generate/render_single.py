import os
import math
import bpy
import numpy as np

def main(obj_path, view_path, output_folder, scale, shape):

    def camera_info(param):
        theta = np.deg2rad(param[0])
        phi = np.deg2rad(param[1])

        camY = param[3]*np.sin(phi)
        temp = param[3]*np.cos(phi)
        camX = temp * np.cos(theta)    
        camZ = -temp * np.sin(theta)        
        cam_pos = np.array([camX, camZ, camY])        

        return cam_pos

    cam_params = np.loadtxt(view_path)

    bpy.data.objects['Cube'].select = True  #background black -> transparent or delete the original cube
    bpy.ops.object.delete()

    def merge_all():
        bpy.ops.object.select_by_type(type="MESH")
        bpy.context.scene.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        obj = bpy.context.scene.objects.active
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.location_clear() # Clear location - set location to (0,0,0)
        return obj

    bpy.ops.import_scene.obj(filepath=obj_path)

    #scale
    cur_obj = merge_all()
    cur_obj.scale = (scale, scale, scale)

    #remove texture
    obj = bpy.context.object
    mats = obj.data.materials

    for mat in mats:
        mat.user_clear()

    matLength = len(mats)
    for i in range(matLength):
        obj.data.materials.pop(0, update_data=True)

    leftOverMatBlocks = [block for block in bpy.data.materials if block.users == 0]
    for block in leftOverMatBlocks:
        bpy.data.materials.remove(block)
    bpy.context.scene.update()

    scene = bpy.context.scene
    #world light
    scene.world.light_settings.use_ambient_occlusion = True  # turn AO on
    scene.world.light_settings.ao_factor = 0.5  # set it to 0.5

    #render setting
    scene.render.resolution_x = shape[1]
    scene.render.resolution_y = shape[0]
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.image_settings.file_format = 'PNG'  # set output format to png

    def parent_obj_to_camera(b_camera):
        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        b_camera.parent = b_empty  # setup parenting
        return b_empty

    cam = scene.objects['Camera']
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    cam = bpy.data.objects["Camera"]


    for index, param in enumerate(cam_params):
        cam_pos = camera_info(param)
        scene.render.filepath = os.path.join(output_folder,'{0:02d}'.format(int(index)))
        cam.location = cam_pos
        bpy.ops.render.render(write_still=True)  # render still


def get_args():
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--obj_path', type=str, default='1a6f615e8b1b5ae4dbbc9440457e303e/model.obj',
                        help='Path to the obj file to be rendered.')
    parser.add_argument('--view_path', type=str, default='1a6f615e8b1b5ae4dbbc9440457e303e/rendering/rendering_metadata.txt',
                        help='Path to the view file which indicates camera position.')                    
    parser.add_argument('--output_folder', type=str, default='1a6f615e8b1b5ae4dbbc9440457e303e/temp/',
                        help='The path the output will be dumped to.')
    parser.add_argument('--scale', type=float, default=0.57,
                        help='Scaling factor applied to model. '
                             'Depends on size of mesh.')
    parser.add_argument('--shape', type=int, default=[256, 256], nargs=2,
                        help='2D shape of rendered images.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    return args

args = get_args()
main(args.obj_path, args.view_path, args.output_folder, args.scale, args.shape)
