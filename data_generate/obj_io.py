import numpy as np


def parse_obj_file(open_file):
    """
    Parse the supplied file.

    Args:
        `open_file`: file-like object with `readlines` method.

    Returns: positions, face_positions, texcoords, face_texcoords, \
        normals, face_normals
    """
    positions = []
    texcoords = []
    normals = []
    face_positions = []
    face_texcoords = []
    face_normals = []

    def parse_face(values):
        if len(values) != 3:
            raise ValueError('not a triangle at line' % lineno)
        for v in values:
            for j, index in enumerate(v.split('/')):
                if len(index):
                    if j == 0:
                        face_positions.append(int(index) - 1)
                    elif j == 1:
                        face_texcoords.append(int(index) - 1)
                    elif j == 2:
                        face_normals.append(int(index) - 1)

    parse_fns = {
        'v': lambda values: positions.append([float(x) for x in values]),
        'vt': lambda values: texcoords.append([float(x) for x in values]),
        'vn': lambda values: normals.append([float(x) for x in values]),
        'f': parse_face,
        'mtllib': lambda values: None,
        'o': lambda values: None,
        'usemtl': lambda values: None,
        's': lambda values: None,
        'newmtl': lambda values: None,
        'Ns': lambda values: None,
        'Ni': lambda values: None,
        'Ka': lambda values: None,
        'Kd': lambda values: None,
        'Ks': lambda values: None,
        'd': lambda values: None,
        'illum': lambda values: None,
        'map_Kd': lambda values: None,
    }

    def parse_line(line):
        line = line.strip()
        if len(line) > 0 and line[0] != '#':
            values = line.split(' ')
            code = values[0]
            values = values[1:]
            if code in parse_fns:
                parse_fns[code](values)

    for lineno, line in enumerate(open_file.readlines()):
        parse_line(line)

    positions = np.array(positions, dtype=np.float32)
    texcoords = np.array(texcoords, dtype=np.float32) if len(texcoords) > 0 \
        else None
    normals = np.array(normals, dtype=np.float32) \
        if len(normals) > 0 else None
    face_positions = np.array(face_positions, dtype=np.uint32).reshape(-1, 3)

    face_texcoords = np.array(face_texcoords, dtype=np.uint32).reshape(-1, 3) \
        if len(face_texcoords) > 0 else None
    face_normals = np.array(face_normals, dtype=np.uint32).reshape(-1, 3) \
        if len(face_normals) > 0 else None

    return positions, face_positions, texcoords, face_texcoords, \
        normals, face_normals


def parse_obj(file_or_filename):
    """
    Parse the given file, or opens the file at filename.

    See `parse_obj_file` for return values.
    """
    if hasattr(file_or_filename, 'readlines'):
        return parse_obj_file(file_or_filename)
    else:
        with open(file_or_filename, 'r') as f:
            return parse_obj_file(f)


def write_obj_file(fp, vertices, faces):
    for vertex in vertices:
        fp.write('v %s\n' % ' '.join((str(v) for v in vertex)))
    for face in faces:
        fp.write('f %s\n' % ' '.join((str(f+1) for f in face)))


def write_obj(path_or_fp, vertices, faces):
    if hasattr(path_or_fp, 'write'):
        write_obj_file(path_or_fp, vertices, faces)
    else:
        with open(path_or_fp, 'w') as fp:
            write_obj_file(fp, vertices, faces)
