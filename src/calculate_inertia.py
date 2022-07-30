import os
import trimesh
import meshio
import numpy as np
import xml.etree.ElementTree as ET


def parse_dea_file(mesh_file):
    root = ET.parse(mesh_file).getroot()

    library_geometrie = root.find(
        '{http://www.collada.org/2005/11/COLLADASchema}library_geometries'
    )
    geometry = library_geometrie.find(
        '{http://www.collada.org/2005/11/COLLADASchema}geometry'
    )
    mesh = geometry.find(
        '{http://www.collada.org/2005/11/COLLADASchema}mesh'
    )
    float_array = mesh.find(
        '{http://www.collada.org/2005/11/COLLADASchema}source'
    ).find(
        '{http://www.collada.org/2005/11/COLLADASchema}float_array'
    )
    triangles = mesh.find(
        '{http://www.collada.org/2005/11/COLLADASchema}triangles'
    ).find(
        '{http://www.collada.org/2005/11/COLLADASchema}p'
    ).text.split(' ')

    points = float_array.text.split(' ')

    cells = {
        'triangle': np.array(
            [
                [int(triangles[i + j]) for j in range(3)]
                for i in range(0, len(triangles), 3)
            ]
        )
    }
    points = np.array(
        [
            [float(points[i + j]) for j in range(3)]
            for i in range(0, len(points), 3)
        ]
    )
    return points, cells


def get_mesh_file(mesh_file, to='stl'):
    points, cells = parse_dea_file(mesh_file)
    mesh = meshio.Mesh(
        points,
        cells,
    )
    mesh.write(
        f'{mesh_file.split(".")[0]}.stl',
    )
    return f'{mesh_file.split(".")[0]}.stl'


def get_inertia(mesh_file):
    mesh_file = get_mesh_file(mesh_file)
    mesh = trimesh.load(mesh_file)
    os.remove(mesh_file)
    return mesh.moment_inertia


if __name__ == '__main__':
    os.chdir('data/objects/Winter/models/collada')
    for file in os.listdir():
        if os.path.isfile(file) and file.split('.')[1] == 'dae':
            get_mesh_file(file)
            get_inertia(f'{file.split(".")[0]}.stl')
            os.remove(f'{file.split(".")[0]}.stl')
