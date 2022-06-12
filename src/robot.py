import xml.etree.cElementTree as ET
import pymeshlab
import os


class Geometry:

    def __init__(self, file_name, scale=None):
        self.file_name = file_name
        if scale:
            self.scale = ' '.join(
                list(
                    map(
                        lambda x: str(x),
                        scale,
                    )
                )
            )
        else:
            self.scale = ' '.join(
                list(
                    map(
                        lambda x: str(x),
                        [1, 1, 1],
                    )
                )
            )

    def get_urdf_element(self):
        mesh = ' '.join([
            '<mesh',
            f'filename="{self.file_name}"',
            f'scale="{self.scale}"',
            '/>',
        ])
        geometry_element = '\n'.join([
            '<geometry>',
            mesh,
            '</geometry>',
        ])
        return geometry_element


class Inertial:

    def __init__(self, mass, origin, inertia):
        self.mass = mass
        self.origin = origin
        self.inertia = inertia

    def get_urdf_element(self):
        mass_element = f'<mass value="{self.mass}"/>'
        xyz = f'xyz="{" ".join(str(value) for value in self.origin["xyz"])}"'
        rpy = f'rpy="{" ".join(str(value) for value in self.origin["rpy"])}"'
        origin_element = f'<origin {xyz} {rpy}/>'
        inertia_element = '<inertia '
        for key, value in self.inertia.items():
            inertia_element += f'{key}="{value}" '
        inertia_element += '/>'
        self.element = '\n'.join([
            '<inertial>',
            mass_element,
            origin_element,
            inertia_element,
            '</inertial>',
        ])
        return self.element


class Visual:

    def __init__(self, geometry):
        self.geometry = geometry

    def get_urdf_element(self):
        geometry_element = self.geometry.get_urdf_element()
        self.element = '\n'.join([
            '<visual>',
            geometry_element,
            '</visual>',
        ])
        return self.element


class Collision:

    def __init__(self, geometry):
        self.geometry = geometry

    def get_urdf_element(self):
        geometry_element = self.geometry.get_urdf_element()
        self.element = '\n'.join([
            '<collision>',
            geometry_element,
            '</collision>'
        ])
        return self.element


class Link:

    def __init__(self, name, inertial, visual, collision):
        self.name = name
        self.visual = visual
        self.collision = collision
        self.inertial = inertial

    def get_urdf_element(self):
        self.element = '\n'.join([
            f'<link name="{self.name}">',
            self.inertial.get_urdf_element() if self.inertial else '',
            self.visual.get_urdf_element() if self.visual else '',
            self.collision.get_urdf_element() if self.collision else '',
            '</link>',
        ])
        return self.element


class Joint:

    def __init__(
        self,
        name,
        type_,
        parent_link,
        child_link,
        axis,
        limit,
        origin,
        dynamics=None,
    ):
        self.name = name
        self.type_ = type_
        self.parent_link = parent_link
        self.child_link = child_link
        self.dynamics = dynamics
        self.limit = limit
        self.axis = axis
        self.origin = origin

    def get_urdf_element(self):
        origin_element = ''
        if self.origin:
            xyz = f'xyz="{" ".join(str(value) for value in self.origin["xyz"])}"'
            rpy = f'rpy="{" ".join(str(value) for value in self.origin["rpy"])}"'
            origin_element = f'<origin {xyz} {rpy}/>'
        parent_link_element = f'<parent link="{self.parent_link}"/>'
        child_link_element = f'<child link="{self.child_link}"/>'
        dynamic_element = ''
        if self.dynamics:
            dynamics_element = ' '.join([
                '<dynamics',
                f'damping="{self.dynamics["damping"]}',
                f'friction="{self.dynamics["friction"]}',
                '/>',
            ])
        limit_element = ''
        if self.limit:
            limit_element = ' '.join([
                '<limit',
                f'effort="{self.limit["effort"]}"',
                f'velocity="{self.limit["velocity"]}"',
                f'lower="{self.limit["lower"]}"',
                f'upper="{self.limit["upper"]}"',
                '/>,',
            ])
        axis_element = ''
        if self.axis:
            xyz = f'xyz="{" ".join(str(value) for value in self.axis["xyz"])}"'
            axis_element = f'<axis {xyz}/>'
        self.element = '\n'.join([
            f'<joint name="{self.name}" type="{self.type_}">',
            origin_element,
            parent_link_element,
            child_link_element,
            dynamics_element if dynamic_element else '',
            limit_element,
            axis_element,
            '</joint>',
        ])
        return self.element


class Robot:

    def __init__(self, height, mass):
        self.height = height
        self.mass = mass
        self.name = 'George'
        self.create_model()
        self.scale = [height / 100] * 3
        self.create_model()
        self.create_urdf_file()

    def create_model(self):
        current_directory = os.getcwd()
        os.chdir(os.path.expanduser('data/objects/Winter'))
        xml_tree = ET.parse('mmm_modified.urdf')
        xml_root = xml_tree.getroot()
        self.links = xml_root.findall('link')
        self.joints = xml_root.findall('joint')
        self.get_links()
        self.get_joints()
        os.chdir(current_directory)

    def get_joints(self):
        self.joint_elements = []
        for joint in self.joints:
            origin = {}
            if joint.findall('origin'):
                origin = {
                    'xyz': list(
                        map(
                            lambda x: float(x),
                            joint.find(
                                'origin'
                            ).get(
                                'xyz'
                            ).split(
                                ' '
                            )
                        )
                    ),
                    'rpy': list(
                        map(
                            lambda x: float(x),
                            joint.find(
                                'origin'
                            ).get(
                                'rpy'
                            ).split(
                                ' '
                            )
                        )
                    ),
                }
            axis = {}
            if joint.findall('axis'):
                axis = {
                    'xyz': list(
                        map(
                            lambda x: int(x),
                            joint.find('axis').get('xyz').split(' '),
                        )
                    ),
                }
            limit = {}
            if joint.findall('limit'):
                limit['effort'] = joint.find('limit').get('effort')
                limit['velocity'] = joint.find('limit').get('velocity')
                limit['lower'] = joint.find('limit').get('lower')
                limit['upper'] = joint.find('limit').get('upper')
            joint = Joint(
                joint.get('name'),
                joint.get('type'),
                joint.find('parent').get('link'),
                joint.find('child').get('link'),
                axis,
                limit,
                origin,
            )
            self.joint_elements.append(joint)

    def get_links(self):
        sound_meshes = 0
        faulty_meshes = 0
        self.link_elements = []
        # show_polyset = input('Show polyset?(y/n)')
        # show_polyset = show_polyset == 'y'
        show_polyset = False

        for link in self.links:

            inertial = None
            visual = None
            collision = None
            if link.find('inertial'):
                mass = float(
                    link.find('inertial').find('mass').get('value')
                ) * self.mass
                origin = {
                    'xyz': list(
                        map(
                            lambda x: float(x),
                            link.find(
                                'inertial'
                            ).find(
                                'origin'
                            ).get(
                                'xyz'
                            ).split(
                                ' '
                            )
                        )
                    ),
                    'rpy': list(
                        map(
                            lambda x: float(x),
                            link.find(
                                'inertial'
                            ).find(
                                'origin'
                            ).get(
                                'rpy'
                            ).split(
                                ' '
                            )
                        )
                    ),
                }
                if link.find('visual'):
                    sound_meshes += 1
                    mesh_file = link.find(
                        'visual'
                    ).find(
                        'geometry'
                    ).find(
                        'mesh'
                    ).get(
                        'filename'
                    )
                    mesh = pymeshlab.MeshSet()
                    mesh.load_new_mesh(mesh_file)
                    print(
                       f'---Processing the mesh of the link {link.get("name")}'
                    )
                    geomatric_measures = mesh.get_geometric_measures()
                    try:
                        inertia = {
                            'ixx': geomatric_measures['inertia_tensor'][0][0],
                            'ixy': geomatric_measures['inertia_tensor'][0][1],
                            'iyy': geomatric_measures['inertia_tensor'][1][1],
                            'ixz': geomatric_measures['inertia_tensor'][0][2],
                            'iyz': geomatric_measures['inertia_tensor'][1][2],
                            'izz': geomatric_measures['inertia_tensor'][2][2],
                        }
                        print(
                            f'The inertia_tensor is ='
                            f'{geomatric_measures["inertia_tensor"]}'
                        )
                        inertial = Inertial(mass, origin, inertia)
                    except KeyError:
                        faulty_meshes += 1
                        print(
                            '-----inertial_tensor cannot be calucualted for '
                            'this mesh'
                        )

            if link.find('visual'):
                sound_meshes += 1
                print(
                   f'---Processing the mesh of the link {link.get("name")}'
                )
                mesh_file = link.find(
                    'visual'
                ).find(
                    'geometry'
                ).find(
                    'mesh'
                ).get(
                    'filename'
                )
                mesh = pymeshlab.MeshSet()
                mesh.load_new_mesh(mesh_file)
                if show_polyset:
                    mesh.show_polyscope()
                geomatric_measures = mesh.get_geometric_measures()
                try:
                    print(
                        f'The inertia_tensor is ='
                        f'{geomatric_measures["inertia_tensor"]}'
                    )
                except KeyError:
                    faulty_meshes += 1
                    print(
                        '-----inertial_tensor cannot be calucualted for'
                        ' this mesh'
                    )
                geometry = Geometry(mesh_file, scale=[self.height] * 3)
                visual = Visual(geometry)
                collision = Collision(geometry)
            else:
                print(
                    f'No visual attributes for the link {link.get("name")}'
                )

            self.link_elements.append(
                Link(
                    link.get('name'),
                    inertial,
                    visual,
                    collision,
                )
            )

        print(f'{faulty_meshes}')
        print(f'{sound_meshes}')
        print(
            f'-------The percentage of faulty meshes is '
            f'{faulty_meshes/sound_meshes*100}%.'
        )

    def create_urdf_file(self):
        urdf_file = '\n'.join([
            '<?xml version="1.0" encoding="utf-8"?>',
            f'<robot name="{self.name}">',
            '\n'.join(
                link.get_urdf_element() for link in self.link_elements
            ),
            '\n'.join(
                joint.get_urdf_element() for joint in self.joint_elements
            ),
            '</robot>',
        ])
        with open('data/objects/Winter/temporary.urdf', 'w') as xml_file:
            xml_file.write(urdf_file)


if __name__ == '__main__':
    robot = Robot(180, 70)
    robot.create_model()
    robot.create_urdf_file()
