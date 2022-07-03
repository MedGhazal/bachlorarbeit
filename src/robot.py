import xml.etree.cElementTree as ET
import pymeshlab
import os


class FaultyMesh(Exception):
    pass


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
                        [0, 1, 1],
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

    def __init__(self, geometry, origin):
        self.geometry = geometry
        self.origin = origin

    def get_urdf_element(self):
        geometry_element = self.geometry.get_urdf_element()
        xyz = f'xyz="{" ".join(str(value) for value in self.origin["xyz"])}"'
        rpy = f'rpy="{" ".join(str(value) for value in self.origin["rpy"])}"'
        origin_element = f'<origin {xyz} {rpy}/>'
        self.element = '\n'.join([
            '<visual>',
            origin_element,
            geometry_element,
            '</visual>',
        ])
        return self.element


class Collision:

    def __init__(self, geometry, origin):
        self.geometry = geometry
        self.origin = origin

    def get_urdf_element(self):
        geometry_element = self.geometry.get_urdf_element()
        xyz = f'xyz="{" ".join(str(value) for value in self.origin["xyz"])}"'
        rpy = f'rpy="{" ".join(str(value) for value in self.origin["rpy"])}"'
        origin_element = f'<origin {xyz} {rpy}/>'
        self.element = '\n'.join([
            '<collision>',
            origin_element,
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
            xyz = f'xyz="{" ".join(str(item) for item in self.origin["xyz"])}"'
            rpy = f'rpy="{" ".join(str(item) for item in self.origin["rpy"])}"'
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
        self.links = []
        self.scale = [height / 100] * 3
        self.create_model()
        self.create_urdf_file()

    def create_model(self):
        current_directory = os.getcwd()
        os.chdir(os.path.expanduser('data/objects/Winter'))
        xml_tree = ET.parse('mmm_new.urdf')
        xml_root = xml_tree.getroot()
        self.links = xml_root.findall('link')
        self.joints = xml_root.findall('joint')
        self.get_links()
        self.get_joints()
        os.chdir(current_directory)

    def get_origin(self, origin):
        return {
            'xyz': list(
                map(
                    lambda x: float(x),
                    origin.get('xyz').split(' ')
                )
            ),
            'rpy': list(
                map(
                    lambda x: float(x),
                    origin.get('rpy').split(' ')
                )
            ),
        }

    def get_axis(self, axis):
        return {
            'xyz': list(map(lambda x: int(x), axis.get('xyz').split(' ')))
        }

    def get_limit(self, limit):
        return {
            'effort': limit.get('effort'),
            'velocity': limit.get('velocity'),
            'lower': limit.get('lower'),
            'upper': limit.get('upper'),
        }

    def get_joints(self):
        self.joint_elements = []
        for joint in self.joints:
            origin = {}
            if joint.findall('origin'):
                origin = self.get_origin(joint.find('origin'))
            axis = {}
            if joint.findall('axis'):
                axis = self.get_axis(joint.find('axis'))
            limit = {}
            if joint.findall('limit'):
                limit = self.get_limit(joint.find('limit'))
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

    def calculate_inertia(self, mesh_file):
        mesh_set = pymeshlab.MeshSet()
        mesh_set.load_new_mesh(mesh_file)
        geometric_measures = mesh_set.get_geometric_measures()
        # mesh_set.show_polyscope()
        try:
            return {
                'ixx': geometric_measures['inertia_tensor'][0][0],
                'ixy': geometric_measures['inertia_tensor'][0][1],
                'iyy': geometric_measures['inertia_tensor'][1][1],
                'ixz': geometric_measures['inertia_tensor'][0][2],
                'iyz': geometric_measures['inertia_tensor'][1][2],
                'izz': geometric_measures['inertia_tensor'][2][2],
            }
        except KeyError:
            raise FaultyMesh

    def get_mesh_file(self, visual):
        return visual.find('geometry').find('mesh').get('filename')

    def get_inertial(self, link, inertial):
        mass = float(
            inertial.find('mass').get('value')
        ) * self.mass
        origin = self.get_origin(inertial.find('origin'))
        inertia = inertial.find('inertia').attrib
        faulty_mesh = False
        if link.findall('visual'):
            mesh_file = self.get_mesh_file(link.find('visual'))
            print(
               f'---Processing the mesh of the link {link.get("name")}'
            )
            try:
                inertia = self.calculate_inertia(mesh_file)
            except FaultyMesh:
                faulty_mesh = True
                print(
                    f'----Cannot calculate inetial_tensor for the link '
                    f'{link.get("name")}'
                )
                inertia = inertial.find('inertia').attrib
        inertial = Inertial(mass, origin, inertia)
        return inertial, faulty_mesh

    def get_links(self):
        sound_meshes = 0
        faulty_meshes = 0
        self.link_elements = []
        # show_polyset = input('Show polyset?(y/n)')
        # show_polyset = show_polyset == 'y'
        # show_polyset = False

        for link in self.links:

            inertial = None
            visual = None
            collision = None
            if link.findall('inertial'):
                sound_meshes += 1
                inertial, faulty = self.get_inertial(
                    link,
                    link.find('inertial'),
                )
                faulty_meshes += 1 if faulty else 0

            if link.findall('visual'):
                sound_meshes += 1
                mesh_file = self.get_mesh_file(link.find('visual'))
                if link.find('inertial'):
                    origin = self.get_origin(
                        link.find('inertial').find('origin'))
                geometry = Geometry(
                    mesh_file,
                    scale=[1, 1, self.height],
                )
                visual = Visual(geometry, origin)
                collision = Collision(geometry, origin)
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
