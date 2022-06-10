import xml.etree.cElementTree as ET
import pymeshlab
import os


class Geemotry:

    def __init__(self, type_, size=None):
        self.type_ = type_
        self.size = size

    def get_urdf_element(self):
        pass


class Inertial:

    def __init__(self, mass, origin, inertia):
        self.mass = mass
        self.origin = origin
        self.inertial = inertia

    def get_urdf_element(self):
        mass_element = f'<mass value="{self.mass}"/>'
        xyz = f'xyz="{" ".join(str(value) for value in self.origin["xyz"])}"'
        rpy = f'rpy="{" ".join(str(value) for value in self.origin["rpy"])}"'
        origin_element = f'<origin {xyz} {rpy}/>'
        inertia_element = '<inertia'
        for key, value in self.inertia.items():
            inertia_element += f'{key}="{value}"'
        inertia_element += '/>'
        self.element = '\n'.join(
            '<intertial>',
            mass_element,
            origin_element,
            inertia_element,
            '</inertial>',
        )
        return self.element


class Visual:

    def __init__(self, origin, geometry, material):
        self.origin = origin
        self.geometry = geometry
        self.material = material

    def get_urdf_element(self):
        xyz = f'xyz="{" ".join(str(value) for value in self.origin["xyz"])}"'
        rpy = f'rpy="{" ".join(str(value) for value in self.origin["rpy"])}"'
        origin_element = f'<origin {xyz} {rpy}/>'
        geometry_element = '\n'.join(
            '<geometry>',
            self.geometry.get_urdf_represetation(),
            '</geometry>',
        )
        material_element = '\n'.join(
            '<material name="{self.material["name"]}">',
            '<color rgba="{self.material.getColorValues()}"/>',
            '</material>',
        )
        self.element = '\n'.join(
            '<visual>',
            origin_element,
            geometry_element,
            material_element,
            '</visual>',
        )
        return self.element


class Collision:

    def __init__(self, origin, geometry):
        self.origin = origin
        self.geometry = geometry

    def get_urfd_element(self):
        xyz = f'xyz="{" ".join(str(value) for value in self.origin["xyz"])}"'
        rpy = f'rpy="{" ".join(str(value) for value in self.origin["rpy"])}"'
        origin_element = f'<origin {xyz} {rpy}'
        geometry_element = '\n'.join(
            '<geometry>',
            self.geometry.get_urdf_represetation(),
            '</geometry>',
        )
        self.element = '\n'.join(
            '<collision>',
            origin_element,
            geometry_element,
            '</collision>'
        )
        return self.element


class Link:

    def __init__(self, name, inertial, visual, collision):
        self.name = name
        self.visual = visual
        self.collision = collision
        self.inertial = inertial

    def get_urdf_element(self):
        self.element = '\n'.join(
            '<link name="{self.name}"',
            self.inertial.get_urdf_element(),
            self.visual.get_urdf_element(),
            self.collsion.get_urdf_element(),
            '</link>',
        )
        return self.element


class Joint:

    def __init__(
        self,
        name,
        type_,
        parent_link,
        child_link,
        dynamics,
        limit,
        axis,
    ):
        self.name = name
        self.type_ = type_
        self.parent_link = parent_link
        self.child_link = child_link
        self.dynamics = dynamics
        self.limit = limit
        self.axis = axis

    def get_urdf_element(self):
        xyz = f'xyz="{" ".join(str(value) for value in self.origin["xyz"])}"'
        rpy = f'rpy="{" ".join(str(value) for value in self.origin["rpy"])}"'
        origin_element = f'<origin {xyz} {rpy}/>'
        parent_link_element = f'<parent link="{self.parent_link.name}"/>'
        child_link_element = f'<child link="{self.child_link.name}"/>'
        dynamics_element = ' '.join(
            '<dynamics',
            f'damping="{self.dynamics["damping"]}',
            f'friction="{self.dynamics["friction"]}',
            '/>',
        )
        limit_element = ' '.join(
            '<limit',
            f'effort="{self.limit["effort"]}"',
            f'velocity="{self.limit["velocity"]}"',
            f'lower="{self.limit["lower"]}"',
            f'upper="{self.limit["upper"]}"',
            '/>,',
        )
        xyz = f'xyz="{" ".join(str(value) for value in self.origin["xyz"])}"'
        axis_element = '<axis {xyz}/>'
        self.element = '\n'.join(
            '<joint name="{self.name}" type="{self.type_}">',
            origin_element,
            parent_link_element,
            child_link_element,
            dynamics_element,
            limit_element,
            axis_element,
            '</joint>',
        )
        return self.element


class Robot:

    def __init__(self, height, mass):
        self.height = height
        self.mass = mass
        self.robot_name = 'George'
        self.create_model()
        self.scale = [height / 100] * 3

    def create_model(self):
        current_directory = os.getcwd()
        os.chdir(os.path.expanduser('data/objects/Winter'))
        xml_tree = ET.parse('mmm_modified.urdf')
        xml_root = xml_tree.getroot()
        self.links = xml_root.findall('link')
        self.joints = xml_root.findall('joint')
        sound_meshes = 0
        faulty_meshes = 0
        for link in self.links:
            if link.find('inertial'):
                mass = float(
                    link.find('inertial').find('mass').get('value')
                ) * self.mass
                origin = {
                    'xyz': list(
                        map(
                            lambda x: float(x),
                            link.find('inertial').find('origin').get('xyz').split(' ')
                        )
                    ),
                    'rpy': list(
                        map(
                            lambda x: float(x),
                            link.find('inertial').find('origin').get('rpy').split(' ')
                        )
                    ),
                }
                if link.find('visual'):
                    sound_meshes += 1
                    mesh_file = link.find('visual').find('geometry').find('mesh').get('filename')
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
                            'iyy': geomatric_measures['inertia_tensor'][1][1],
                            'izz': geomatric_measures['inertia_tensor'][2][2],
                        }
                        print(f'The inertia_tensor is = {geomatric_measures["inertia_tensor"]}')
                        inertial = Inertial(mass, origin, inertia)
                    except KeyError:
                        faulty_meshes += 1
                        print(
                            '-----inertial_tensor cannot be calucualted for this mesh'
                        )
                    # continue_ = input(
                    #     'Do you want to check the next mesh?(y/n)'
                    # )
                    # if continue_ is 'y':
                    #     continue
                    # else:
                    #     break
            else:
                if link.find('visual'):
                    sound_meshes += 1
                    print(
                       f'---Processing the mesh of the link {link.get("name")}'
                    )
                    mesh_file = link.find('visual').find('geometry').find('mesh').get('filename')
                    mesh = pymeshlab.MeshSet()
                    mesh.load_new_mesh(mesh_file)
                    geomatric_measures = mesh.get_geometric_measures()
                    try:
                        print(f'The inertia_tensor is = {geomatric_measures["inertia_tensor"]}')
                    except KeyError:
                        faulty_meshes += 1
                        print(
                            '-----inertial_tensor cannot be calucualted for this mesh'
                        )
                else:
                    print(
                        f'No visual attributes for the link {link.get("name")}'
                    )
        print(f'{faulty_meshes}')
        print(f'{sound_meshes}')
        print(
            f'-------The percentage of faulty meshes is '
            f'{faulty_meshes/sound_meshes*100}%.'
        )

        # for link in self.links:
        #     print(
        #         f'The link with the name {link.get("name")} has the following'
        #     )
        #     intertial = link.find('inertial')
        #     if intertial:
        #         print(
        #             f'  - Inertial:\n      - With mass '
        #             f'{intertial.find("mass").get("value")}\n'
        #             f'      - with origin {intertial.find("origin").get("xyz")} and '
        #             f'and {intertial.find("origin").get("rpy")} '
        #             f'      - with inertia'
        #             f'{intertial.find("inertia").get("ixx")}'
        #             f'{intertial.find("inertia").get("ixy")}'
        #             f'{intertial.find("inertia").get("iyy")}'
        #             f'{intertial.find("inertia").get("iyz")}'
        #             f'{intertial.find("inertia").get("ixz")}'
        #             f'{intertial.find("inertia").get("izz")}'
        #         )
        # for joint in self.joints:
        #     print(joint)

        os.chdir(current_directory)

    def get_joints(self):
        pass

    def get_links(self):
        pass

    def create_urdf_file(self):
        pass


if __name__ == '__main__':
    robot = Robot(180, 70)
    robot.create_model()
