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

    def __init__(self, name, type_, parent_link, child_link, dynamics, limit, axis):
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

    def __init__(self, height, mass, joints_names):
        self.height = height
        self.mass = mass
        self.joints = joints_names 
        self.robot_name = 'George'
        self.create_model()

    def create_model(self):
        pass

    def get_joints(self):
        pass

    def get_links(self):
        pass

    def create_urdf_file(self):
        pass
