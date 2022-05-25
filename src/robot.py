class Robot:

    def __init__(self, height, mass, joints):
        self.height = height
        self.mass = mass
        self.joints = joints
        self.robot_name = 'George'
        self.create_model()

    def create_model(self):
        pass
        # with open('model.urdf', 'w') as urdf_file:
        #     urdf_file.write(f'<robot name={self.robot_name}>')
        # for link_name, link_index in zip(
        #     self.joints['names'],
        #     self.joints['indexes'],
        # ):
        #     print(f'The link name is {link_name} with the index {link_index}')
        #     with open('model.urdf', 'a') as urdf_file:
        #         urdf_file.write(f'<link name={self.robot_name}/>')
        # with open('model.urfd', 'a') as urdf_file:
        #     urdf_file.write('</robot>')
        # print(f'<robot name="{self.robot_name}">')
        # print('<link name="head"/>')
        # print('<link name="neck"/>')
        # print('<link name="left_shoulder"/>')
        # print('<link name="right_shoulder"/>')
        # print('<link name="left_arm"/>')
        # print('<link name="right_arm"/>')
        # print('<link name="left_forarm"/>')
        # print('<link name="right_forarm"/>')
        # print('<link name="left_hand"/>')
        # print('<link name="right_hand"/>')
        # print('<link name=""/>')
        # for joint_name in self.joints['names']:
        #     break
        #     print(joint_name)
        # print('<robot/>')
