from dataset import MotionDataset
import pybullet as pb
# import time
import os
import numpy as np
from robot import Robot


def parse_dataset(motion_dataset):

    try:
        motion_dataset.parse()
    except FileNotFoundError:
        motion_dataset.extract()
        motion_dataset.parse()


def load_urdf_models(motion):
    planeId = pb.loadURDF(
        os.path.expanduser(
            os.path.join(
                'data/objects',
                'plane.urdf',
            )
        )
    )
    scale = motion.scale_factor
    initial_root_position = np.array(motion.get_initial_root_position()) / 1000
    startOrientation = pb.getQuaternionFromEuler(
        [0, 0, 0]
    )
    boxId = pb.loadURDF(
        os.path.expanduser(
            os.path.join(
                'data/objects/Winter',
                'temporary.urdf',
                # 'mmm.urdf',
            ),
        ),
        list(initial_root_position),
        startOrientation,
        useFixedBase=1,
        flags=(
            pb.URDF_MERGE_FIXED_LINKS
        ),
        globalScaling=scale,
    )
    return planeId, boxId


def get_joints_ids(model):
    joint_ids = {
        joint.decode(): id_
        for id_, joint in [
            pb.getJointInfo(1, i)[:2] for i
            in range(66)
        ]
    }
    return joint_ids


def play_frame(frame, planeId, boxId, motion_time):
    root_position = np.array(frame.root_position) / 1000
    pb.setJointMotorControlMultiDof(
        boxId,
        0,
        pb.POSITION_CONTROL,
        targetPosition=list(root_position),
    )
    # root_rotation = frame.root_rotation
    # pb.setJointMotorControlMultiDof(
    #     boxId,
    #     0,
    #     pb.POSITION_CONTROL,
    #     targetPosition=root_rotation,
    # )
    joint_positions = frame.joint_positions
    joint_velocities = frame.joint_velocities
    # joint_accelerations = frame.joint_accelerations
    # time_step = frame.timestep

    for jointId in range(pb.getNumJoints(boxId)):

        try:
            position = joint_positions[
                pb.getJointInfo(boxId, jointId)[1].decode()
            ]
            velocity = joint_velocities[
                pb.getJointInfo(boxId, jointId)[1].decode()
            ]
            # acceleration = joint_accelerations[
            #     pb.getJointInfo(boxId, jointId)[1].decode()
            # ]
            pb.setJointMotorControl2(
                boxId,
                jointId,
                pb.POSITION_CONTROL,
                targetVelocity=velocity,
                targetPosition=position,
            )
        except KeyError:
            pass

    contact_points_infos = pb.getContactPoints(planeId, boxId)

    # time.sleep((time_step - motion_time) / 1000)
    # motion_time = time_step

    pb.stepSimulation()

    return contact_points_infos


def extract_normal_force(contact_points_info):
    return contact_points_info[-1] if contact_points_info else 0


if __name__ == '__main__':
    motion_dataset = MotionDataset()
    parse_dataset(motion_dataset)

    next_motion = int(input('Choose motion: '))
    physicsClient = pb.connect(pb.GUI)
    pb.setGravity(0, 0, -9.81)

    motion = motion_dataset.motions[next_motion]
    motion.parse()
    motion.robot = Robot(motion.mass)
    planeId, boxId = load_urdf_models(motion)
    motion_time = 0
    contact_points_infos = []
    print(motion.annotation)

    for frame in motion.frames:
        play_frame(frame, planeId, boxId, motion_time)

    normal_forces = list(
        map(extract_normal_force, contact_points_infos)
    )
    print(*normal_forces, sep='\n')

    cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
    pb.disconnect()
