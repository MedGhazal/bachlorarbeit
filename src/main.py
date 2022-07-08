from dataset import MotionDataset, DEFAULT_ROOT
import pybullet as pb
import time
import os
import numpy as np


def parse_dataset(motion_dataset):

    if os.path.exists(DEFAULT_ROOT):
        motion_dataset.parse()
    else:
        motion_dataset.extract()
        motion_dataset.parse()


def load_urdf_models():
    planeId = pb.loadURDF(
        os.path.expanduser(
            os.path.join(
                'data/objects',
                'plane.urdf',
            )
        )
    )
    scale = motion_dataset.motions[next_motion].scale_factor
    # startPosition = [0, 0, .53 * scale]
    initial_root_position = np.array(motion.get_initial_root_position()) / 1000
    startOrientation = pb.getQuaternionFromEuler(
        [0, 0, 0]
    )
    boxId = pb.loadURDF(
        os.path.expanduser(
            os.path.join(
                'data/objects/Winter',
                'temporary.urdf',
            ),
        ),
        # startPosition,
        list(initial_root_position),
        startOrientation,
        # useFixedBase=1,
        flags=(
            pb.URDF_MERGE_FIXED_LINKS
            # | pb.URDF_USE_INERTIA_FROM_FILE
        ),
        globalScaling=scale,
    )
    return planeId, boxId


def get_joints_ids():
    joint_ids = {
        joint.decode(): id_
        for id_, joint in [
            pb.getJointInfo(1, i)[:2] for i
            in range(66)
        ]
    }
    return joint_ids


def play_frame(frame, motion_time):
    root_position = np.array(frame.root_position)
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
    time_step = frame.timestep

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
                # targetVelocity=velocity,
                targetPosition=position,
            )
        except KeyError:
            pass

    contact_points_infos.append(
        pb.getContactPoints(planeId, boxId,),
    )

    time.sleep((time_step - motion_time) / 1000)
    motion_time = time_step

    pb.stepSimulation()


if __name__ == '__main__':
    motion_dataset = MotionDataset()
    parse_dataset(motion_dataset)

    next_motion = int(input('Choose motion: '))
    physicsClient = pb.connect(pb.GUI)
    pb.setGravity(0, 0, -9.81)

    motion = motion_dataset.motions[next_motion]
    motion.parse()
    planeId, boxId = load_urdf_models()
    motion_time = 0
    contact_points_infos = []

    for frame in motion.frames:
        play_frame(frame, motion_time)

    cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
    pb.disconnect()
