from dataset import MotionDataset, DEFAULT_ROOT
import pybullet as pb
import time
import os


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
    startPosition = [0, 0, .54]
    startOrientation = pb.getQuaternionFromEuler([0, 0, 0])
    boxId = pb.loadURDF(
        os.path.expanduser(
            os.path.join(
                'data/objects/Winter',
                'mmm.urdf',
            ),
        ),
        startPosition,
        startOrientation,
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
    root_position = frame.root_position
    # root_rotation = frame.root_rotation
    pb.setJointMotorControlMultiDof(
        boxId,
        0,
        pb.POSITION_CONTROL,
        targetPosition=root_position,
        # force=[1, 1, 1],
    )
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
                velocityGain=velocity,
                positionGain=position,
                # force=1,
            )
        except KeyError:
            pass

        # time.sleep(time_step - motion_time)
        # motion_time = time_step

    pb.stepSimulation()


if __name__ == '__main__':
    motion_dataset = MotionDataset()
    parse_dataset(motion_dataset)

    next_motion = int(input('Choose motion: '))
    physicsClient = pb.connect(pb.GUI)
    pb.setGravity(0, 0, -10)
    planeId, boxId = load_urdf_models()
    print(get_joints_ids())

    while True:
        motion = motion_dataset.motions[next_motion]
        motion.parse()
        motion_time = 0

        for frame in motion.frames:
            play_frame(frame, motion_time)

        continue_break = input('0 to exit or enter next motion: ')
        if continue_break != '0':
            next_motion = int(continue_break)
        else:
            break

    cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
    pb.disconnect()
