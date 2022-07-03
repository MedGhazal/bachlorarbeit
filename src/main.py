from dataset import MotionDataset, DEFAULT_ROOT
import pybullet as pb
import time
import os

if __name__ == '__main__':
    motion_dataset = MotionDataset()
    if os.path.exists(DEFAULT_ROOT):
        motion_dataset.parse()
    else:
        motion_dataset.extract()
        motion_dataset.parse()
    next_motion = int(input('Choose motion: '))
    physicsClient = pb.connect(pb.GUI)
    pb.setGravity(0, 0, -10)
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
    joint_ids = {
        joint.decode(): id_
        for id_, joint in [
            pb.getJointInfo(1, i)[:2] for i
            in range(66)
        ]
    }

    while True:
        motion = motion_dataset.motions[next_motion]
        motion.parse()
        motion_time = 0

        for frame in motion.frames:
            root_position = frame.root_position
            root_rotation = frame.root_rotation
            joint_positions = frame.joint_positions
            joint_velocities = frame.joint_velocities
            joint_accelerations = frame.joint_accelerations
            time_step = frame.timestep

            for jointId in range(pb.getNumJoints(boxId)):
                try:
                    position = joint_positions[
                        pb.getJointInfo(boxId, jointId)[1].decode()
                    ]
                    velocity = joint_velocities[
                        pb.getJointInfo(boxId, jointId)[1].decode()
                    ]
                    acceleration = joint_accelerations[
                        pb.getJointInfo(boxId, jointId)[1].decode()
                    ]
                    pb.setJointMotorControl2(
                        boxId,
                        jointId,
                        pb.POSITION_CONTROL,
                        targetVelocity=velocity,
                        targetPosition=position/100,
                        force=1,
                    )
                except KeyError:
                    pass

            pb.stepSimulation()
            # time.sleep(time_step - motion_time)
            frame_time = time_step

        continue_break = input(
            '0 to exit or enter next motion: '
        )
        if continue_break != '0':
            next_motion = int(continue_break)
        else:
            break
    cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
    pb.disconnect()
