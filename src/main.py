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
    motion = motion_dataset.motions[next_motion]
    motion.parse()
    startPos = [0, 0, .55]
    startOrientation = pb.getQuaternionFromEuler([0, 0, 0])
    boxId = pb.loadURDF(
        os.path.expanduser(
            os.path.join(
                'data/objects/Winter',
                # 'mmm_modified.urdf',
                'temporary.urdf'
            ),
        ),
        startPos,
        startOrientation,
    )
    # print(f'The number of joints is {pb.getNumJoints(boxId)}')
    # print(
        # f'The number of joints in the mmm-files is '
        # f'{len(motion.motions[0][0])}'
    # )
    # print(
        # f'The joints in the mmm-files are'
        # f'{motion.motions[0][0]}'
    # )
    joint_ids = {
        joint.decode(): id_
        for id_, joint in [
            pb.getJointInfo(1, i)[:2] for i
            in range(66)
        ]
    }
    # print(f'The joints and their ids ind the simulation are {joint_ids}')
    # initialPosition = .5
    # for i in range(30):
    #     pb.setJointMotorControl2(
    #         boxId,
    #         12,
    #         joint_ids['jL5S1_rotz'],
    #         targetPosition=initialPosition,
    #     )
    #     pb.stepSimulation()
    #     time.sleep(1/100)
    #     initialPosition += 1
    while True:
        motion = motion_dataset.motions[next_motion]
        motion.parse()
        joints_not_mmm_conform = set()
        positions = {joint: position for joint, position in zip(
            motion.motions[0][0],
            motion.motions[0][1][0],
        )}
        for joinId in range(len(joint_ids)):
            try:
                pb.setJointMotorControl2(
                    boxId,
                    pb.getJointInfo(boxId, joinId)[0],
                    2,
                    targetPosition=positions[
                        pb.getJointInfo(boxId, joinId)[1].decode()
                    ],
                )
            except KeyError:
                joints_not_mmm_conform.add(
                    pb.getJointInfo(boxId, joinId)[1].decode(),
                )
        pb.stepSimulation()
        for _ in range(1):
            for positions in motion.motions[0][1][1:]:
                positions = {joint: position for joint, position in zip(
                    motion.motions[0][0],
                    positions,
                )}
                for jointId in range(pb.getNumJoints(boxId)):
                    # print(
                    #     f'Moving the link: {pb.getJointInfo(boxId, jointId)[1].decode()}'
                    # )
                    try:
                        next_position = positions[
                            pb.getJointInfo(boxId, jointId)[1].decode()
                        ]
                        pb.setJointMotorControl2(
                            boxId,
                            joinId,
                            2,
                            targetPosition=next_position,
                        )
                    except KeyError:
                        continue
                pb.stepSimulation()
            time.sleep(1./10.)
        print(f'The joint that aren\'t conform are {joints_not_mmm_conform}')
        continue_break = input(
            '0 to exit or enter next motion: '
        )
        if continue_break != '0':
            next_motion = int(continue_break)
        else:
            break
    cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
    pb.disconnect()
