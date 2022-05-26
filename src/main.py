from dataset import MotionDataset
import pybullet as pb
import time
import os

if __name__ == '__main__':
    motion_dataset = MotionDataset()
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
    startPos = [0, 0, .9]
    startOrientation = pb.getQuaternionFromEuler([0, 0, 0])
    boxId = pb.loadURDF(
        os.path.expanduser(
            os.path.join(
                'data/objects',
                'robot.urdf',
            ),
        ),
        startPos,
        startOrientation,
    )
    motion = motion_dataset.motions[next_motion]
    motion.parse()
    print(motion.motions[0][0])
    print([pb.getJointInfo(1, i) for i in range(44)])
    # initialPosition = .5
    # print(pb.getJointInfo(boxId, 12))
    # for i in range(10):
    #     pb.setJointMotorControl2(
    #         boxId,
    #         12,
    #         2,
    #         targetPosition=initialPosition,
    #     )
    #     pb.stepSimulation()
    #     time.sleep(1/100)
    #     initialPosition += 1
    while True:
        motion = motion_dataset.motions[next_motion]
        motion.parse()
        for joinId in range(43):
            pb.setJointMotorControl2(
                boxId,
                joinId,
                2,
                targetPosition=motion.motions[0][1][0][joinId],
            )
        for _ in range(1):
            for positions in motion.motions[0][1]:
                positions = {joint: position for joint, position in zip(
                    motion.motions[0][0],
                    positions,
                )}
                for joinId in range(43):
                    try:
                        pb.setJointMotorControl2(
                            boxId,
                            joinId,
                            2,
                            targetPosition=positions[
                                pb.getJointInfo(boxId, joinId)[1].decode()
                            ],
                        )
                    except KeyError:
                        print(f'Joint {pb.getJointInfo(boxId, joinId)[1].decode()} not in mmm File')
                pb.stepSimulation()
            time.sleep(1./10.)
        continue_break = input(
            '0 to exit or enter next motion: '
        )
        if continue_break != '0':
            next_motion = int(continue_break)
        else:
            break
    cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
    pb.disconnect()
