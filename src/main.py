from dataset import MotionDataset, Motion
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
    for i in range(44):
        print(pb.getJointInfo(1, i))
        motion = motion_dataset.motions[next_motion]
        motion.parse()
        print(motion.motions[0][0][i])
    while True:
        motion = motion_dataset.motions[next_motion]
        motion.parse()
        for _ in range(1):
            for positions in motion.motions[0][1]:
                for joinId in range(43):
                    pb.setJointMotorControl2(
                        boxId,
                        joinId,
                        2,
                        targetPosition=positions[joinId],
                    )
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
    # print(cubePos, cubeOrn)
    pb.disconnect()
