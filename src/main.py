# from dataset import MotionDataset, Motion
import pybullet as pb
import time
import os

if __name__ == '__main__':
    physicsClient = pb.connect(pb.GUI)
    pb.setGravity(0, 0, -10)
    print(os.getcwd())
    planeId = pb.loadURDF(
        os.path.expanduser(
            os.path.join(
                'data/objects',
                'plane.urdf',
            )
        )
    )
    startPos = [0, 0, 1]
    startOrientation = pb.getQuaternionFromEuler([0, 0, 0])
    boxId = pb.loadURDF(
        os.path.expanduser(
            os.path.join(
                'data/objects',
                'robot.urdf',
            ),
        ),
        startPos,
        startOrientation
    )
    for i in range(10000):
        pb.stepSimulation()
        time.sleep(1./2240.)
    cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
    print(cubePos, cubeOrn)
    pb.disconnect()
