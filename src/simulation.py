import time
from dataset import Motion
import pybullet as pb
import pybullet_data

physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, 10)
planeId = pb.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = pb.getQuaternionFromEuler([0, 0, 0])
boxId = pb.loadURDF("data/objects/robot.urdf", startPos, startOrientation)
for i in range(10000):
    pb.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
pb.disconnect()
