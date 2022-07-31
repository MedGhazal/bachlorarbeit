import pybullet as pb
import pybullet_data


def save_world():
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, 10)
    pb.loadURDF("plane.urdf")
    startPos = [0, 0, 1]
    startOrientation = pb.getQuaternionFromEuler([0, 0, 0])
    boxId = pb.loadURDF(
        "data/objects/Winter/mmm.urdf", startPos, startOrientation)
    cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
    pb.saveWorld('world.py')
    pb.saveBullet('world.bullet')
    pb.disconnect()


if __name__ == '__main__':
    save_world()
