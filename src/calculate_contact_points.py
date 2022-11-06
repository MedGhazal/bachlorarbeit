from dataset import MotionDataset
import pybullet as pb
# import time
# import os
# import numpy as np
from robot import Robot
from main import (
    parse_dataset,
    load_urdf_models,
    play_frame,
)
from utils import change_to
from dataset import DEFAULT_ROOT
# import xml.etree.cElementTree as ET


class ContactPoints:

    def __init__(self, id_, contact_positions, contact_normals, normal_forces):
        self.id_ = id_
        self.contact_positions = contact_positions
        self.contact_normals = contact_normals
        self.normal_forces = normal_forces

    @change_to(DEFAULT_ROOT)
    def save(self):
        with open(f'{self.id_}_contact_points.xml', 'w') as xml_file:
            xml_file.write('<?xml version=\'1.0\'?>\n')
            xml_file.write('<contact_points>\n')
            xml_file.write('<contact_positions>\n')
            for contact_position in self.contact_positions:
                if contact_position:
                    xml_file.write(
                        '\n'.join(
                            [
                                '<contact_position>',
                                ''.join(
                                    str(item) for item in contact_position
                                ),
                                '</contact_position>',
                            ]
                        )
                    )
                else:
                    xml_file.write(
                        '\n'.join(
                            [
                                '<contact_position>',
                                'None',
                                '</contact_position>',
                            ]
                        )
                    )

            xml_file.write('</contact_positions>\n')
            xml_file.write('<contact_normals>\n')
            for contact_normal in self.contact_normals:
                if contact_normal:
                    xml_file.write(
                        '\n'.join(
                            [
                                '<contact_normal>',
                                ''.join(str(item) for item in contact_normal),
                                '</contact_normal>',
                            ]
                        )
                    )
                else:
                    xml_file.write(
                        '\n'.join(
                            [
                                '<contact_normal>',
                                'None',
                                '</contact_normal>',
                            ]
                        )
                    )
            xml_file.write('</contact_normals>\n')
            xml_file.write('<normal_forces>\n')
            for normal_force in self.normal_forces:
                if normal_force:
                    xml_file.write(
                        '\n'.join(
                            [
                                '<normal_force>',
                                str(normal_force),
                                '</normal_force>',
                            ]
                        )
                    )
                else:
                    xml_file.write(
                        '\n'.join(
                            [
                                '<normal_force>',
                                'None',
                                '</normal_force>',
                            ]
                        )
                    )
            xml_file.write('</normal_forces>\n')
            xml_file.write('</contact_points>\n')


def save_contact_points(motion, planeId, boxId, motion_time):
    contact_points_infos = []

    for frame in motion.frames:
        contact_points_info = play_frame(frame, planeId, boxId, motion_time)
        if contact_points_info:
            if len(contact_points_info) >= 1:
                print(
                    f'\nIn timestep we have {len(contact_points_info)} points'
                )
            print(
                f'contactFlag: {contact_points_info[0][0]}',
                f'bodyUniqueIdA: {contact_points_info[0][1]}',
                f'bodyUniqueIdB: {contact_points_info[0][2]}',
                f'linkIndexA: {contact_points_info[0][3]}',
                f'linkIndexB: {contact_points_info[0][4]}',
                f'positionOnA: {contact_points_info[0][5]}',
                f'positionOnb: {contact_points_info[0][6]}',
                f'contactNormalOnB: {contact_points_info[0][7]}',
                f'contactDistance: {contact_points_info[0][8]}',
                f'normalForce: {contact_points_info[0][9]}',
                sep='\n', end='',
            )
            contact_points_infos.append(
                    [
                        contact_points_info[0][6],
                        contact_points_info[0][7],
                        contact_points_info[0][9],
                    ]
            )
        else:
            contact_points_infos.append([None] * 3)

    to_xml_contact_points = ContactPoints(
        motion.id_,
        [item[0] for item in contact_points_infos],
        [item[1] for item in contact_points_infos],
        [item[2] for item in contact_points_infos],
    )
    to_xml_contact_points.save()


def save_contact_points_all(motions):

    for motion in motions:
        pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -9.81)
        motion.parse()
        motion.robot = Robot(motion.mass)
        planeId, boxId = load_urdf_models(motion)
        motion_time = 0
        save_contact_points(motion, planeId, boxId, motion_time)
        pb.disconnect()
        break


if __name__ == '__main__':
    motion_dataset = MotionDataset()
    parse_dataset(motion_dataset)
    save_contact_points_all(motion_dataset.motions)
