# Transformations that can be applied to ModelNet40 poinclouds.
# Code based on: https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-ii-pytorch

import random
import math

import numpy as np
import torch

random.seed = 42


class PointSampler(object):
    def __init__(self, output_size):
        """Sample points randomly from a given pointcloud."""
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = self.triangle_area(
                verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]]
            )

        sampled_faces = random.choices(
            faces, weights=areas, cum_weights=None, k=self.output_size
        )

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(
                verts[sampled_faces[i][0]],
                verts[sampled_faces[i][1]],
                verts[sampled_faces[i][2]],
            )

        return sampled_points


class Normalise(object):
    def __call__(self, pointcloud):
        """Center and minmax normalise the pointcloud vectors."""
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


class RandRotationZ(object):
    def __call__(self, pointcloud):
        """Apply a random rotation around the Z-axis."""
        theta = random.random() * 2.0 * math.pi
        rot_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud
    
    
# The pointcloud is expected to be a numpy array where each row corresponds to a point.
# I simply added a random shuffle function for randomly shuffeling the data points. 
# The seed you seem to specify in the launch file. It seems you do not use the transformation yet. 

class RandPermutation(object):
    def __call__(self, pointcloud):
        """Randomly permutes the points in the pointcloud."""
        
        perm_pointcloud = np.random.shuffle(pointcloud)
        return perm_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        """Apply noise to the positions of the pointcloud."""
        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        """Convert numpy array to pytorch tensor."""
        return torch.from_numpy(pointcloud)
