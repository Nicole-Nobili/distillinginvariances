# Loader class for the ModelNet40 princeton data set.
# Available here: https://modelnet.cs.princeton.edu/
# Code based on: https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-ii-pytorch

import os
import numpy as np
import itertools
from pathlib import Path
from dataclasses import dataclass

import torch
import torchvision
import scipy.spatial.distance
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import sklearn

import util
import modelnet_transforms


@dataclass
class PointCloud:
    """3D object represented by pointcloud data."""

    filepath: Path
    verts: list | torch.Tensor
    faces: list

    def get_verts_coords(self):
        x, y, z = np.array(self.verts).T
        return x, y, z

    def get_faces_coords(self):
        i, j, k = np.array(self.faces).T
        return i, j, k

    def visualise_faces(self, outdir: str = "modelnet_images"):
        """Saves 3D figure of the pointcloud with faces."""
        util.make_output_directories("./", outdir)
        x, y, z = self.get_verts_coords()
        i, j, k = self.get_faces_coords()
        data = go.Mesh3d(
            x=x, y=y, z=z, color="yellowgreen", opacity=0.50, i=i, j=j, k=k
        )
        figr = go.Figure(data=data)
        image_name = self.filepath.stem + "_faces.png"
        figr.write_image(os.path.join(outdir, image_name))

    def visualise_verts(self, outdir: str = "modelnet_images"):
        """Saves 3D figure of pointcloud with just the points."""
        util.make_output_directories("./", outdir)
        x, y, z = self.get_verts_coords()
        data = [go.Scatter3d(x=x, y=y, z=z, mode="markers")]
        figr = go.Figure(data=data)
        figr.update_traces(
            marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        image_name = self.filepath.stem + "_verts.png"
        figr.write_image(os.path.join(outdir, image_name))


class Modelnet40DataLoader(object):
    def __init__(
        self,
        pointclouds_rootdir: str,
        validation: bool,
        transforms: torchvision.transforms.Compose
        ):
        """ModelNet40 pointcloud data for object classification.

        Attributes:
            pointclouds_rootdir: The filepath to the pointcloud datafile.
            validation: Whether this is validation data (True) or training (False).
            transforms: Composition of torchvision transformations to apply to data.
        """
        self.rootdir = Path(pointclouds_rootdir)
        self.dirs = [
            directory
            for directory in sorted(os.listdir(self.rootdir))
            if os.path.isdir(os.path.join(self.rootdir, directory))
        ]
        self.classes = {folder: i for i, folder in enumerate(self.dirs)}
        self.validation = validation
        self.files = self._get_files()
        self.transforms = transforms

    def _get_files(self):
        """Gets the modelnet pointcloud files."""
        files = []
        dtype_folder = "test" if self.validation else "train"
        for category in self.classes.keys():
            current_dir = os.path.join(self.rootdir, Path(category), dtype_folder)
            for file in os.listdir(current_dir):
                if file.endswith(".off"):
                    sample = {}
                    sample["pointcloud_path"] = os.path.join(current_dir, file)
                    sample["category"] = category
                    files.append(sample)

        return files

    def __len__(self):
        return len(self.files)

    def __preprocess__(self, file):
        """Reads a pointcloud file and transforms it using the given transformations."""
        pointcloud = self._read_off_file(file)
        if self.transforms:
            pointcloud_verts = self.transforms((pointcloud.verts, pointcloud.faces))
            return PointCloud(pointcloud.filepath, pointcloud_verts, pointcloud.faces)

        return pointcloud

    def __getitem__(self, idx):
        """Gets a transformed pointcloud and the category class it belongs to."""
        pointcloud_path = self.files[idx]["pointcloud_path"]
        category = self.files[idx]["category"]
        with open(pointcloud_path, "r") as pointcloud_file:
            pointcloud_verts = self.__preprocess__(pointcloud_file)

        return {"pointcloud": pointcloud_verts, "category": self.classes[category]}

    def show_pointcloud(self, idx, outdir = "modelnet_images"):
        """Shows the pointcloud with sample number equal to idx."""
        pointcloud_path = self.files[idx]["pointcloud_path"]
        category = self.files[idx]["category"]
        with open(pointcloud_path, "r") as pointcloud_file:
            pointcloud = self.__preprocess__(pointcloud_file)
            pointcloud.visualise_faces(outdir)
            pointcloud.visualise_verts(outdir)

    def _read_off_file(self, file):
        """Reads the given pointcloud file.

        Gives the vertices and faces vectors of the object in the file.
        """
        off_header = file.readline().strip()
        if "OFF" == off_header:
            n_verts, n_faces, __ = tuple(
                [int(s) for s in file.readline().strip().split(" ")]
            )
        else:
            n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(" ")])

        verts = [
            [float(s) for s in file.readline().strip().split(" ")]
            for i_vert in range(n_verts)
        ]
        faces = [
            [int(s) for s in file.readline().strip().split(" ")][1:]
            for i_face in range(n_faces)
        ]

        return PointCloud(Path(file.name), verts, faces)
