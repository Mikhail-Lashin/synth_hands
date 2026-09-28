import os
import numpy as np
from pxr import UsdGeom, UsdShade, Vt, Gf, Sdf
import torch
import smplx
import omni.replicator.core as rep

from config import TEXTURE_PATH, NPZ_PATH, MANO_DIR

class Hand:
    def __init__(self):
        self.verts: np.ndarray | None = None
        self.uv_coords: np.ndarray | None = None
        self.faces: np.ndarray | None = None
        self.face_uv_indices: np.ndarray | None = None
        self.material: UsdShade.Material | None = None
        self.mesh: UsdGeom.Mesh | None = None
    
    def load_obj(self, obj_path: str):
        """Load vertices, faces and UV texture coords from .obj"""
        verts = []
        uv_coords = []
        face_vertex_indices = []
        face_uv_indices = []
        
        with open(obj_path, "r") as f:
            for line in f:
                if line.startswith("v "): # "v x y z"
                    parts = line.split()[1:]
                    parts = [float(p) for p in parts]
                    verts.append(parts)
                elif line.startswith("vt "): # "vt u v"
                    parts = line.split()[1:]
                    parts = [float(part) for part in parts]
                    uv_coords.append(parts)
                elif line.startswith("f "): # "f i_v1/i_vt1/i_vn1 i_v2/i_vt2/i_vn2 i_v3/i_vt3/i_vn3"
                    parts = line.split()[1:]
                    f_v = [int(p.split("/")[0]) - 1 for p in parts]
                    f_vt = [int(p.split("/")[1]) - 1 for p in parts]
                    
                    face_vertex_indices.extend(f_v)
                    face_uv_indices.extend(f_vt)
                    
        verts = np.array(verts, dtype=np.float32)
        
        self.verts = verts
        self.uv_coords = np.array(uv_coords, dtype=np.float32)
        self.faces = np.array(face_vertex_indices, dtype=np.int32).reshape(-1, 3)
        self.face_uv_indices = np.array(face_uv_indices, dtype=np.int32)
        
    def _create_skin_material(self, stage, texture_path: str, path: str = "/World/Looks/SkinMaterial"):
        """Create OmniPBR material for skin"""
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    
        # bind diffuse color texture
        tex_abs_path = os.path.abspath(texture_path)
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_abs_path))
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    
        # natural skin reflection parameters
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.65)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.25)
    
        # connect shader outputs
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")
    
        self.material = material
    
    def create_usd_mesh(self, stage, prim_path: str = "/World/ManoHand"):
        """Create USD mesh"""
        mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
        
        # geometry
        mesh_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(self.verts))
        face_counts = np.full(len(self.faces), 3, dtype=np.int32) # all faces are triangles
        mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(face_counts))
        mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(self.faces.ravel()))
        mesh_prim.CreateExtentAttr(mesh_prim.ComputeExtent(mesh_prim.GetPointsAttr().Get()))
    
        # uv coordinates
        tex_primvar = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        tex_primvar.Set(Vt.Vec2fArray.FromNumpy(self.uv_coords))
        tex_primvar.SetIndices(Vt.IntArray.FromNumpy(self.face_uv_indices))
    
        # Catmull-Clark subdivision (generates smooth limit normals without breaking UV seams)
        mesh_prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
        mesh_prim.CreateInterpolateBoundaryAttr(UsdGeom.Tokens.edgeAndCorner)
        mesh_prim.CreateDoubleSidedAttr(True)
    
        # bind PBR material
        self._create_skin_material(stage, TEXTURE_PATH)
        UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim()).Bind(self.material)
        
        self.mesh = mesh_prim
        
    def upd_verts(self, npz_path=NPZ_PATH, clip_idx=0, frame_idx=0, hand_type="right"):
        data = np.load(npz_path, allow_pickle=True)
        
        clip = data[str(clip_idx)].item()
        pose_frame = torch.tensor(clip[hand_type + "_pose"][frame_idx : frame_idx + 1], dtype=torch.float32)   # (1, 48)
        shape_frame = torch.tensor(clip[hand_type + "_shape"][frame_idx : frame_idx + 1], dtype=torch.float32) # (1, 10)
        
        model_type = "MANO_RIGHT.pkl" if hand_type == "right" else "MANO_LEFT.pkl"
        mano_layer = smplx.create(
            model_path=str(MANO_DIR),
            model_type=model_type,
            is_rhand=True if hand_type == "right" else False,
            use_pca=False,
            flat_hand_mean=False
        )

        output = mano_layer(
            hand_pose=pose_frame[:, 3:],
            betas=shape_frame,
        )

        new_verts = output.vertices[0].detach().cpu().numpy()
        
        # update attr
        self.verts = new_verts
        
        # update mesh
        if self.mesh is not None:
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.verts))
            self.mesh.GetExtentAttr().Set(self.mesh.ComputeExtent(self.mesh.GetPointsAttr().Get()))
        
        return new_verts
    
    def rot_hand(self, R: np.ndarray | None = None) -> np.ndarray:
        if self.verts is None:
            raise ValueError("Verts arent initialised yet. Use load_obj() or upd_verts().")

        if R is None:
            R = np.array([
                [ 0.0, 0.0, 1.0],
                [ 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0]
            ], dtype=np.float32)

        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5), "Rotation matrix must have det = +1"

        # update attr
        self.verts = np.ascontiguousarray(self.verts @ R.T, dtype=np.float32)

        # upd mesh
        if self.mesh is not None:
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.verts))
            self.mesh.GetExtentAttr().Set(self.mesh.ComputeExtent(self.mesh.GetPointsAttr().Get()))

        return self.verts
    
class StereoCamera:
    """
    Default - Intel RealSense D405:
        ~ baseline - 18 mm
        ~ color camera fov - H:84 / V:58 / D:92 (degrees)
        ~ sensor: OV9782
    """
    def __init__(
        self,
        position: tuple = (0.0, -0.28, 0.0),
        look_at: tuple = (0.0, 0.0, 0.0),
        resolution: tuple = (848, 480),
        baseline: float = 0.018,
        focal_length: float | None = None,
        h_fov: float = 84.0,
        v_fov: float = 58.0,
        sensor_width: float = 3.896,           # default for OV9782 (mm)
        sensor_height: float = 2.453,          # default for OV9782 (mm)
        clip_range: tuple = (0.05, 2.0)
    ):
        self.resolution = resolution
        self.baseline = baseline
        self.position = np.array(position, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.clip_range = clip_range
        
        w, h = resolution
        ratio_resolution = h / w
        ratio_sensor = sensor_height / sensor_width
        self.horizontal_aperture = sensor_width if ratio_resolution <= ratio_sensor else sensor_height / ratio_resolution
        self.vertical_aperture = sensor_height if ratio_resolution >= ratio_sensor else sensor_width * ratio_resolution
        
        if focal_length is not None:
            self.focal_length = focal_length
        else:
            self.focal_length = self.horizontal_aperture / (2.0 * np.tan(np.radians(h_fov / 2.0)))
        
        self.cam_left = None
        self.cam_right = None
        self.rp_left = None
        self.rp_right = None
        self.annot_rgb_left = None
        self.annot_rgb_right = None
        self.annot_depth_left = None
        
        self._init_cameras()
        self.setup_pipeline()

    def _compute_stereo_vectors(self, pos, target):
        """Calculate orthonormal basis for left and right imagers offsets"""
        forward = target - pos
        forward /= np.linalg.norm(forward)
        
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if np.abs(np.dot(forward, up_world)) > 0.99:
            up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right)
        
        return forward, right

    def _init_cameras(self):
        _, right = self._compute_stereo_vectors(self.position, self.look_at)
        
        # imagers offsets
        half_b = (self.baseline / 2.0) * right
        pos_left = tuple(self.position - half_b)
        pos_right = tuple(self.position + half_b)
        look_left = tuple(self.look_at - half_b)
        look_right = tuple(self.look_at + half_b)

        self.cam_left = rep.create.camera(
            position=pos_left,
            look_at=look_left,
            focal_length=self.focal_length,
            horizontal_aperture=self.horizontal_aperture,
            vertical_aperture=self.vertical_aperture,
            clipping_range=self.clip_range,
            name="LeftCam"
        )
        
        self.cam_right = rep.create.camera(
            position=pos_right,
            look_at=look_right,
            focal_length=self.focal_length,
            horizontal_aperture=self.horizontal_aperture,
            vertical_aperture=self.vertical_aperture,
            clipping_range=self.clip_range,
            name="RightCam"
        )

    def setup_pipeline(self):
        """Init render buffers & annotators (RGB + Depth)"""
        # render products
        self.rp_left = rep.create.render_product(self.cam_left, self.resolution)
        self.rp_right = rep.create.render_product(self.cam_right, self.resolution)

        # rgb annotators
        self.annot_rgb_left = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annot_rgb_left.attach([self.rp_left])

        self.annot_rgb_right = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annot_rgb_right.attach([self.rp_right])

        # depth (only for left cam)
        self.annot_depth_left = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self.annot_depth_left.attach([self.rp_left])

    def render(self, num_subframes: int = 16) -> dict:
        """Render RGB & Depth"""
        rep.orchestrator.step(rt_subframes=num_subframes, pause_timeline=True)
        
        left_rgb = self.annot_rgb_left.get_data()[:, :, :3]
        right_rgb = self.annot_rgb_right.get_data()[:, :, :3]
        left_depth = self.annot_depth_left.get_data()
        
        return {
            "left_rgb": left_rgb,
            "right_rgb": right_rgb,
            "left_depth": left_depth
        }

    def get_intrinsics(self) -> np.ndarray:
        w, h = self.resolution
        fx = (w * self.focal_length) / self.horizontal_aperture
        fy = (h * self.focal_length) / self.vertical_aperture
        cx = w / 2.0
        cy = h / 2.0
        
        return np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)

class SceneManager:
    pass