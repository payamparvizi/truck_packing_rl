#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:45:57 2026

@author: payam
"""

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import gymnasium as gym
from gymnasium import spaces

class PACKEnv(gym.Env):
    def __init__(
        self,
        render=False,
        num_boxes=120,
        release_interval=5.0,
        grid_res=40,
        settle_steps_max=1000,
        vel_threshold=3e-3,
        seed = 0,
    ):
        
        self.render = render
        self.num_boxes = num_boxes
        self.release_interval = release_interval
        self.grid_res = grid_res
        self.settle_steps_max = settle_steps_max
        self.vel_threshold = vel_threshold
        self.seed = seed

        self.rng = jax.random.PRNGKey(self.seed)
        self.current_box = 0
        self.next_release_time = 0.0

        self.box_specs = self._build_box_specs()
        self.xml = self.build_xml(self.box_specs)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self._cache_model_indices()
        self.viewer = None
        self._build_grid_points()
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.grid_res, self.grid_res),
            dtype=np.float32,
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 2.6, 2.75, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        if self.render:
            plt.figure(figsize=(15, 4))
        


    def _cache_model_indices(self):
        self.box_geom_ids = [self.model.geom(f"box{i}_geom").id for i in range(self.num_boxes)]
        self.box_qpos_adrs = [self.model.jnt_qposadr[i] for i in range(self.num_boxes)]
        self.box_qvel_adrs = [self.model.jnt_dofadr[i] for i in range(self.num_boxes)]

    
    # sampling the box:
    def sample_box(self, rng):
        dim_min = jnp.array([0.115, 0.110, 0.066], dtype=jnp.float32)
        dim_max = jnp.array([0.788, 0.824, 0.482], dtype=jnp.float32)
        weight_min = jnp.float32(0.001)
        weight_max = jnp.float32(23.69)
    
        rng_dims, rng_weight, next_rng = jax.random.split(rng, 3)
    
        dims = jax.random.uniform(
            rng_dims,
            shape=(3,),
            minval=dim_min,
            maxval=dim_max,
            dtype=jnp.float32,
        )
    
        weight = jax.random.uniform(
            rng_weight,
            shape=(),
            minval=weight_min,
            maxval=weight_max,
            dtype=jnp.float32,
        )
    
        dims = jnp.round(dims, 3)
        weight = jnp.round(weight, 2)
    
        return dims, weight, next_rng


    def sample_quaternion(self, rng):
        s = jnp.sqrt(jnp.float32(0.5))
        quats = jnp.array([
            [1.0,   0.0,   0.0,   0.0],   # identity
            [s, 0.0,   0.0,   s], # 90 deg around Z
            [s, 0.0,   s, 0.0],   # 90 deg around Y
            [s, s, 0.0,   0.0],   # 90 deg around X
        ], dtype=jnp.float32)
    
        rng_idx, next_rng = jax.random.split(rng)
        idx = jax.random.randint(rng_idx, shape=(), minval=0, maxval=4)
        quat = quats[idx]
        return quat, next_rng


    def rotated_dims_from_quat(self, dims, quat):
        dims = jnp.asarray(dims, dtype=jnp.float32)
        quat = jnp.asarray(quat, dtype=jnp.float32)
    
        s = jnp.sqrt(jnp.float32(0.5))
    
        quats = jnp.array([
            [1.0, 0.0, 0.0, 0.0],   # identity
            [s,   0.0, 0.0, s],     # Z
            [s,   0.0, s,   0.0],   # Y
            [s,   s,   0.0, 0.0],   # X
        ], dtype=jnp.float32)
    
        idx = jnp.argmin(jnp.sum((quats - quat) ** 2, axis=1))
    
        perms = jnp.stack([
            dims,
            dims[jnp.array([1, 0, 2], dtype=jnp.int32)],  # Z: swap x,y
            dims[jnp.array([2, 1, 0], dtype=jnp.int32)],  # Y: swap x,z
            dims[jnp.array([0, 2, 1], dtype=jnp.int32)],  # X: swap y,z
        ], axis=0)
    
        return perms[idx]


    def sample_position(self, rng, dims):
        container_min = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        container_max = jnp.array([2.0, 2.6, 2.75], dtype=jnp.float32)
    
        margin = jnp.float32(0.01)
    
        half = dims / 2.0
        pos_min = container_min + half + margin
        pos_max = container_max - half - margin
    
        valid = jnp.all(pos_max >= pos_min)
    
        safe_pos_min = jnp.where(valid, pos_min, container_min + half)
        safe_pos_max = jnp.where(valid, pos_max, container_max - half)
    
        pos = jax.random.uniform(
            rng,
            shape=(3,),
            minval=safe_pos_min,
            maxval=safe_pos_max,
            dtype=jnp.float32,
        )
    
        pos = jnp.round(pos, 3)
        return pos, bool(valid)


    def _build_box_specs(self):
        box_specs = []
        rng = self.rng
    
        for i in range(self.num_boxes):
            dims, weight, rng = self.sample_box(rng)
            quat, rng = self.sample_quaternion(rng)
    
            dims_np = [float(v) for v in dims]
            hx, hy, hz = [d / 2.0 for d in dims_np]
    
            box_specs.append({
                "dims": dims_np,
                "half": [hx, hy, hz],
                "mass": max(float(weight), 0.001),
                "quat": [float(v) for v in quat],
                "rgba": [0.2, min(0.3 + 0.12 * i, 1.0), max(0.8 - 0.1 * i, 0.2), 1.0]
            })
    
        self.rng = rng
        return box_specs

    
    def build_xml(self, box_specs):
        bodies = []
        for i, b in enumerate(box_specs):
            hx, hy, hz = b["half"]
            mass = b["mass"]
            r, g, bl, a = b["rgba"]
    
            park_x = 10.0 + 2.0 * i
            park_y = 0.0
            park_z = hz + 0.05
    
            bodies.append(f"""
            <body name="box{i}" pos="{park_x} {park_y} {park_z}">
              <freejoint/>
              <geom name="box{i}_geom"
                    type="box"
                    size="{hx} {hy} {hz}"
                    mass="{mass}"
                    rgba="{r} {g} {bl} {a}"
                    friction="0.9 0.05 0.05"/>
            </body>
            """)
    
        xml = f"""
        <mujoco model="sequential_random_xyz_drop">
          <option timestep="0.005" gravity="0 0 -9.81" iterations="50"/>
          
          <default>
              <geom condim="3"
                    friction="0.9 0.05 0.05"
                    solref="0.002 1"
                    solimp="0.99 0.999 0.001"/>
            </default>
        
          <worldbody>
            <geom name="floor"
                  type="plane"
                  pos="1.0 1.3 0"
                  size="1.0 1.3 0.1"
                  rgba="0.9 0.9 0.9 1"/>
        
            <geom name="wall_y0"
                  type="box"
                  pos="1.0 -0.01 1.375"
                  size="1.0 0.01 1.375"
                  rgba="0.7 0.7 0.7 0.4"/>
                  
            <geom name="wall_y2p6"
                  type="box"
                  pos="1.0 2.61 1.375"
                  size="1.0 0.01 1.375"
                  rgba="0.7 0.7 0.7 0.4"/>
            
            <geom name="wall_x0"
                  type="box"
                  pos="-0.01 1.3 1.375"
                  size="0.01 1.3 1.375"
                  rgba="0.7 0.7 0.7 0.4"/>
                  
            <geom name="wall_x2"
                  type="box"
                  pos="2.01 1.3 1.375"
                  size="0.01 1.3 1.375"
                  rgba="0.7 0.7 0.7 0.4"/>
          
            <geom name="ceiling_z2p75"
              type="box"
              pos="1.0 1.3 2.76"
              size="1.0 1.3 0.01"
              rgba="0.7 0.7 0.7 0.4"/>
        
            {''.join(bodies)}
          </worldbody>
        </mujoco>
        """
        return xml


    def get_box_xy_polygon(self, box_idx):
    
        geom_id = self.box_geom_ids[box_idx]
    
        center = np.array(self.data.geom_xpos[geom_id])          # (3,)
        rot = np.array(self.data.geom_xmat[geom_id]).reshape(3, 3)  # (3,3)
    
        hx, hy, hz = self.model.geom_size[geom_id]
    
        local_corners = np.array([
            [-hx, -hy, -hz],
            [-hx, -hy,  hz],
            [-hx,  hy, -hz],
            [-hx,  hy,  hz],
            [ hx, -hy, -hz],
            [ hx, -hy,  hz],
            [ hx,  hy, -hz],
            [ hx,  hy,  hz],
        ], dtype=np.float32)
    
        world_corners = center[None, :] + local_corners @ rot.T
    
        xy = world_corners[:, :2]
        return xy


    def get_box_yz_polygon(self, box_idx):
    
        geom_id = self.box_geom_ids[box_idx]
    
        center = np.array(self.data.geom_xpos[geom_id])              # (3,)
        rot = np.array(self.data.geom_xmat[geom_id]).reshape(3, 3)   # (3,3)
    
        hx, hy, hz = self.model.geom_size[geom_id]
    
        local_corners = np.array([
            [-hx, -hy, -hz],
            [-hx, -hy,  hz],
            [-hx,  hy, -hz],
            [-hx,  hy,  hz],
            [ hx, -hy, -hz],
            [ hx, -hy,  hz],
            [ hx,  hy, -hz],
            [ hx,  hy,  hz],
        ], dtype=np.float32)
    
        world_corners = center[None, :] + local_corners @ rot.T
    
        yz = world_corners[:, 1:3]   # project to YZ plane
        return yz


    def get_box_xz_polygon(self, box_idx):
    
        geom_id = self.box_geom_ids[box_idx]
    
        center = np.array(self.data.geom_xpos[geom_id])              # (3,)
        rot = np.array(self.data.geom_xmat[geom_id]).reshape(3, 3)   # (3,3)
    
        hx, hy, hz = self.model.geom_size[geom_id]
    
        local_corners = np.array([
            [-hx, -hy, -hz],
            [-hx, -hy,  hz],
            [-hx,  hy, -hz],
            [-hx,  hy,  hz],
            [ hx, -hy, -hz],
            [ hx, -hy,  hz],
            [ hx,  hy, -hz],
            [ hx,  hy,  hz],
        ], dtype=np.float32)
    
        world_corners = center[None, :] + local_corners @ rot.T
    
        xz = world_corners[:, [0, 2]]   # project to XZ plane
        return xz


    @staticmethod
    def convex_hull_2d(points):
    
        pts = np.unique(points, axis=0)
        if len(pts) <= 1:
            return pts
    
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))
    
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))
    
        hull = np.array(lower[:-1] + upper[:-1], dtype=np.float32)
        return hull

    
    def _build_grid_points(self):
        nx = ny = nz = self.grid_res
    
        # XY points
        x_min, x_max = 0.0, 2.0
        y_min, y_max = 0.0, 2.6
        x_centers = np.linspace(
            x_min + (x_max - x_min) / (2 * nx),
            x_max - (x_max - x_min) / (2 * nx),
            nx,
            dtype=np.float32,
        )
        y_centers = np.linspace(
            y_min + (y_max - y_min) / (2 * ny),
            y_max - (y_max - y_min) / (2 * ny),
            ny,
            dtype=np.float32,
        )
        X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
        self.xy_cell_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    
        # YZ points
        z_min, z_max = 0.0, 2.75
        Y2, Z2 = np.meshgrid(y_centers, np.linspace(
            z_min + (z_max - z_min) / (2 * nz),
            z_max - (z_max - z_min) / (2 * nz),
            nz,
            dtype=np.float32,
        ), indexing="ij")
        self.yz_cell_points = np.stack([Y2.ravel(), Z2.ravel()], axis=1)
    
        # XZ points
        X3, Z3 = np.meshgrid(x_centers, np.linspace(
            z_min + (z_max - z_min) / (2 * nz),
            z_max - (z_max - z_min) / (2 * nz),
            nz,
            dtype=np.float32,
        ), indexing="ij")
        self.xz_cell_points = np.stack([X3.ravel(), Z3.ravel()], axis=1)


    def build_xy_grid_exact(self):
    
        nx = ny = self.grid_res
        cell_points = self.xy_cell_points
    
        grid = np.zeros((nx, ny), dtype=np.int32)
    
        for i in range(self.current_box):
            hull = self.convex_hull_2d(self.get_box_xy_polygon(i))
    
            if len(hull) < 3:
                continue
    
            inside = Path(hull).contains_points(cell_points).reshape(nx, ny)
            grid = np.maximum(grid, inside.astype(np.int32))
    
        return grid


    def build_xy_grid_exact_old(self):
        
        nx = ny = self.grid_res
        x_min, x_max = 0.0, 2.0
        y_min, y_max = 0.0, 2.6
        
        x_centers = np.linspace(x_min + (x_max - x_min)/(2*nx),
                                x_max - (x_max - x_min)/(2*nx),
                                nx, dtype=np.float32)
        y_centers = np.linspace(y_min + (y_max - y_min)/(2*ny),
                                y_max - (y_max - y_min)/(2*ny),
                                ny, dtype=np.float32)
    
        X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
        cell_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    
        grid = np.zeros((nx, ny), dtype=np.int32)
    
        for i in range(self.current_box):
            hull = self.convex_hull_2d(self.get_box_xy_polygon(i))
    
            if len(hull) < 3:
                continue
    
            inside = Path(hull).contains_points(cell_points).reshape(nx, ny)
            grid = np.maximum(grid, inside.astype(np.int32))
    
        return grid


    def build_yz_grid_exact_old(self):
        
        ny = nz = self.grid_res
        y_min, y_max = 0.0, 2.6
        z_min, z_max = 0.0, 2.75
        
        y_centers = np.linspace(y_min + (y_max - y_min)/(2*ny),
                                y_max - (y_max - y_min)/(2*ny),
                                ny, dtype=np.float32)
        z_centers = np.linspace(z_min + (z_max - z_min)/(2*nz),
                                z_max - (z_max - z_min)/(2*nz),
                                nz, dtype=np.float32)
    
        Y, Z = np.meshgrid(y_centers, z_centers, indexing="ij")
        cell_points = np.stack([Y.ravel(), Z.ravel()], axis=1)
        grid = np.zeros((ny, nz), dtype=np.int32)
    
        for i in range(self.current_box):
            hull = self.convex_hull_2d(self.get_box_yz_polygon(i))
    
            if len(hull) < 3:
                continue
    
            inside = Path(hull).contains_points(cell_points).reshape(ny, nz)
            grid = np.maximum(grid, inside.astype(np.int32))
    
        return grid
    
    
    def build_yz_grid_exact(self):
    
        ny = nz = self.grid_res
        cell_points = self.yz_cell_points
        grid = np.zeros((ny, nz), dtype=np.int32)
    
        for i in range(self.current_box):
            hull = self.convex_hull_2d(self.get_box_yz_polygon(i))
    
            if len(hull) < 3:
                continue
    
            inside = Path(hull).contains_points(cell_points).reshape(ny, nz)
            grid = np.maximum(grid, inside.astype(np.int32))
    
        return grid


    def build_xz_grid_exact_old(self):
        nx = nz = self.grid_res
        x_min, x_max = 0.0, 2.0
        z_min, z_max = 0.0, 2.75
    
        x_centers = np.linspace(x_min + (x_max - x_min)/(2*nx),
                                x_max - (x_max - x_min)/(2*nx),
                                nx, dtype=np.float32)
        z_centers = np.linspace(z_min + (z_max - z_min)/(2*nz),
                                z_max - (z_max - z_min)/(2*nz),
                                nz, dtype=np.float32)
    
        X, Z = np.meshgrid(x_centers, z_centers, indexing="ij")
        cell_points = np.stack([X.ravel(), Z.ravel()], axis=1)
    
        grid = np.zeros((nx, nz), dtype=np.int32)
    
        for i in range(self.current_box):
            hull = self.convex_hull_2d(self.get_box_xz_polygon(i))
    
            if len(hull) < 3:
                continue
    
            inside = Path(hull).contains_points(cell_points).reshape(nx, nz)
            grid = np.maximum(grid, inside.astype(np.int32))
    
        return grid


    def build_xz_grid_exact(self):
        nx = nz = self.grid_res
        cell_points = self.xz_cell_points
    
        grid = np.zeros((nx, nz), dtype=np.int32)
    
        for i in range(self.current_box):
            hull = self.convex_hull_2d(self.get_box_xz_polygon(i))
    
            if len(hull) < 3:
                continue
    
            inside = Path(hull).contains_points(cell_points).reshape(nx, nz)
            grid = np.maximum(grid, inside.astype(np.int32))
    
        return grid


    def _settle_scene(self, viewer=None):
        check_every = 10
        vel_threshold_sq = self.vel_threshold * self.vel_threshold
    
        for k in range(self.settle_steps_max):
            mujoco.mj_step(self.model, self.data)
        
            if k % 20 == 0:
                self.data.qvel[:] *= 0.5
    
            if viewer is not None and (k % check_every == 0):
                viewer.sync()
    
            if k % check_every != 0:
                continue
    
            all_settled = True
            for i in range(self.current_box):
                qvel_adr_i = self.box_qvel_adrs[i]
                vel = self.data.qvel[qvel_adr_i:qvel_adr_i + 6]
    
                if np.dot(vel, vel) > vel_threshold_sq:
                    all_settled = False
                    break
    
            if all_settled:
                break
        
        return k
            

    def _release_next_box(self, viewer=None):
        dims = jnp.array(self.box_specs[self.current_box]["dims"], dtype=jnp.float32)
        quat = jnp.array(self.box_specs[self.current_box]["quat"], dtype=jnp.float32)
        dims_rot = self.rotated_dims_from_quat(dims, quat)

        self.rng, rng_pos = jax.random.split(self.rng)
        pos, found = self.sample_position(rng_pos, dims_rot)

        if not found:
            print(f"Could not find non-overlapping start for box {self.current_box}")
            self.current_box += 1
            self.next_release_time += self.release_interval
            return

        px, py, pz = [float(v) for v in pos]

        qpos_adr = self.box_qpos_adrs[self.current_box]
        qvel_adr = self.box_qvel_adrs[self.current_box]

        self.data.qpos[qpos_adr:qpos_adr + 3] = [px, py, pz]
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = self.box_specs[self.current_box]["quat"]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        released_idx = self.current_box
        self.current_box += 1
        
        self._settle_scene(viewer)

        if self.render:
            self._draw_grids()

        print(f"Released box {released_idx}")
        print("  sampled pos =", [px, py, pz])

        self.next_release_time += self.release_interval
        

    def get_grids(self):
        xy_grid = self.build_xy_grid_exact()
        yz_grid = self.build_yz_grid_exact()
        xz_grid = self.build_xz_grid_exact()
        
        return xy_grid, yz_grid, xz_grid


    def get_state(self):
        
        xy_grid, yz_grid, xz_grid = self.get_grids()
        state = np.stack([xy_grid, yz_grid, xz_grid], axis=0).astype(np.float32)
        return state
    
    
    def sample_action(self):
        dims = jnp.array(self.box_specs[self.current_box]["dims"], dtype=jnp.float32)
        quat = jnp.array(self.box_specs[self.current_box]["quat"], dtype=jnp.float32)
        dims_rot = self.rotated_dims_from_quat(dims, quat)
    
        self.rng, rng_pos = jax.random.split(self.rng)
        pos, found = self.sample_position(rng_pos, dims_rot)
    
        action = {
            "pos": pos,
            "quat": quat,
            "valid": found
        }
    
        return action


    def _get_box_centers(self, num_boxes=None):
        if num_boxes is None:
            num_boxes = self.current_box
    
        centers = []
        for i in range(num_boxes):
            geom_id = self.box_geom_ids[i]
            center = np.array(self.data.geom_xpos[geom_id], dtype=np.float32)
            centers.append(center)
    
        if len(centers) == 0:
            return np.zeros((0, 3), dtype=np.float32)
    
        return np.stack(centers, axis=0)


    def reset(self, *, seed=None, options=None):
        
        self.prev_density = 0.0 
        if seed is not None:
            self.seed = seed
    
        self.rng = jax.random.PRNGKey(self.seed)
        self.current_box = 0
        self.next_release_time = 0.0
    
        # self.box_specs = self._build_box_specs()
        # self.xml = self.build_xml(self.box_specs)
        # self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
    
        if self.render:
            plt.clf()
        
        obs = self.get_state()
        self.state_cache = obs.copy()
        info = {}
    
        return obs, info
    

    def compute_reward(
        self,
        prev_state,
        next_state,
        assigned_pos,
        released_idx,
        terminated_early=False,
        terminate=False,
        termination_reason=None,
        k_settle=10,
    ):
        # ----- current density and max_x -----
        total_volume = 0.0
        max_x = 0.0
        # print(k_settle)
    
        for i in range(self.current_box):
            dims = self.box_specs[i]["dims"]
            vol = dims[0] * dims[1] * dims[2]
            total_volume += vol
    
            geom_id = self.box_geom_ids[i]
            center_x = self.data.geom_xpos[geom_id][0]
            hx = self.model.geom_size[geom_id][0]
            x_max_i = center_x + hx
            if x_max_i > max_x:
                max_x = x_max_i
    
        density = 0.0 if max_x <= 1e-6 else total_volume / (max_x * 2.6 * 2.75)
    
        if not hasattr(self, "prev_density"):
            self.prev_density = density
        if not hasattr(self, "prev_max_x"):
            self.prev_max_x = max_x
    
        density_delta = (density - self.prev_density)
        front_expansion = max(0.0, max_x - self.prev_max_x)
    
        self.prev_density = density
        self.prev_max_x = max_x
    
        # ----- placement drift -----
        # geom_id = self.box_geom_ids[released_idx]
        # settled_pos = np.array(self.data.geom_xpos[geom_id], dtype=np.float32)
        # assigned_pos = np.array(assigned_pos, dtype=np.float32)
        # position_error = float(np.linalg.norm(settled_pos - assigned_pos))
    
        #print(density_delta, front_expansion, k_settle)
        # ----- reward -----
        reward = (
            100.0 * density_delta
            - 2.0 * front_expansion
            - 0.02 * k_settle
        )
        
        # # optional tiny helper
        # prev_occ = float(np.sum(prev_state))
        # next_occ = float(np.sum(next_state))
        # occ_delta = next_occ - prev_occ
        # reward += 0.01 * occ_delta
            
        if termination_reason in ["out_of_bounds", "overlap"]:
            reward -= 25.0

        if termination_reason == "unstable":
            reward -= 10.0
    
        if termination_reason == "completed":
            reward += 20.0
        
        if terminate:
            reward += 2 * density
    
        return reward, density
    

    def _quat_wxyz_to_rotmat(self, quat):
        qw, qx, qy, qz = [float(v) for v in quat]
    
        n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if n < 1e-8:
            return np.eye(3, dtype=np.float32)
    
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    
        return np.array([
            [1 - 2 * (qy * qy + qz * qz),     2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
            [    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),     2 * (qy * qz - qx * qw)],
            [    2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ], dtype=np.float32)


    def _allowed_quaternions(self):
        s = np.sqrt(0.5).astype(np.float32)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],   # identity
            [s,   0.0, 0.0, s],     # 90 deg around Z
            [s,   0.0, s,   0.0],   # 90 deg around Y
            [s,   s,   0.0, 0.0],   # 90 deg around X
        ], dtype=np.float32)


    def _rotation_from_scalar(self, rot_scalar):
        allowed = self._allowed_quaternions()
    
        # rot_scalar expected in [-1, 1]
        r = float(np.clip(rot_scalar, -1.0, 1.0))
    
        if r < -0.5:
            idx = 0
        elif r < 0.0:
            idx = 1
        elif r < 0.5:
            idx = 2
        else:
            idx = 3
    
        return allowed[idx], idx

    def _candidate_world_corners(self, pos, quat, dims):
        px, py, pz = [float(v) for v in pos]
        dx, dy, dz = [float(v) for v in dims]
    
        hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    
        local_corners = np.array([
            [-hx, -hy, -hz],
            [-hx, -hy,  hz],
            [-hx,  hy, -hz],
            [-hx,  hy,  hz],
            [ hx, -hy, -hz],
            [ hx, -hy,  hz],
            [ hx,  hy, -hz],
            [ hx,  hy,  hz],
        ], dtype=np.float32)
    
        R = self._quat_wxyz_to_rotmat(quat)
        center = np.array([px, py, pz], dtype=np.float32)
    
        world_corners = center[None, :] + local_corners @ R.T
        return world_corners

    
    def _assigned_box_overlaps_old_boxes(self, assigned_pos, assigned_quat, released_idx):
        dims = self.box_specs[released_idx]["dims"]
    
        cand_corners = self._candidate_world_corners(
            pos=assigned_pos,
            quat=assigned_quat,
            dims=dims,
        )
        cand_mins = cand_corners.min(axis=0)
        cand_maxs = cand_corners.max(axis=0)
    
        # Compare only against boxes that existed before the current one
        for i in range(released_idx):
            geom_id = self.box_geom_ids[i]
    
            center = np.array(self.data.geom_xpos[geom_id], dtype=np.float32)
            rot = np.array(self.data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
            hx, hy, hz = self.model.geom_size[geom_id]
    
            local_corners = np.array([
                [-hx, -hy, -hz],
                [-hx, -hy,  hz],
                [-hx,  hy, -hz],
                [-hx,  hy,  hz],
                [ hx, -hy, -hz],
                [ hx, -hy,  hz],
                [ hx,  hy, -hz],
                [ hx,  hy,  hz],
            ], dtype=np.float32)
    
            world_corners = center[None, :] + local_corners @ rot.T
            box_mins = world_corners.min(axis=0)
            box_maxs = world_corners.max(axis=0)
    
            if self._aabb_overlap(cand_mins, cand_maxs, box_mins, box_maxs):
                return True
    
        return False



    def _aabb_overlap(self, mins1, maxs1, mins2, maxs2, eps=1e-6):
        overlap_x = (mins1[0] < maxs2[0] - eps) and (maxs1[0] > mins2[0] + eps)
        overlap_y = (mins1[1] < maxs2[1] - eps) and (maxs1[1] > mins2[1] + eps)
        overlap_z = (mins1[2] < maxs2[2] - eps) and (maxs1[2] > mins2[2] + eps)
        return overlap_x and overlap_y and overlap_z


    def check_termination(
            self,
            pre_settle_centers=None,
            displacement_threshold=0.10,
            assigned_pos=None,
            assigned_quat=None,
            released_idx=None,
        ):
        
        eps = 0.02
        truck_min = np.array([0.0, 0.0, 0.0], dtype=np.float32) - eps
        truck_max = np.array([2.0, 2.6, 2.75], dtype=np.float32) + eps
    
        for i in range(self.current_box):
            geom_id = self.box_geom_ids[i]
    
            center = np.array(self.data.geom_xpos[geom_id], dtype=np.float32)
            rot = np.array(self.data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
            hx, hy, hz = self.model.geom_size[geom_id]
    
            local_corners = np.array([
                [-hx, -hy, -hz],
                [-hx, -hy,  hz],
                [-hx,  hy, -hz],
                [-hx,  hy,  hz],
                [ hx, -hy, -hz],
                [ hx, -hy,  hz],
                [ hx,  hy, -hz],
                [ hx,  hy,  hz],
            ], dtype=np.float32)
    
            world_corners = center[None, :] + local_corners @ rot.T
    
            if np.any(world_corners < truck_min) or np.any(world_corners > truck_max):
                # mins = world_corners.min(axis=0)
                # maxs = world_corners.max(axis=0)
                # print("mins:", mins, "maxs:", maxs)
                return True, "out_of_bounds"
    
        if assigned_pos is not None and assigned_quat is not None and released_idx is not None:
            if self._assigned_box_overlaps_old_boxes(
                assigned_pos=assigned_pos,
                assigned_quat=assigned_quat,
                released_idx=released_idx,
            ):
                return True, "overlap"
    
        if pre_settle_centers is not None and len(pre_settle_centers) > 0:
            num_old_boxes = pre_settle_centers.shape[0]
            post_settle_centers = self._get_box_centers(num_old_boxes)
    
            displacements = np.linalg.norm(post_settle_centers - pre_settle_centers, axis=1)
            unstable_count = int(np.sum(displacements > displacement_threshold))
    
            if unstable_count >= 3:
                return False, "unstable"
    
        if self.current_box >= self.num_boxes:
            return True, "completed"
    
        return False, None


    def is_done(self):
        return self.current_box >= self.num_boxes


    def step(self, action, viewer=None):
        
        if viewer is not None:
            self.viewer = viewer
        if self.render and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25
            self.viewer.cam.distance = 6.0
            self.viewer.cam.lookat[:] = [1.0, 1.3, 1.0]
        truncate = False
        
        if self.is_done():
            state = self.state_cache.copy()
            return state, 0.0, True, False, {"msg": "Episode already done"}
    
        prev_state = self.state_cache.copy()
    
        action = np.asarray(action, dtype=np.float32)
    
        px, py, pz = action[0:3]
        rot_scalar = action[3]
        
        quat, rot_idx = self._rotation_from_scalar(rot_scalar)
        qw, qx, qy, qz = quat
    
        qpos_adr = self.box_qpos_adrs[self.current_box]
        qvel_adr = self.box_qvel_adrs[self.current_box]
        
        # snapshot centers of boxes that already existed before this new placement
        pre_settle_centers = self._get_box_centers(self.current_box)
        
        assigned_pos = np.array([px, py, pz], dtype=np.float32)
        
        self.data.qpos[qpos_adr:qpos_adr + 3] = [px, py, pz]
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [qw, qx, qy, qz]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        released_idx = self.current_box
        self.current_box += 1
        
        k_settle = self._settle_scene(self.viewer)
    
        if self.render:
            self._draw_grids()
    
        next_state = self.get_state()
        self.state_cache = next_state.copy()
        
        terminate, termination_reason = self.check_termination(
            pre_settle_centers=pre_settle_centers,
            displacement_threshold=0.10,
            assigned_pos=[px, py, pz],
            assigned_quat=[qw, qx, qy, qz],
            released_idx=released_idx,
        )

        reward, density = self.compute_reward(
            prev_state=prev_state,
            next_state=next_state,
            assigned_pos=assigned_pos,
            released_idx=released_idx,
            terminated_early=(terminate and termination_reason != "completed"),
            terminate=terminate,
            termination_reason=termination_reason,
            k_settle=k_settle
        )
        
        info = {
            "valid_action": True,
            "released_box": released_idx,
            "action_pos": [float(px), float(py), float(pz)],
            "action_quat": [float(qw), float(qx), float(qy), float(qz)],
            "box_dims": self.box_specs[released_idx]["dims"],
            "box_mass": self.box_specs[released_idx]["mass"],
            "termination_reason": termination_reason,
            "current_density": k_settle,
            "final_density": density if terminate else 0,
        }
        
        if self.viewer is not None:
            self.viewer.sync()
        
        # if terminate == True:
        #     print(density)
        
        return next_state, reward, terminate, truncate, info


    def _draw_grids(self):
        xy_grid, yz_grid, xz_grid = self.get_grids()
        
        plt.subplot(1, 3, 1)
        plt.imshow(
            xy_grid.T,
            origin="lower",
            extent=[0.0, 2.0, 0.0, 2.6],
            vmin=0,
            vmax=1,
            cmap="gray",
        )
        plt.title("Exact XY")
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.subplot(1, 3, 2)
        plt.imshow(
            yz_grid.T,
            origin="lower",
            extent=[0.0, 2.6, 0.0, 2.75],
            vmin=0,
            vmax=1,
            cmap="gray",
        )
        plt.title("Exact YZ")
        plt.xlabel("Y")
        plt.ylabel("Z")

        plt.subplot(1, 3, 3)
        plt.imshow(
            xz_grid.T,
            origin="lower",
            extent=[0.0, 2.0, 0.0, 2.75],
            vmin=0,
            vmax=1,
            cmap="gray",
        )
        plt.title("Exact XZ")
        plt.xlabel("X")
        plt.ylabel("Z")

        plt.pause(0.01)
        plt.clf()
        
