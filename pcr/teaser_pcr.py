# code adapted from TEASER_PLUSPLUS official implementation:
# https://github.com/MIT-SPARK/TEASER-plusplus
from pathlib import Path
import open3d as o3d
import teaserpp_python
import numpy as np 
import copy

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def get_teaser_solver(noise_bound = 0.002):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = True
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 1000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver

def parse_3d_corr(path):
    query_pts = []
    ref_pts = []
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q = p.split()
            query_pts.append(q[0:3])
            ref_pts.append(q[3:6])
    
    query_pts = np.asarray(query_pts).reshape((-1,3)).T
    ref_pts = np.asarray(ref_pts).reshape((-1,3)).T   
    
    return  query_pts, ref_pts

def main(corr_path, pcd_1, pcd_2, VISUALIZE = True):
    
    if VISUALIZE:
        A_pcd = o3d.io.read_point_cloud(pcd_1)
        B_pcd = o3d.io.read_point_cloud(pcd_2)
        # A_pcd = A_pcd.voxel_down_sample(voxel_size=0.04)
        # B_pcd = B_pcd.voxel_down_sample(voxel_size=0.04)
        A_pcd.paint_uniform_color([1.0, 0.5, 0.0])
        o3d.visualization.draw_geometries([A_pcd]) # plot A and B 
        o3d.visualization.draw_geometries([B_pcd])

    A_corr, B_corr = parse_3d_corr(corr_path)
    num_corrs = A_corr.shape[1]
    
    if VISUALIZE:
        points = np.concatenate((A_corr.T,B_corr.T),axis=0)
        lines = []
        for i in range(num_corrs):
            lines.append([i,i+num_corrs])
        colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])
    
    #Teaser solver for pcr
    NOISE_BOUND = 0.1
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr,B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    scale_teaser = solution.scale
    T_teaser = Rt2T(R_teaser*scale_teaser,t_teaser)
    
    if VISUALIZE:
        A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)
        o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd])
        localized_pcd = Path(pcd_1).parent/'localized.ply'
        o3d.io.write_point_cloud(str(localized_pcd), A_pcd_T_teaser)
    # icp refinement using result from teaser
    # icp_sol = o3d.registration.registration_icp(
    #   A_pcd, B_pcd, 0.04, T_teaser,
    #   o3d.registration.TransformationEstimationPointToPoint())
    # T_icp = icp_sol.transformation
    
    # if VISUALIZE:
    #     A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
    #     o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd])
    
    return T_teaser, scale_teaser
    
    

if __name__ == '__main__':
    proj_path = Path('/home/marvin/ETH_Study/3DV/3DV/datasets/test1/')
    corr_path = proj_path/'q_ref_match/3d_corr.txt'
    query_pcd = proj_path/'query/outputs/point_cloud.ply'
    ref_pcd = proj_path/'ref/outputs/point_cloud.ply'
    T = main(str(corr_path), str(query_pcd), str(ref_pcd))
    print(T)
    
    