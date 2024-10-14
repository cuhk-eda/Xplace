# We follow the work [1] and [2] to implement the macro legalizer.
# [1] Cong, Jason, and Min Xie. "A robust mixed-size legalization and detailed placement algorithm." IEEE TCAD 2008.
# [2] Moffitt, M. D., Ng, A. N., Markov, I. L., & Pollack, M. E. "Constraint-driven floorplan repair." ACM TODAES 2008.

# NOTE: Known Issue: Adding graph edge constraint in LP is very slow when num_macros is large.
#       Because pulp lib use Python OrderDict to store all constraints, when #Constraints
#       is large, the performance is bad.
#       Re-write the LP in C++ may achieve some speed up.
# TODO: 1) macro spreading for routability optimization
#       2) graph pruning for speed up

import numpy as np
import numba as nb
import logging
import pulp as pl
import igraph as ig


pulp_logger = logging.getLogger('pulp')
pulp_logger.setLevel(logging.INFO)
use_numba_parallel = False

@nb.jit(nopython=True, cache=True, parallel=use_numba_parallel)
def check_macro_legality(macro_pos, macro_size, macro_fixed, die_info, check_all=True):
    num_macros = macro_pos.shape[0]
    # overlap = np.zeros((num_macros, num_macros), dtype=np.bool8)
    legal = True
    overlaps = []
    for i in nb.prange(num_macros):
        lx_i = macro_pos[i][0] - macro_size[i][0] / 2
        ly_i = macro_pos[i][1] - macro_size[i][1] / 2
        hx_i = macro_pos[i][0] + macro_size[i][0] / 2
        hy_i = macro_pos[i][1] + macro_size[i][1] / 2
        for j in nb.prange(num_macros):
            if i >= j:
                continue
            lx_j = macro_pos[j][0] - macro_size[j][0] / 2
            ly_j = macro_pos[j][1] - macro_size[j][1] / 2
            hx_j = macro_pos[j][0] + macro_size[j][0] / 2
            hy_j = macro_pos[j][1] + macro_size[j][1] / 2
            if min(hx_i, hx_j) - max(lx_i, lx_j) > 1e-3 and min(hy_i, hy_j) - max(ly_i, ly_j) > 1e-3:
                # overlap[i][j] = True
                # overlap[j][i] = True
                legal = False
                if not check_all and not legal:
                    return legal, overlaps
                overlaps.append((i, j))
                # print("Macro", i, "and Macro", j, "Overlap.")

    return legal, overlaps
    
    

@nb.jit(nopython=True, nogil=True, cache=True, parallel=use_numba_parallel)
def constraint_graph_construction(
    macro_pos, macro_size, macro_fixed, die_info, prune=True
):
    num_macros = macro_pos.shape[0]
    macro_lpos = macro_pos - macro_size / 2
    edge_type = np.zeros((num_macros, num_macros), dtype=np.int8) # 0: x, 1: y, -1: None
    edge_dist_x = np.zeros((num_macros, num_macros), dtype=np.float32)
    edge_dist_y = np.zeros((num_macros, num_macros), dtype=np.float32)
    edge_type[:, :] = -1
    for i in nb.prange(num_macros):
        for  j in nb.prange(num_macros):
            if i >= j:
                continue
            # Detect x/y order
            if macro_pos[i][0] <= macro_pos[j][0]:
                # x_i -> x_j
                x_order = 0
            else:
                # x_j -> x_i
                x_order = 1
            if macro_pos[i][1] <= macro_pos[j][1]:
                # y_i -> y_j
                y_order = 0
            else:
                # y_j -> y_i
                y_order = 1

            # Calculate displacement
            lx_i = macro_pos[i][0] - macro_size[i][0] / 2
            lx_j = macro_pos[j][0] - macro_size[j][0] / 2
            hx_i = macro_pos[i][0] + macro_size[i][0] / 2
            hx_j = macro_pos[j][0] + macro_size[j][0] / 2
            if x_order == 0:
                dist_x = lx_j - hx_i
            else:
                dist_x = lx_i - hx_j

            ly_i = macro_pos[i][1] - macro_size[i][1] / 2
            ly_j = macro_pos[j][1] - macro_size[j][1] / 2
            hy_i = macro_pos[i][1] + macro_size[i][1] / 2
            hy_j = macro_pos[j][1] + macro_size[j][1] / 2
            if y_order == 0:
                dist_y = ly_j - hy_i
            else:
                dist_y = ly_i - hy_j

            # dist martix is undirected and symmetric
            edge_dist_x[i][j] = dist_x
            edge_dist_x[j][i] = dist_x
            edge_dist_y[i][j] = dist_y
            edge_dist_y[j][i] = dist_y

            # Determine the edge type (horizontal or vertical)
            if dist_x >= 0 and dist_y >= 0:
                # non-overlap
                edge_type[i][j] = 0 if dist_x >= dist_y else 1
            elif dist_x >= 0 and dist_y < 0:
                # y projection overlap
                edge_type[i][j] = 0
            elif dist_x < 0 and dist_y >= 0:
                # x projection overlap
                edge_type[i][j] = 1
            elif dist_x < 0 and dist_y < 0:
                # overlap
                edge_type[i][j] = 0 if dist_x >= dist_y else 1

            # Prune edges between objects without x/y projection overlap
            if prune:
                if edge_type[i][j] == 0 and not (ly_i <= hy_j and ly_j <= hy_i):
                    edge_type[i][j] = -1
                if edge_type[i][j] == 1 and not (lx_i <= hx_j and lx_j <= hx_i):
                    edge_type[i][j] = -1

            # Make sure edge orders
            if edge_type[i][j] == 0 and x_order == 1:
                edge_type[i][j] = -1
                edge_type[j][i] = 0
            elif edge_type[i][j] == 1 and y_order == 1:
                edge_type[i][j] = -1
                edge_type[j][i] = 1

    return edge_type, edge_dist_x, edge_dist_y


@nb.jit(nopython=True, nogil=True, cache=True, parallel=use_numba_parallel)
def initialize_xy_adj_weight(
    edge_type, adj_matrix, weight_matrix, macro_size, num_macros, num_nodes, s_id, t_id
):
    adj_matrix[s_id, :num_macros, :] = 1
    adj_matrix[:num_macros, t_id, :] = 1
    
    for i in nb.prange(num_nodes):
        for j in nb.prange(num_nodes):
            if i == j:
                continue
            if i < num_macros and j < num_macros:
                if edge_type[i][j] == 0:
                    # horizontal
                    adj_matrix[i][j][0] = 1
                elif edge_type[i][j] == 1:
                    adj_matrix[i][j][1] = 1
            if i >= num_macros:
                weight_matrix[i][j][0] = np.divide(macro_size[j][0], 2)
                weight_matrix[i][j][1] = np.divide(macro_size[j][1], 2)
            elif j >= num_macros:
                weight_matrix[i][j][0] = np.divide(macro_size[i][0], 2)
                weight_matrix[i][j][1] = np.divide(macro_size[i][1], 2)
            else:
                weight_matrix[i][j][0] = np.divide(np.add(macro_size[i][0], macro_size[j][0]), 2)
                weight_matrix[i][j][1] = np.divide(np.add(macro_size[i][1], macro_size[j][1]), 2)


@nb.jit(nopython=True, nogil=True, cache=True)
def compute_L_value(
    macro_pos, macro_fixed, axis, topo_order_out, L, affected_L, adj_matrix, weight_matrix, die_ll, s_id, num_macros
):
    for i in topo_order_out:
        if affected_L[i, axis] == 0:
            continue
        if i < num_macros:
            if macro_fixed[i]:
                L[i, axis] = macro_pos[i, axis]
                continue
        if i == s_id:
            L[i, axis] = die_ll[axis]
        else:
            is_preds = adj_matrix[:, i, axis]
            for j, is_pred in enumerate(is_preds):
                if is_pred == 0 or i == j:
                    continue
                # j -> i
                L[i, axis] = max(L[j, axis] + weight_matrix[j][i][axis], L[i, axis])


@nb.jit(nopython=True, nogil=True, cache=True)
def compute_R_value(
    macro_pos, macro_fixed, axis, topo_order_in, R, affected_R, adj_matrix, weight_matrix, die_ur, t_id, num_macros
):
    for i in topo_order_in:
        if affected_R[i, axis] == 0:
            continue
        if i < num_macros:
            if macro_fixed[i]:
                R[i, axis] = macro_pos[i, axis]
                continue
        if i == t_id:
            R[i, axis] = die_ur[axis]
        else:
            is_succs = adj_matrix[i, :, axis]
            for j, is_succ in enumerate(is_succs):
                if is_succ == 0 or i == j:
                    continue
                # i -> j
                R[i, axis] = min(R[j, axis] - weight_matrix[i][j][axis], R[i, axis])


def propagate_L_R(
    g, macro_pos, macro_fixed, affected_L, affected_R, adj_matrix, weight_matrix, 
    L, R, die_ll, die_ur, s_id, t_id, num_macros, edges_pair=None
):
    topo_order_out_X = nb.typed.List(g[0].topological_sorting(mode="out"))
    topo_order_in_X = nb.typed.List(g[0].topological_sorting(mode="in"))
    topo_order_out_Y = nb.typed.List(g[1].topological_sorting(mode="out"))
    topo_order_in_Y = nb.typed.List(g[1].topological_sorting(mode="in"))

    if edges_pair:
        axis_del, (u_del, v_del), axis_add, (u_add, v_add) = edges_pair
        affected_L[:, :] = False
        affected_R[:, :] = False
        # In the old graph, u may not be topologically <= v
        bfs_order, _, _ = g[axis_del].bfs(u_del, mode='out')
        affected_L[np.array(bfs_order), axis_del] = True
        bfs_order, _, _ = g[axis_del].bfs(v_del, mode='out')
        affected_L[np.array(bfs_order), axis_del] = True
        bfs_order, _, _ = g[axis_del].bfs(u_del, mode='in')
        affected_R[np.array(bfs_order), axis_del] = True
        bfs_order, _, _ = g[axis_del].bfs(v_del, mode='in')
        affected_R[np.array(bfs_order), axis_del] = True
        # In the new graph, u -> v, so u should be topologically <= v
        bfs_order, _, _ = g[axis_add].bfs(u_add, mode='out')
        affected_L[np.array(bfs_order), axis_add] = True
        bfs_order, _, _ = g[axis_add].bfs(v_add, mode='in')
        affected_R[np.array(bfs_order), axis_add] = True

    L[affected_L] = -np.inf
    R[affected_R] = np.inf
    compute_L_value(macro_pos, macro_fixed, 0, topo_order_out_X, L, affected_L,
                    adj_matrix, weight_matrix, die_ll, s_id, num_macros)
    compute_R_value(macro_pos, macro_fixed, 0, topo_order_in_X, R, affected_R,
                    adj_matrix, weight_matrix, die_ur, t_id, num_macros)
    compute_L_value(macro_pos, macro_fixed, 1, topo_order_out_Y, L, affected_L,
                    adj_matrix, weight_matrix, die_ll, s_id, num_macros)
    compute_R_value(macro_pos, macro_fixed, 1, topo_order_in_Y, R, affected_R,
                    adj_matrix, weight_matrix, die_ur, t_id, num_macros)


@nb.jit(nopython=True, nogil=True, cache=True, parallel=use_numba_parallel)
def compute_edge_slack(
    adj_matrix, weight_matrix, L, R, num_nodes, slack_matrix_e,
):
    for i in nb.prange(num_nodes):
        for j in nb.prange(num_nodes):
            if adj_matrix[i][j][0] == 1:
                slack_matrix_e[i][j][0] = R[j][0] - L[i][0] - weight_matrix[i][j][0]
            if adj_matrix[i][j][1] == 1:
                slack_matrix_e[i][j][1] = R[j][1] - L[i][1] - weight_matrix[i][j][1]


def slack_info(
    adj_matrix, weight_matrix, L, R, num_macros, num_nodes, slack_v, slack_matrix_e,
    update_edge_slack=True,
):  
    # Node Slack
    slack_v[:num_macros, :] = R[:num_macros, :] - L[:num_macros, :]
    x_nslack = np.minimum(slack_v[:, 0], 0)
    y_nslack = np.minimum(slack_v[:, 1], 0)
    x_tns, x_wns, nonzero_x = np.sum(x_nslack), np.min(x_nslack), np.sum(x_nslack < 0)
    y_tns, y_wns, nonzero_y = np.sum(y_nslack), np.min(y_nslack), np.sum(y_nslack < 0)
    info_v = (x_tns, x_wns, nonzero_x, y_tns, y_wns, nonzero_y)
    if update_edge_slack:
        slack_matrix_e[:,:,:] = 0
        compute_edge_slack(adj_matrix, weight_matrix, L, R, num_nodes, slack_matrix_e)
        x_nslack = np.minimum(slack_matrix_e[:, :, 0], 0)
        y_nslack = np.minimum(slack_matrix_e[:, :, 1], 0)
        x_tns, x_wns, nonzero_x = np.sum(x_nslack), np.min(x_nslack), np.sum(x_nslack < 0)
        y_tns, y_wns, nonzero_y = np.sum(y_nslack), np.min(y_nslack), np.sum(y_nslack < 0)
        info_e = (x_tns, x_wns, nonzero_x, y_tns, y_wns, nonzero_y)
        return info_v, info_e
    return info_v, None


@nb.jit(nopython=True, nogil=True, cache=True)
def mark_edge_to_move(adj_matrix, weight_matrix, slack_v, i, L, R, s_id, t_id, macro_pos):
    edges_pair = []
    x_slack = slack_v[i, 0]
    y_slack = slack_v[i, 1]
    if (x_slack >= 0 and y_slack >= 0) or (x_slack < 0 and y_slack < 0):
        return edges_pair
    if x_slack >= 0 and y_slack < 0:
        # need to handle in g[1]
        axis = 1
    else:
        # need to handle in g[0]
        axis = 0
    o_axis = 0 if axis == 1 else 1
    is_preds = adj_matrix[:, i, axis]
    for j, is_pred in enumerate(is_preds):
        # j -> i
        if is_pred == 0 or i == j:
            continue
        if j == s_id or j == t_id:
            continue
        if L[i][axis] == L[j][axis] + weight_matrix[j][i][axis]:
            edge_ij = False
            edge_ji = False
            if L[i][o_axis] + weight_matrix[i][j][o_axis] <= R[j][o_axis]:
                edge_ij = True
                is_succs_j = adj_matrix[j, :, o_axis]
                for k, is_succ in enumerate(is_succs_j):
                    # i -> j -> k
                    if is_succ == 0 or k == j:
                        continue
                    if L[i][o_axis] + weight_matrix[i][j][o_axis] + weight_matrix[j][k][o_axis] > R[k][o_axis]:
                        edge_ij = False
                        break
                    if L[j][o_axis] + weight_matrix[j][k][o_axis] > R[k][o_axis]:
                        edge_ij = False
                        break
            if L[j][o_axis] + weight_matrix[j][i][o_axis] <= R[i][o_axis]:
                edge_ji = True
                is_succs_i = adj_matrix[i, :, o_axis] 
                for k, is_succ in enumerate(is_succs_i):
                    # j -> i -> k
                    if is_succ == 0 or k == i:
                        continue
                    if L[j][o_axis] + weight_matrix[j][i][o_axis] + weight_matrix[i][k][o_axis] > R[k][o_axis]:
                        edge_ij = False
                        break
                    if L[i][o_axis] + weight_matrix[i][k][o_axis] > R[k][o_axis]:
                        edge_ij = False
                        break
            if edge_ij and edge_ji:
                if macro_pos[i][o_axis] <= macro_pos[j][o_axis]:
                    # i -> j
                    edges_pair.append((axis, (j, i), o_axis, (i, j)))
                else:
                    # j -> i
                    edges_pair.append((axis, (j, i), o_axis, (j, i)))
            elif edge_ij:
                edges_pair.append((axis, (j, i), o_axis, (i, j)))
            elif edge_ji:
                edges_pair.append((axis, (j, i), o_axis, (j, i)))
            else:
                edges_pair.clear()
    
    return edges_pair


def longest_path_refinement(macro_pos, macro_size, macro_fixed, die_info, die_ll, die_ur, logger, naive=False, prune=False):
    edge_type, _, _ = constraint_graph_construction(macro_pos, macro_size, macro_fixed, die_info, prune=prune)
    if naive:
        return edge_type, None, None
    logger.debug("Finish Graph construction.")
    die_lx, die_hx, die_ly, die_hy = die_info
    num_macros = macro_pos.shape[0]
    logger.debug("Longest Path Refinement #Macros: %d #FixedMacros: %d" % (num_macros, macro_fixed.sum()))
    num_nodes = num_macros + 2 # including source and target
    s_id = num_macros
    t_id = num_macros + 1
    dtype = macro_size.dtype

    adj_matrix = np.zeros((num_nodes, num_nodes, 2), dtype=np.int8)
    weight_matrix = np.full((num_nodes, num_nodes, 2), -1, dtype=dtype)

    initialize_xy_adj_weight(edge_type, adj_matrix, weight_matrix, macro_size,
        num_macros, num_nodes, s_id, t_id)

    edge_type[:, :] = -1 # not used anymore

    g_x = ig.Graph.Adjacency(adj_matrix[:,:,0], mode= "directed")
    g_y = ig.Graph.Adjacency(adj_matrix[:,:,1], mode= "directed")
    g = [g_x, g_y]
    # Use negative weight to find longest path
    if not g[0].is_dag():
        logger.warning("g_x is not a DAG.")
    if not g[1].is_dag():
        logger.warning("g_y is not a DAG.")

    # Calcuate x_L, x_R, y_L and y_R
    L = np.full((num_nodes, 2), -np.inf, dtype=dtype)
    R = np.full((num_nodes, 2), np.inf, dtype=dtype)
    affected_L = np.ones((num_nodes, 2), dtype=np.bool8)
    affected_R = np.ones((num_nodes, 2), dtype=np.bool8)
    # Propagate all nodes' L and R
    propagate_L_R(g, macro_pos, macro_fixed, affected_L, affected_R, adj_matrix, weight_matrix, 
        L, R, die_ll, die_ur, s_id, t_id, num_macros)
    # Calculate Node and Edge Slacks
    slack_v = np.zeros((num_macros, 2), dtype=dtype)
    slack_matrix_e = np.zeros((num_nodes, num_nodes, 2), dtype=dtype)
    info_v, info_e = slack_info(
        adj_matrix, weight_matrix, L, R, num_macros, num_nodes, slack_v, slack_matrix_e)
    logger.debug("Before longest path refinement:")
    logger.debug("  Node X: TNS/WNS/#NegSlks %.2f/%.2f/%d | Node Y: TNS/WNS/#NegSlks %.2f/%.2f/%d" % info_v)
    logger.debug("  Edge X: TNS/WNS/#NegSlks %.2f/%.2f/%d | Edge Y: TNS/WNS/#NegSlks %.2f/%.2f/%d" % info_e)

    # plot_negative_slack_macro(macro_pos, macro_size, macro_fixed, die_info, slack_v)

    macro_area = np.prod(macro_size, axis=1)
    num_trials = 0
    num_movement = 0
    while np.minimum(slack_v, 0).sum() < 0:
        if num_trials == 5:
            logger.error("Cannot fix longest path after %d trials." % num_trials)
            break
        macro_order = list(range(num_macros))
        sum_slack = slack_v.sum(axis=1)
        macro_order.sort(key=lambda x: (macro_area[x], -sum_slack[x], macro_pos[x][0], macro_pos[x][1], x))
        logger.debug("--- Trial %d ---" % num_trials)
        num_trials += 1
        for i in macro_order:
            # Mark edges to move
            edges_pair = mark_edge_to_move(adj_matrix, weight_matrix, slack_v, i, L, R, s_id, t_id, macro_pos)
            # Move selected edges
            for axis_del, (u_del, v_del), axis_add, (u_add, v_add) in edges_pair:
                g[axis_del].delete_edges([(u_del, v_del)])
                g[axis_add].add_edges([(u_add, v_add)])
                assert adj_matrix[u_del, v_del, axis_del] == 1
                assert adj_matrix[u_add, v_add, axis_add] == 0
                adj_matrix[u_del, v_del, axis_del] = 0
                adj_matrix[u_add, v_add, axis_add] = 1
                logger.debug("Move %d: G_%d (%d, %d) -> G_%d (%d, %d)." % (
                    num_movement, axis_del, u_del, v_del, axis_add, u_add, v_add))

                edges_pair = (axis_del, (u_del, v_del), axis_add, (u_add, v_add))
                # edges_pair = None # Debug only
                propagate_L_R(g, macro_pos, macro_fixed, affected_L, affected_R, adj_matrix, weight_matrix, 
                    L, R, die_ll, die_ur, s_id, t_id, num_macros, edges_pair=edges_pair)

                info_v, _ = slack_info(adj_matrix, weight_matrix, L, R, num_macros, num_nodes, slack_v, None,
                    update_edge_slack=False)
                logger.debug("  Updated Node X: TNS/WNS/#NegSlks %.2f/%.2f/%d | Node Y: TNS/WNS/#NegSlks %.2f/%.2f/%d" % info_v)
                num_movement += 1
                # plot_negative_slack_macro(macro_pos, macro_size, macro_fixed, die_info, slack_v)
    
    info_v, info_e = slack_info(
        adj_matrix, weight_matrix, L, R, num_macros, num_nodes, slack_v, slack_matrix_e)
    logger.debug("Finish longest path refinement:")
    logger.debug("  Node X: TNS/WNS/#NegSlks %.2f/%.2f/%d | Node Y: TNS/WNS/#NegSlks %.2f/%.2f/%d" % info_v)
    logger.debug("  Edge X: TNS/WNS/#NegSlks %.2f/%.2f/%d | Edge Y: TNS/WNS/#NegSlks %.2f/%.2f/%d" % info_e)

    assert (adj_matrix[:num_macros,:num_macros] == 1).all(axis=2).sum() == 0
    edge_type[:, :] = -1
    edge_type[adj_matrix[:num_macros,:num_macros,0] == 1] = 0
    edge_type[adj_matrix[:num_macros,:num_macros,1] == 1] = 1
    
    return edge_type, g_x, g_y


def basic_variable(macro_pos, macro_size, macro_fixed, die_info):
    num_macros = macro_pos.shape[0]
    die_lx, die_hx, die_ly, die_hy = die_info
    x_set, y_set, dx_set, dy_set = [], [], [], []
    for i in range(num_macros):
        if not macro_fixed[i]:
            x_set.append(pl.LpVariable(
                "x_%d" % i, macro_size[i][0] / 2, die_hx - macro_size[i][0] / 2
            ))
            y_set.append(pl.LpVariable(
                "y_%d" % i, macro_size[i][1] / 2, die_hy - macro_size[i][1] / 2
            ))
            dx_set.append(pl.LpVariable("d_x_%d" % i, 0, die_hx))
            dy_set.append(pl.LpVariable("d_y_%d" % i, 0, die_hy))
        else:
            x_value = macro_pos[i][0]
            y_value = macro_pos[i][1]
            x_set.append(pl.LpVariable("x_%d" % i, x_value, x_value))
            y_set.append(pl.LpVariable("y_%d" % i, y_value, y_value))
            dx_set.append(pl.LpVariable("d_x_%d" % i, 0, 0))
            dy_set.append(pl.LpVariable("d_y_%d" % i, 0, 0))

    for i in range(num_macros):
        x_value = macro_pos[i][0]
        y_value = macro_pos[i][1]
        x_set[i].setInitialValue(x_value)
        y_set[i].setInitialValue(y_value)
        dx_set[i].setInitialValue(0)
        dy_set[i].setInitialValue(0)
    return x_set, y_set, dx_set, dy_set


def macro_legalization_xy(args, logger, macro_pos, macro_size, macro_fixed, macro_weights, die_info, die_ll, die_ur,
                          num_items=None, lpbackend=None, naive=False, prune=False, edge_type=None):
    logger.info("Start macro_legalization_xy...")
    if num_items is not None:
        macro_pos_cache = np.copy(macro_pos)
        macro_pos = macro_pos[:num_items]
        macro_size = macro_size[:num_items]
        macro_fixed = macro_fixed[:num_items]

    num_macros = macro_pos.shape[0]
    if edge_type is None:
        edge_type, _, _ = longest_path_refinement(macro_pos, macro_size, macro_fixed, die_info, die_ll, die_ur,
                                                  logger, naive=naive, prune=prune)
    logger.debug("Finish Graph X construction.")
    
    prob_x = pl.LpProblem("MacroLegalizationX", pl.LpMinimize)
    prob_y = pl.LpProblem("MacroLegalizationY", pl.LpMinimize)
    x_set, y_set, dx_set, dy_set = basic_variable(macro_pos, macro_size, macro_fixed, die_info)
    prob_x += (
        pl.lpSum([macro_weights[i, 0] * dx_set[i] for i in range(num_macros)]),
        "Sum_of_Total_displacement",
    )
    prob_y += (
        pl.lpSum([macro_weights[i, 1] * dy_set[i] for i in range(num_macros)]),
        "Sum_of_Total_displacement",
    )
    for i in range(num_macros):
        ori_x = macro_pos[i][0]
        ori_y = macro_pos[i][1]
        prob_x += (
            x_set[i] - ori_x <= dx_set[i],
            "Displacement_x_%d" % i,
        )
        prob_x += (
            x_set[i] - ori_x >= -dx_set[i],
            "NegDisplacement_x_%d" % i,
        )
        prob_y += (
            y_set[i] - ori_y <= dy_set[i],
            "Displacement_y_%d" % i,
        )
        prob_y += (
            y_set[i] - ori_y >= -dy_set[i],
            "NegDisplacement_y_%d" % i,
        )

    # 2) Graph Version X:
    dist_x = (macro_size[:,0] + macro_size[:,0].reshape(-1,1)) / 2
    for i in range(num_macros):
        for j in range(num_macros):
            if edge_type[i][j] == -1 or i == j:
                continue
            if macro_fixed[i] and macro_fixed[j]:
                continue
            if edge_type[i][j] == 0:
                prob_x += (
                    x_set[i] + dist_x[i][j] <= x_set[j],
                    "Horizontal_%d_%d" % (i, j),
                )
    
    # Write LP for debugging
    # prob_x.writeLP("MacroLegalization.lp")

    # Solve by pl
    logger.debug("Start solving...")
    prob_x.solve(lpbackend)

    pl_status_x = pl.LpStatus[prob_x.status]
    solve_success = pl_status_x == "Optimal"
    displacement_x = pl.value(prob_x.objective)
    
    # Commit Solver Results
    macro_pos_new = np.copy(macro_pos)
    for v in prob_x.variables():
        if str(v.name).startswith("x_"):
            macro_pos_new[int(str(v.name).split("_")[1])][0] = float(v.varValue)
    macro_pos = macro_pos_new

    # 3) Graph Version Y: Need to update graph edges since placement is changed
    edge_type, _, _ = longest_path_refinement(macro_pos, macro_size, macro_fixed, die_info, die_ll, die_ur,
                                              logger, naive=naive, prune=prune)
    logger.debug("Finish Graph Y construction.")
    dist_y = (macro_size[:,1] + macro_size[:,1].reshape(-1,1)) / 2
    for i in range(num_macros):
        for j in range(num_macros):
            if edge_type[i][j] == -1 or i == j:
                continue
            if macro_fixed[i] and macro_fixed[j]:
                continue
            if edge_type[i][j] == 1:
                prob_y += (
                    y_set[i] + dist_y[i][j] <= y_set[j],
                    "Vertical_%d_%d" % (i, j),
                )
    
    # Write LP for debugging
    # prob_y.writeLP("MacroLegalization.lp")

    # Solve by pl
    logger.debug("Start solving...")
    prob_y.solve(lpbackend)

    pl_status_y = pl.LpStatus[prob_y.status]
    solve_success = (pl_status_y == "Optimal") and solve_success
    displacement_y = pl.value(prob_y.objective)

    # Commit Solver Results
    macro_pos_new = np.copy(macro_pos)
    for v in prob_y.variables():
        if str(v.name).startswith("y_"):
            macro_pos_new[int(str(v.name).split("_")[1])][1] = float(v.varValue)

    logger.info(
        "X Status: %s, DisplaceX = %.2f | Y Status: %s, DisplaceY = %.2f | Total Displacement = %.2f" % (
            pl_status_x, displacement_x, pl_status_y, displacement_y, displacement_x + displacement_y
    ))

    # 4) Iterative Legalization:
    if num_items is not None:
        macro_pos_cache[:num_items] = macro_pos_new[:num_items]
        logger.info("#Macros: %d, #Macros in step: %d" % (macro_pos_cache.shape[0], macro_pos_new.shape[0]))
        macro_pos_new = macro_pos_cache

    return macro_pos_new, solve_success, displacement_x + displacement_y


def macro_legalization_mix(args, logger, macro_pos, macro_size, macro_fixed, macro_weights, die_info, die_ll, die_ur,
                           num_items=None, lpbackend=None, naive=False, prune=False, edge_type=None):
    logger.info("Start macro_legalization_mix...")
    if num_items is not None:
        macro_pos_cache = np.copy(macro_pos)
        macro_pos = macro_pos[:num_items]
        macro_size = macro_size[:num_items]
        macro_fixed = macro_fixed[:num_items]

    num_macros = macro_pos.shape[0]
    if edge_type is None:
        edge_type, _, _ = longest_path_refinement(macro_pos, macro_size, macro_fixed, die_info,
                                                  die_ll, die_ur, logger, naive=naive, prune=prune)
    logger.debug("Finish Graph construction.")

    prob = pl.LpProblem("MacroLegalization", pl.LpMinimize)
    logger.debug("Setup lp variables")
    x_set, y_set, dx_set, dy_set = basic_variable(macro_pos, macro_size, macro_fixed, die_info)
    logger.debug("Setup lp objectives")
    prob += (
        pl.lpSum([macro_weights[i, 0] * dx_set[i] + macro_weights[i, 1] * dy_set[i] for i in range(num_macros)]),
        "Sum_of_Total_displacement",
    )
    logger.debug("Setup lp displacement constrains")
    for i in range(num_macros):
        ori_x = macro_pos[i][0]
        ori_y = macro_pos[i][1]
        prob += (
            x_set[i] - ori_x <= dx_set[i],
            "Displacement_x_%d" % i,
        )
        prob += (
            x_set[i] - ori_x >= -dx_set[i],
            "NegDisplacement_x_%d" % i,
        )
        prob += (
            y_set[i] - ori_y <= dy_set[i],
            "Displacement_y_%d" % i,
        )
        prob += (
            y_set[i] - ori_y >= -dy_set[i],
            "NegDisplacement_y_%d" % i,
        )
    logger.debug("Setup lp edge constraints")
    dist_x = (macro_size[:,0] + macro_size[:,0].reshape(-1,1)) / 2
    dist_y = (macro_size[:,1] + macro_size[:,1].reshape(-1,1)) / 2
    for i in range(num_macros):
        for j in range(num_macros):
            if edge_type[i][j] == -1 or i == j:
                continue
            if macro_fixed[i] and macro_fixed[j]:
                continue
            if edge_type[i][j] == 0:
                prob += (
                    x_set[i] + dist_x[i][j] <= x_set[j],
                    "Horizontal_%d_%d" % (i, j),
                )
            elif edge_type[i][j] == 1:
                prob += (
                    y_set[i] + dist_y[i][j] <= y_set[j],
                    "Vertical_%d_%d" % (i, j),
                )
    # Write LP for debugging
    # prob.writeLP("MacroLegalization.lp")

    # Solve by pl
    logger.debug("Start solving...")
    prob.solve(lpbackend)

    solve_success = pl.LpStatus[prob.status] == "Optimal"
    displacement = pl.value(prob.objective)
    logger.info("Status: %s, Total Displacement of MacroLegalization = %.2f" % (pl.LpStatus[prob.status], displacement))
    
    # Commit Solver Results
    macro_pos_new = np.copy(macro_pos)
    for v in prob.variables():
        if str(v.name).startswith("x_"):
            macro_pos_new[int(str(v.name).split("_")[1])][0] = float(v.varValue)
        if str(v.name).startswith("y_"):
            macro_pos_new[int(str(v.name).split("_")[1])][1] = float(v.varValue)

    if num_items is not None:
        macro_pos_cache[:num_items] = macro_pos_new[:num_items]
        logger.info("#Macros: %d, #Macros in step: %d" % (macro_pos_cache.shape[0], macro_pos_new.shape[0]))
        macro_pos_new = macro_pos_cache

    return macro_pos_new, solve_success, displacement


def macro_legalization_ilp(args, logger, macro_pos, macro_size, macro_fixed, macro_weights, die_info, die_ll, die_ur,
                           num_items=None, lpbackend=None, edge_type=None):
    logger.info("Start macro_legalization_ilp...")
    if num_items is not None:
        macro_pos_cache = np.copy(macro_pos)
        macro_pos = macro_pos[:num_items]
        macro_size = macro_size[:num_items]
        macro_fixed = macro_fixed[:num_items]
    num_macros = macro_pos.shape[0]

    prob = pl.LpProblem("MacroLegalization", pl.LpMinimize)
    die_lx, die_hx, die_ly, die_hy = die_info
    x_set, y_set, dx_set, dy_set = basic_variable(macro_pos, macro_size, macro_fixed, die_info)
    prob += (
        pl.lpSum([macro_weights[i, 0] * dx_set[i] + macro_weights[i, 1] * dy_set[i] for i in range(num_macros)]),
        "Sum_of_Total_displacement",
    )
    for i in range(num_macros):
        ori_x = macro_pos[i][0]
        ori_y = macro_pos[i][1]
        prob += (
            x_set[i] - ori_x <= dx_set[i],
            "Displacement_x_%d" % i,
        )
        prob += (
            x_set[i] - ori_x >= -dx_set[i],
            "NegDisplacement_x_%d" % i,
        )
        prob += (
            y_set[i] - ori_y <= dy_set[i],
            "Displacement_y_%d" % i,
        )
        prob += (
            y_set[i] - ori_y >= -dy_set[i],
            "NegDisplacement_y_%d" % i,
        )

    # Naive Binary Variables Version:
    dist_x = (macro_size[:,0] + macro_size[:,0].reshape(-1,1)) / 2
    dist_y = (macro_size[:,1] + macro_size[:,1].reshape(-1,1)) / 2
    choices = pl.LpVariable.dicts("Choice", (range(num_macros), range(num_macros), range(2)), 0, 1, cat=pl.const.LpInteger)
    for i in range(num_macros):
        for j in range(num_macros):
            if i >= j:
                continue
            if macro_fixed[i] and macro_fixed[j]:
                continue
            prob += (
                x_set[i] + dist_x[i][j] <= x_set[j] + die_hx * (choices[i][j][0] + choices[i][j][1]),
                "XLhs_%d_%d" % (i, j),
            )
            prob += (
                x_set[i] - dist_x[i][j] >= x_set[j] - die_hx * (1 + choices[i][j][0] - choices[i][j][1]),
                "XRhs_%d_%d" % (i, j),
            )
            prob += (
                y_set[i] + dist_y[i][j] <= y_set[j] + die_hy * (1 - choices[i][j][0] + choices[i][j][1]),
                "YLhs_%d_%d" % (i, j),
            )
            prob += (
                y_set[i] - dist_y[i][j] >= y_set[j] - die_hy * (2 - choices[i][j][0] - choices[i][j][1]),
                "YRhs_%d_%d" % (i, j),
            )

    # Write LP for debugging
    # prob.writeLP("MacroLegalization.lp")

    # Solve by pl
    logger.debug("Start solving...")
    prob.solve(lpbackend)

    solve_success = pl.LpStatus[prob.status] == "Optimal"
    displacement = pl.value(prob.objective)
    logger.info("Status: %s, Total Displacement of MacroLegalization = %.2f" % (pl.LpStatus[prob.status], displacement))
    
    # Commit Solver Results
    macro_pos_new = np.copy(macro_pos)
    for v in prob.variables():
        if str(v.name).startswith("x_"):
            macro_pos_new[int(str(v.name).split("_")[1])][0] = float(v.varValue)
        if str(v.name).startswith("y_"):
            macro_pos_new[int(str(v.name).split("_")[1])][1] = float(v.varValue)

    if num_items is not None:
        macro_pos_cache[:num_items] = macro_pos_new[:num_items]
        logger.info("#Macros: %d, #Macros in step: %d" % (macro_pos_cache.shape[0], macro_pos_new.shape[0]))
        macro_pos_new = macro_pos_cache

    return macro_pos_new, solve_success, displacement


def macro_rough_align(macro_pos, macro_size, macro_fixed, die_ll, die_ur, inv_scalar):
    is_mov = np.logical_not(macro_fixed)

    macro_pos_lb = macro_size / 2 + die_ll + 1e-4
    macro_pos_ub = die_ur - macro_size / 2 + die_ll - 1e-4
    mov_macro_pos_lb = macro_pos_lb[is_mov]
    mov_macro_pos_ub = macro_pos_ub[is_mov]
    macro_pos[is_mov] = macro_pos[is_mov].clip(min=mov_macro_pos_lb, max=mov_macro_pos_ub)

    macro_lpos = macro_pos - macro_size / 2
    macro_lpos = np.multiply(macro_lpos, inv_scalar, out=macro_lpos)
    macro_lpos = np.round_(macro_lpos, out=macro_lpos)
    macro_lpos = np.divide(macro_lpos, inv_scalar, out=macro_lpos)

    macro_pos_ = macro_lpos + macro_size / 2
    macro_pos[is_mov] = macro_pos_[is_mov]

    return macro_pos


def plot_macros(macro_pos, macro_size, macro_fixed, die_info, given_colors=None, img_path=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    base_x = 8
    die_lx, die_hx, die_ly, die_hy = die_info
    base_y = base_x / die_hx * die_hy 
    fig, ax = plt.subplots(1, figsize=(base_x, base_y))
    die_rects = [Rectangle((die_lx, die_ly), die_hx - die_lx, die_hy - die_ly)]
    pc = PatchCollection(die_rects, facecolor='w', alpha=0.5, edgecolor='black')
    ax.add_collection(pc)
    all_color = plt.get_cmap('Set2').colors
    for i in range(macro_pos.shape[0]):
        x, y = macro_pos[i]
        w, h = macro_size[i]
        color_idx = given_colors[i] if given_colors is not None else 0
        color = all_color[color_idx]
        rect = Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, alpha=0.5, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, "%d" % i, fontsize=14)

    plt.xlim(-die_hx * 0.02, die_hx * 1.02)
    plt.ylim(-die_hy * 0.02, die_hy * 1.02)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    img_path = img_path if img_path is not None else "test.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()


def plot_negative_slack_macro(macro_pos, macro_size, macro_fixed, die_info, slack_v, img_path=None):
    num_macros = slack_v.shape[0]
    x_nslack = np.minimum(slack_v[:, 0], 0)
    y_nslack = np.minimum(slack_v[:, 1], 0)
    print(x_nslack)
    print(y_nslack)
    unique_values = np.sort(np.unique(x_nslack + y_nslack))[::-1]
    given_colors = np.zeros(num_macros, dtype=np.int8)
    for color_idx, value in enumerate(unique_values):
        given_colors[(x_nslack + y_nslack) == value] = color_idx
    plot_macros(macro_pos, macro_size, macro_fixed, die_info, given_colors=given_colors, img_path=img_path)


def macro_legalization_multi(macro_info, args, logger):
    macro_pos, macro_size, macro_fixed, macro_weights, die_ll, die_ur, die_info, inv_scalar = macro_info
    macro_pos = macro_pos.cpu().numpy()
    macro_size = macro_size.cpu().numpy()
    macro_fixed = macro_fixed.cpu().numpy()
    macro_weights = macro_weights.cpu().numpy()
    die_ll = die_ll.cpu().numpy()
    die_ur = die_ur.cpu().numpy()
    die_info = die_info.cpu().numpy()
    inv_scalar = inv_scalar.cpu().numpy()

    # plot_macros(macro_pos, macro_size, macro_fixed, die_info, img_path="legalized_before.png")
    nb.set_num_threads(args.num_threads)

    if check_macro_legality(macro_pos, macro_size, macro_fixed, die_info, check_all=False)[0]:
        # Macros are legal, skip legalization
        return macro_pos, True

    macro_pos = macro_rough_align(macro_pos, macro_size, macro_fixed, die_ll, die_ur, inv_scalar)

    def macro_lg_handler(
        ml_func, method_name, solver, best_result, *func_args, max_times=1, timeLimit=None, **ml_func_kwargs
    ):
        total_displacement = 0
        for i in range(max_times):
            solver.timeLimit = (i + 1) * 20 if timeLimit is None else timeLimit
            logger.info("Use cbc to solve LP. TimeLimit = %ds." % solver.timeLimit)
            macro_pos_tmp, solve_success, displacement = ml_func(*func_args, lpbackend=solver, **ml_func_kwargs)
            legal, overlaps = check_macro_legality(macro_pos_tmp, macro_size, macro_fixed, die_info)
            if not legal:
                # update macro_pos to macro_pos_tmp
                logger.warning("Current solution is illegal. Macro Overlap Pairs: %s." % overlaps)
                func_args = (*func_args[:2], macro_pos_tmp, *func_args[3:])
                ml_func_kwargs["edge_type"] = None
                solve_success = False
            total_displacement += displacement
            if solve_success:
                break
        if not solve_success and not best_result[1] and method_name != "ilp":
            best_result = (macro_pos_tmp, False, total_displacement, method_name)
        if solve_success and (not best_result[1] or total_displacement < best_result[2]):
            best_result = (macro_pos_tmp, True, total_displacement, method_name)
        return best_result

    # best_result: (macro_pos, solve_success, displacement, method_name)
    best_result = (None, False, float('inf'), None)
    func_args = (args, logger, macro_pos, macro_size, macro_fixed, macro_weights, die_info, die_ll, die_ur)
    solver = pl.PULP_CBC_CMD(msg=0, timeLimit=20, threads=args.num_threads)
    if macro_pos.shape[0] > 500:
        logger.info("Too many macros, try pruned graph version first.")
        edge_type, _, _ = longest_path_refinement(macro_pos, macro_size, macro_fixed, die_info, die_ll, die_ur,
                                                logger, naive=False, prune=True)
        best_result = macro_lg_handler(macro_legalization_mix, "mix", solver, best_result, *func_args, max_times=1, edge_type=edge_type, prune=True)
        best_result = macro_lg_handler(macro_legalization_xy, "xy", solver, best_result, *func_args, max_times=3, edge_type=edge_type, prune=True)

    if not best_result[1]:
        edge_type, _, _ = longest_path_refinement(macro_pos, macro_size, macro_fixed, die_info, die_ll, die_ur,
                                                logger, naive=False, prune=False)
        best_result = macro_lg_handler(macro_legalization_mix, "mix", solver, best_result, *func_args, max_times=1, edge_type=edge_type)
        best_result = macro_lg_handler(macro_legalization_xy, "xy", solver, best_result, *func_args, max_times=3, edge_type=edge_type)

    if not best_result[1]:
        logger.warning("Both LPs are infeasible. Try ILP version.")
        macro_lg_handler(macro_legalization_ilp, "ilp", solver, best_result, *func_args, timeLimit=120)
        if not best_result[1]:
            logger.error("ILP is infeasible.")

    # commit the result no matter it is legal or not
    macro_pos, solve_success, displacement, method_name = best_result
    # plot_macros(macro_pos, macro_size, macro_fixed, die_info, img_path="legalized_after.png")
    if solve_success:
        logger.info("Macro Legalization Success. Select macro legalization [%s]. Displacement = %.2f" % (method_name, displacement))

    return macro_pos, solve_success


# For debug only
# numba_logger = logging.getLogger('numba')
# numba_logger.setLevel(logging.WARNING)
# import random
# import torch
# import time
# random.seed(0)
# class Args:
#     def __init__(self) -> None:
#         self.num_threads = 20
# args = Args()
# logger = logging.getLogger(__name__)
# logging.basicConfig(encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
# if __name__ == "__main__":
#     macro_info = torch.load("macro_info.pt")
#     logger.info("Start...")
#     start_time = time.time()
#     macro_pos, solve_success = macro_legalization_multi(macro_info, args, logger)
#     logger.info("Macro legalization time: %.2f" % (time.time() - start_time))
