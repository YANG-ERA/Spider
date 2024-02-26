import warnings
import sys
import os
#from .joint_simulator import Simulator, get_gene_prior, get_nf_prior
#from .opt import valid_neighbourhood_frequency
from os.path import join
import time
from .sim_expr import *
from .random_based_utils import *

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_sim_spot_level_expr( spot_num = None, spot_row = None, spot_col = None,
                            Num_sample = None, spot_diameter = None, image_width = None,
                            image_height = None,celltype_assignment = None,coord_type = None,
                            cell_spatial = None,sim_cell_expr = None,gap = 0,
                           spot_generate_type = "square",cell_coord_type = "grid",
                            custom_spot_loc=None, grid_row = None, grid_col = None):
    if spot_generate_type == "square" and cell_coord_type == "generic":
        ## 随机点的网格划分，不是网格merge 

        grid_row = np.ceil(image_width / spot_diameter).astype(int)
        grid_col = np.ceil(image_height / spot_diameter).astype(int)
        grid_num = grid_row * grid_col
        points_x_indices = (cell_spatial[:, 0] / spot_diameter).astype(int)
        points_y_indices = (cell_spatial[:, 1] / spot_diameter).astype(int)

        ## 细胞的spot编号
        cell_spot_id = (points_x_indices*grid_col + points_y_indices).astype(int)
        ## 细胞的编号
        cell_id = np.arange(cell_spot_id.size)


        ##  替换np.where 每一个grid  obtain 每一个spot的cell的index 行spot 列cell_id
        spot_cell_idx_matrix = csr_matrix(
            (np.ones(cell_spot_id.size), (cell_spot_id, cell_id)),
            shape=(grid_num, cell_spot_id.size))

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        ## 注意有些spot里面可能没有cell
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        ## 取网格的中心点作为spot坐标
        a, b = np.meshgrid(np.arange(grid_row),
                       np.arange(grid_col),
                       indexing='ij')
        spot_pixel_row = (a.reshape(-1)+0.5)*spot_diameter
        spot_pixel_col = (b.reshape(-1)+0.5)*spot_diameter

        spot_loc =  np.vstack((spot_pixel_row, spot_pixel_col)).T

        return spot_expr, spot_loc, spot_cell_idx_matrix, spot_ct_count

    if spot_generate_type == "circle" and cell_coord_type == "generic":

     ####### 规定直径 and gap

        grid_row = int(image_width / gap)
        grid_col = int(image_height / gap)
        grid_num = grid_row * grid_col

        ## 取网格的中心点作为spot坐标
        a, b = np.meshgrid(np.arange(grid_row),
                       np.arange(grid_col),
                       indexing='ij')
        spot_pixel_row = (a.reshape(-1)+0.5)*gap
        spot_pixel_col = (b.reshape(-1)+0.5)*gap

        spot_loc =  np.vstack((spot_pixel_row, spot_pixel_col)).T

        cellandspot_loc =  np.vstack((cell_spatial,spot_loc))

        ## get中心点的邻居
        sn = get_spaital_network(Num_sample=cellandspot_loc.shape[0],spatial=cellandspot_loc,
                         coord_type = "generic",n_rings = 1,set_diag=False,
                                 radius = spot_diameter/2)

        spot_cell_idx_matrix = sn[Num_sample:,Num_sample]

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        ## 注意有些spot里面可能没有cell
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        return spot_expr, spot_loc, spot_cell_idx_matrix

    if spot_generate_type == "square" and cell_coord_type == "grid":

      ##### 还有网格merge
        if grid_row is None and grid_col is None:
            grid_row = np.sqrt(Num_sample).astype(int)
            grid_col = grid_row
        ## 网格merge,我们取九个格子merge 类似ST 同时设置spot之间的gap
        windows_row = 3
        windows_col = 3
        gap = gap
        spot_diameter = 1

        ## 边缘没有3*3的网格,设不设置gap?
        spot_row_num = int(grid_row/(windows_row+gap))
        spot_col_num = int(grid_col/(windows_col+gap))

        a, b = np.meshgrid(np.arange(grid_row),
                   np.arange(grid_col),
                   indexing='ij')
        ## lra 与lrb是9个网格的中心网格 实际是中心网格左下角的点的坐标
        lra = a[np.arange(1, spot_row_num*(windows_row+gap), (windows_row+gap)
                  ), :len(np.arange(1, spot_col_num*(windows_col+gap), (windows_col+gap)))].reshape(-1)
        lrb = b[:len(np.arange(1, spot_row_num*(windows_row+gap), (windows_row+gap))),
            np.arange(1, spot_col_num*(windows_col+gap), (windows_col+gap))].reshape(-1)

        ## 中心网格标号
        lrspot_layer_idx = lra*grid_col+lrb

        ## get中心点的邻居，即9个网格 包括自己，
        sn = get_spaital_network(Num_sample=cell_spatial.shape[0],spatial=cell_spatial,
                         n_neighs=8,coord_type = "grid",n_rings = 1,set_diag=True)

        spot_cell_idx_matrix = sn[lrspot_layer_idx]
        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        ## 注意有些spot里面可能没有cell
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        spot_pixel_row = (lra.reshape(-1)+0.5)*spot_diameter
        spot_pixel_col = (lrb.reshape(-1)+0.5)*spot_diameter
        spot_loc = np.vstack((spot_pixel_row, spot_pixel_col)).T


        return spot_expr, spot_loc, spot_cell_idx_matrix

    if custom_spot_loc is not None:
        ## 以及指定spot坐标set  

        ## must numpy shape(n*2)
        ## custom_spot_loc and spot_diameter
        spot_loc = custom_spot_loc

        cellandspot_loc =  np.vstack((cell_spatial,spot_loc ))
        sn = get_spaital_network(Num_sample=cellandspot_loc.shape[0],spatial=cellandspot_loc,
                         coord_type = "generic",n_rings = 1,set_diag=False,
                                 radius = spot_diameter/2)

        spot_cell_idx_matrix = sn[cell_spatial.shape[0]:,cell_spatial.shape[0]]

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        ## 注意有些spot里面可能没有cell
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

    return spot_expr, spot_loc, spot_cell_idx_matrix


# def rctd(adata=None, ctkey=None, method=None, file_path=None, seed=123, ):
#     W, exp_spots2gene, spatial, spots2cell = RCTD_naive(scdata=adata,
#                                                         row=row_col_size,
#                                                         col=row_col_size,
#                                                         Min=0,
#                                                         Max=1,
#                                                         ctkey=ctkey,
#                                                         maxctnum=len(adata.obs.celltype.unique()))
#     exp_spots2gene = downsample_matrix_by_cell(exp_spots2gene, 10000)
#     simulatedat = sc.AnnData(exp_spots2gene[:len(adata), :])
#     simulatedat.obs = pd.DataFrame(
#         np.array(W[:len(adata), :] @ range(len(adata.obs.celltype.unique())), dtype=int))


def sim_naive_cell(use_real_adata=None,
              ctkey=None,method=None,
              file_path=None,seed = 123):
    global simulatedat
    np.random.seed(seed)
    out_dir = file_path+'newsimu/'
    row_col_size = round(np.sqrt(len(use_real_adata)))+1
    #choose method
    if method == 'RCTD':
        W,exp_spots2gene,spatial,spots2cell = RCTD_naive(scdata=use_real_adata,
                                                     row=row_col_size,
                                                     col=row_col_size,
                                                     Min = 0,
                                                     Max = 1,
                                                     ctkey=ctkey,
                                                     maxctnum = len(use_real_adata.obs.celltype.unique()))
        exp_spots2gene = downsample_matrix_by_cell(exp_spots2gene,10000)
        simulatedat = sc.AnnData(exp_spots2gene[:len(use_real_adata),:])
        simulatedat.obs = pd.DataFrame(np.array(W[:len(use_real_adata),:] @ range(len(use_real_adata.obs.celltype.unique())),dtype = int))

    if method == 'STRIDE':
        W,exp_spots2gene,spatial = STRIDE_naive(use_real_adata, ctkey, row_col_size*row_col_size, min_cells=1, max_cells=2)
        exp_spots2gene = downsample_matrix_by_cell(exp_spots2gene,10000)
        simulatedat = sc.AnnData(exp_spots2gene[:len(use_real_adata),:])
        simulatedat.obs = pd.DataFrame(np.array(W[:len(use_real_adata),:] @ range(len(use_real_adata.obs.celltype.unique())),dtype = int))

    if method == 'sterepscope':
        ct = use_real_adata.obs.celltype.unique()
        use_exp = pd.DataFrame(use_real_adata.X,columns=use_real_adata.var.gene)
        use_exp.to_csv(file_path+str(len(ct))+'ct_real_exp.csv')
        use_label = use_real_adata.obs.celltype.to_frame()
        use_label.to_csv(file_path+str(len(ct))+'ct_real_celltype.csv')
        sc_cnt_pth = file_path+str(len(use_real_adata.obs.celltype.unique()))+'ct_real_exp.csv'
        sc_lbl_pth = file_path+str(len(use_real_adata.obs.celltype.unique()))+'ct_real_celltype.csv'
        stereoscope_naive(sc_cnt_pth = sc_cnt_pth,
                      sc_lbl_pth = sc_lbl_pth,
                      out_dir = out_dir,
                      data = use_real_adata)
        data1 = pd.read_csv(out_dir+'counts.st_synth.tsv',index_col = 0)
        data1 = data1.sort_index(axis=1)
        menber = pd.read_csv(out_dir+'members.st_synth.tsv',index_col = 0)
        simulatedat = sc.AnnData(data1)
        simulatedat.obs = pd.DataFrame(np.array(menber @ range(len(use_real_adata.obs.celltype.unique())),dtype = int))

    #save
    if np.isin(method,['RCTD','STRIDE']):
        simulatedat.obs.columns = ['label']
        simulatedat.obs.label = simulatedat.obs.label.astype(int).astype('category')
        simulatedat.obs['celltype'] = use_real_adata.obs[ctkey]
        for j in range(len(use_real_adata.obs[ctkey].unique())):
            simulatedat.obs['celltype'][np.where(simulatedat.obs['label'] == j)[0]] = np.unique(use_real_adata.obs[ctkey])[j]
        simulatedat.obs.index = ['cell'+str(i+1) for i in range(use_real_adata.shape[0])]

        simulatedat.obsm = use_real_adata.obsm  #real location
        simulatedat.uns['W'] = W[:len(use_real_adata),:]
        simulatedat.uns['celltype_name'] = np.unique(use_real_adata.obs[ctkey])
        simulatedat.var = use_real_adata.var
        # if not os.path.exists(file_path+'newsimu/'):
        #     os.mkdir(file_path+'newsimu/')
        simulatedat.write(file_path+method+'_simu_cell_level'+'.h5ad')

    return simulatedat

def sim_naive_spot(use_real_adata=None,level='cell',spot_diameter=500,
                   ctkey=None,method=None,file_path=None,seed = 123):
    global sim_spot_adata
    sim_adata = sim_naive_cell(use_real_adata,ctkey,method,file_path,seed)

    if level == "cell":
        return sim_adata
    elif level == 'spot':
        ## transform coordinate
        temp1 = np.min(sim_adata.obsm["spatial"][:,0])
        temp2 = np.min(sim_adata.obsm["spatial"][:,1])
        sim_adata.obsm["spatial"][:,0] = sim_adata.obsm["spatial"][:,0]- temp1
        sim_adata.obsm["spatial"][:,1] = sim_adata.obsm["spatial"][:,1]- temp2
        spot_expr, spot_loc, spot_cell_idx_matrix,spot_ct_count  = get_sim_spot_level_expr(spot_num=None,
                    spot_row=None,
                    spot_col=None,
                    Num_sample=sim_adata.shape[0],
                    spot_diameter=spot_diameter,
                    image_width=np.max(sim_adata.obsm["spatial"][:,0]),
                    image_height=np.max(sim_adata.obsm["spatial"][:,1]),
                    celltype_assignment=sim_adata.obs.label,
                    coord_type="generic",
                    cell_spatial=sim_adata.obsm["spatial"],
                    sim_cell_expr=sim_adata,
                    gap=0,
                    spot_generate_type="square",
                    cell_coord_type="generic",
                    grid_row=None,
                    grid_col=None)
        mask = np.all(spot_ct_count == 0, axis=1)
        ## adjust spot_loc
        spot_loc[:,0] = spot_loc[:,0] + temp1
        spot_loc[:,1] = spot_loc[:,1] + temp2
        sim_spot_adata = ad.AnnData(spot_expr)
        sim_spot_adata.obsm["spatial"] = spot_loc
        sim_spot_adata.var = sim_adata.var
        sim_spot_adata.obs.index =  ['spot'+str(j) for j in range(spot_expr.shape[0])]
        sim_spot_adata = sim_spot_adata[~mask]
        sim_spot_adata.uns["W"] = spot_ct_count[~mask]
        sim_spot_adata.write_h5ad(file_path + method +"_simu_spot_level" + ".h5ad")
        return sim_spot_adata
    else:
        raise ValueError("The parameter level must be either 'cell' or 'spot'")

#generate gene expression using splatter
from .scsim import scsim
"""
This part is modified code from 'Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq'
from this repository https://github.com/dylkot/scsim
Modified version allows the simulator take a pre-defined cell group assignment.
"""
def sim_naive_spot_splatter(use_real_adata=None,level='cell',spot_diameter=500,
                   ctkey=None,method=None,file_path=None,seed = 123, doubletfrac = 0,
                    deloc = 1, progdeloc = 1, descale = 1.0, progcellfrac = .35,
                    deprob = .025, nproggenes = 400):
    adata_simu = sim_naive_spot(use_real_adata, level, spot_diameter,
                                   ctkey, method, file_path, seed)
    #gen exp using splatter
    ncells = adata_simu.shape[0]
    ngenes = adata_simu.shape[1]
    K = len(adata_simu.obs.celltype.unique())
    groupid = np.array(adata_simu.obs.label.values)
    ndoublets = int(doubletfrac*ncells)
    nproggroups = K
    proggroups = list(range(1, nproggroups+1))
    simulator = scsim(ngenes=ngenes, ncells=ncells, ngroups=K, groupid = groupid, libloc=7.64, libscale=0.78,
                         mean_rate=7.68,mean_shape=0.34, expoutprob=0.00286,
                         expoutloc=6.15, expoutscale=0.49,
                         diffexpprob=deprob, diffexpdownprob=0., diffexploc=deloc, diffexpscale=descale,
                         bcv_dispersion=0.448, bcv_dof=22.087, ndoublets=ndoublets,
                         nproggenes=nproggenes, progdownprob=0., progdeloc=progdeloc,
                         progdescale=descale, progcellfrac=progcellfrac, proggoups=proggroups,
                         minprogusage=.1, maxprogusage=.7, seed=seed)
    start_time = time.time()
    simulator.simulate()
    end_time = time.time()
    print("Elapsing time is %.2f"%(end_time - start_time))
    gene_expression_new = np.asarray(simulator.counts)
    adata_simu.X = gene_expression_new
    return adata_simu