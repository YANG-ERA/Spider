import warnings
import sys
import os
from .joint_simulator import Simulator, get_gene_prior, get_nf_prior
from .opt import valid_neighbourhood_frequency
from os.path import join
from .run_scsim import *
from .sim_expr import *

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#run fict func
def plot_freq(neighbour,axes,color,cell_tag):
    sample_n = neighbour.shape[1]
    neighbour = neighbour/np.sum(neighbour,axis = 1,keepdims = True)
    std = np.std(neighbour, axis = 0)/np.sqrt(sample_n)
    mean = np.mean(neighbour, axis = 0)
    x = np.arange(sample_n)
    yerror = np.asarray([-std,std])

    patches = axes.boxplot(neighbour,
                        vert=True,  # vertical box alignment
                        patch_artist=True,
                        notch=True,
                        usermedians = mean) # fill with color
    for patch in patches['boxes']:
        patch.set_facecolor(color)
        patch.set_color(color)
        patch.set_alpha(0.5)
    for patch in patches['fliers']:
        patch.set_markeredgecolor(color)
        patch.set_color(color)
    for patch in patches['whiskers']:
        patch.set_color(color)
    for patch in patches['caps']:
        patch.set_color(color)
    axes.errorbar(x+1,mean,color = color,label = cell_tag)
    return mean,yerror

def simulation(sim_folder,  #path
               sample_n = 2500,
               n_g = 1000,
               n_c = 3,
               density = 20,
               threshold_distance = 1,
               using_splatter = False,
               method = 4,
               use_refrence_coordinate = False,
               coor = None,
               cell_types = None,
               gene_mean = None,
               gene_std = None,
               neighbour_freq_prior = None,
               tags = None,
               type_count = None,
               random_seed = None
                        ):
    if not os.path.isdir(sim_folder):
        os.mkdir(sim_folder)
    methods = ['real_tran','real_tran1','stripe','real']
    o_f = join(sim_folder,"%s"%(methods[method]))
    if not os.path.isdir(o_f):
        os.mkdir(o_f)
    if not os.path.isdir(join(o_f,"figures")):
        os.mkdir(join(o_f,"figures"))
    if (method == 3) | use_refrence_coordinate:
        if n_c == 3:
            type_gather = ['Inhibitory','Excitatory','Ependymal']
        else:
            type_gather = None
    print("######## Begin simulation with %s configuration ########"%(methods[method]))
    #changed
    def real_tran(n_c):
        target_freq = np.ones((n_c,n_c))
        for i in np.arange(n_c):
            target_freq[i,i] = 4*(n_c-1)
        #新加,改成real
        target_freq = neighbour_freq_prior
        target_freq/np.sum(target_freq,axis=1,keepdims=True)
        return valid_neighbourhood_frequency(target_freq)[0]

    def real_tran1(n_c):
        target_freq = neighbour_freq_prior
        return target_freq

    def stripe_freq(n_c):
        target_freq = np.ones((n_c,n_c))
        for i in np.arange(n_c):
            target_freq[i,i] = 3*(n_c-1)
            if i>0:
                target_freq[i-1,i] = 3*(n_c-1)
        target_freq/np.sum(target_freq,axis=1,keepdims=True)
        return valid_neighbourhood_frequency(target_freq)[0]
    def real_freq(n_c):
        assert len(neighbour_freq_prior) == n_c
        target_freq = np.asarray(neighbour_freq_prior)
        target_freq/np.sum(target_freq,axis=1,keepdims=True)
        return valid_neighbourhood_frequency(target_freq)[0]

    freq_map = {0:real_tran,1:real_tran1,2:stripe_freq,3:real_freq} #changed
    target_freq = freq_map[method](n_c)
    #检测target_freq中是否有小于0的项
    target_freq[np.where(target_freq<0)] = 1e-6
    target_freq[np.where(target_freq.sum(axis = 1)>1)]/target_freq[np.where(target_freq.sum(axis = 1)>1)].sum()

    sim = Simulator(sample_n,n_g,n_c,density,seed = random_seed)
    sim.gen_parameters(gene_mean_prior = None)
    if use_refrence_coordinate:
        reference_coordinate = coor
    else:
        reference_coordinate = None
    cell_idxs = sim.gen_coordinate(density = density,
                                   ref_coor = reference_coordinate)
    cell_types = cell_types[cell_idxs]
    ### Assign cell types by Gibbs sampling and load
    if method == 3:
        print("Assign cell types using refernece.")
        sim.assign_cell_type(target_neighbourhood_frequency = target_freq,
                             method = "Direct-assignment",
                             ref_assignment = cell_types.astype(int))
    else:
        print("Assign cell type using Gibbs sampling.")
        sim.assign_cell_type(target_neighbourhood_frequency=target_freq,
                             method = "Gibbs-sampling",
                             max_iter = 500,
                             use_exist_assignment = False)
        print("Refine cell type using Metropolisâ€“Hastings algorithm.")
        sim.assign_cell_type(target_neighbourhood_frequency=target_freq,
                             method = "Metropolis-swap",
                             max_iter = 30000,
                             use_exist_assignment = True,
                             annealing = True)  #changed
    fig,axs = plt.subplots()
    axs.scatter(sim.coor[:,0],sim.coor[:,1], c = sim.cell_type_assignment,s = 20)
    axs.set_title("Cell type assignment after assign_neighbour")
    axs.set_xlabel("X")
    axs.set_ylabel("Y")
    fig.savefig(join(o_f,"figures/Cell_location.png"))

    sim._get_neighbourhood_frequency()

    if using_splatter:
        print("Generate gene expression using Splatter.")
        sim_gene_expression,sim_cell_type,sim_cell_neighbour = sim.gen_expression_splatter()
    else:
        print("Generate gene expression.")
        sim_gene_expression,sim_cell_type,sim_cell_neighbour = sim.gen_expression(drop_rate = None)

    ### Save the simulator to the file
    print("Saving...")
    np.savetxt(join(o_f,"simulator_gene.csv"),sim_gene_expression,delimiter = ",")
    #new
    np.savetxt(join(o_f,"simulator_celltype.csv"),sim_cell_type,delimiter = ",")
    return cell_idxs   #changed

def get_trans(adata = None, ct = None):
    sn = get_spaital_network(Num_sample=adata.obs.shape[0],
                         spatial=adata.obsm["spatial"], coord_type = "generic",
                         n_neighs=8)
    onehot_ct = get_onehot_ct(init_assign=ct)
    nb_count = np.array(sn * onehot_ct, dtype=np.float32)
    target_trans = get_nb_freq(nb_count=nb_count, onehot_ct=onehot_ct)
    return target_trans

def get_onehot_ct(init_assign = None):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(init_assign)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_ct = onehot_encoder.fit_transform(integer_encoded)
    return onehot_ct.astype(np.float32)
def get_sim_spot_level_expr( spot_num = None, spot_row = None, spot_col = None,
                            Num_sample = None, spot_diameter = None, image_width = None,
                            image_height = None,celltype_assignment = None,coord_type = None,
                            cell_spatial = None,sim_cell_expr = None,gap = 0,
                           spot_generate_type = "square",cell_coord_type = "grid",
                           grid_row = None, grid_col = None):
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

        cellandspot_loc =  np.vstack((cell_location,spot_loc))

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
        spot_loc =  np.vstack((spot_pixel_row, spot_pixel_col)).T


        return spot_expr, spot_loc, spot_cell_idx_matrix

    if custom_spot_loc is not None:
        ## 以及指定spot坐标set  

        ## must numpy shape(n*2)
        ## custom_spot_loc and spot_diameter
        spot_loc = custom_spot_loc

        cellandspot_loc =  np.vstack((cell_location,spot_loc ))
        sn = get_spaital_network(Num_sample=cellandspot_loc.shape[0],spatial=cellandspot_loc,
                         coord_type = "generic",n_rings = 1,set_diag=False,
                                 radius = spot_diameter/2)

        spot_cell_idx_matrix = sn[cell_location.shape[0]:,cell_location.shape[0]]

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        ## 注意有些spot里面可能没有cell
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

    return spot_expr, spot_loc, spot_cell_idx_matrix

def sim_naive_cell(use_real_adata=None,
              ctkey=None,method=None,
              file_path=None,seed = 123):
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


    if method == 'fict':
        adata = use_real_adata.copy()
        gene_mean,gene_std = get_gene_prior(adata.to_df(),cell_types)
        neighbour_freq_prior,tags,type_count = get_nf_prior(coor,cell_types)
        cell_types_tamp = np.zeros(len(cell_types))
        real_target_trans = get_trans(adata = adata , ct = adata.obs.label)
        for i_fict in range(len(np.unique(cell_types))):
            cell_types_tamp[np.where(cell_types == np.unique(cell_types)[i_fict])[0]] = i_fict
        cell_types = cell_types_tamp.astype(int)

        if not os.path.isdir(file_path+'/fict_simu/'):
            os.mkdir(file_path+'/fict_simu/')
        cell_idxs = simulation(file_path+'/fict_simu/'+'fict_simu_real',
               sample_n = adata.shape[0],
               density = 20,
               method = 0, #use trans matrix
               n_g = adata.shape[1],
               n_c = len(adata.obs.celltype.unique()),
               using_splatter= False,
               use_refrence_coordinate=True, #use real location
               coor = adata.obsm['spatial'].copy(),
               cell_types = np.asarray(adata.obs.celltype),
               gene_mean = gene_mean,
               gene_std = gene_std,
               neighbour_freq_prior = real_target_trans,
               tags = tags,
               type_count = type_count,
               random_seed = seed
                )
        exp = pd.read_csv(file_path+'/fict_simu/'+'fict_simu_real'+'/real_tran'+'/simulator_gene.csv',header=None)
        exp.columns = adata.var['gene'].values
        exp.index = adata.obs.index.values
        label = pd.read_csv(file_path+'/fict_simu/'+'fict_simu_real'+'/real_tran'+'/simulator_celltype.csv',header=None)
        label.columns = ['label']
        label['celltype'] = adata.obs.celltype.values
        for j in range(len(np.unique(adata.obs.celltype))):
            label.iloc[np.where(label['label'] == j)[0],1] = np.unique(adata.obs.celltype)[j]
            label['label'] = label['label'].astype(int).astype('category')
            simu_adata = sc.AnnData(exp)
            # 对fict_dist数据单独做处理，因为生成的不是counts,进行取整操作
            simu_adata.X = np.round(simu_adata.X)
            simu_adata.obs['celltype'] = label['celltype'].values
            simu_adata.obs['label'] = label['label'].values
            simu_adata.obsm['spatial'] = adata.obsm['spatial'][cell_idxs,:]
            simu_adata.write_h5ad(file_path+'/fict_simu_cell_level'+'.h5ad')
        simulatedat = simu_adata
    #save
    if np.isin(method,['RCTD','STRIDE','stereoscope']):
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
        if not os.path.exists(file_path+'newsimu/'):
            os.mkdir(file_path+'newsimu/')
        simulatedat.write(file_path+'newsimu/'+method+'simu_cell_level'+'.h5ad')

    return simulatedat

def sim_naive_spot(use_real_adata=None,level='cell',spot_diameter=500,
                   ctkey=None,method=None,file_path=None,seed = 123):
    sim_adata = sim_naive_cell(use_real_adata,ctkey,method,file_path,seed)
    adata = sim_adata.copy()
    if level!='cell':
        ## 平移坐标xy
        temp1 = np.min(simu_adata.obsm["spatial"][:,0])
        temp2 = np.min(simu_adata.obsm["spatial"][:,1])
        simu_adata.obsm["spatial"][:,0] = simu_adata.obsm["spatial"][:,0]- temp1
        simu_adata.obsm["spatial"][:,1] = simu_adata.obsm["spatial"][:,1]- temp2
        spot_expr, spot_loc, spot_cell_idx_matrix,spot_ct_count  = get_sim_spot_level_expr(spot_num=None,
                    spot_row=None,
                    spot_col=None,
                    Num_sample=simu_adata.shape[0],
                    spot_diameter=spot_diameter,
                    image_width=np.max(simu_adata.obsm["spatial"][:,0]),
                    image_height=np.max(simu_adata.obsm["spatial"][:,1]),
                    celltype_assignment=simu_adata.obs.label,
                    coord_type="generic",
                    cell_spatial=simu_adata.obsm["spatial"],
                    sim_cell_expr=simu_adata,
                    gap=0,
                    spot_generate_type="square",
                    cell_coord_type="generic",
                    grid_row=None,
                    grid_col=None)
        mask = np.all(spot_ct_count == 0, axis=1)
        ## adjust spot_loc
        spot_loc[:,0] = spot_loc[:,0] + temp1
        spot_loc[:,1] = spot_loc[:,1] + temp2
        adata = ad.AnnData(spot_expr)
        adata.obsm["spatial"] = spot_loc
        adata.var = simu_adata.var
        adata.obs.index =  ['spot'+str(j) for j in range(spot_expr.shape[0])]
        adata = adata[~mask]
        adata.uns["W"] = spot_ct_count[~mask]
        adata.write_h5ad(file_path+'newsimu/'+ method +"simu_spot_level" + ".h5ad")
    return adata

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