import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import sys
import os
from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.typing import List
from pathlib import Path
from pprint import pprint

parser = ArgumentParser()

parser.add_argument(
    "--meerpower-path",
    type=str,
    help="Path to meerpower repository."
)
parser.add_argument(
    "--survey",
    type=str,
    default="2021",
    help="One of either '2019' or '2021'."
)
parser.add_argument(
    "--filepath-HI",
    type=str,
    help="Path to an HI map file in FITS format."
)
parser.add_argument(
    "--gal-cat",
    type=str,
    default="gama",
    help="Galaxy catalog name.  Can be one of 'gama', 'wigglez', or 'cmass'."
)
parser.add_argument(
    "--filepath-g",
    type=str,
    help="Path to a galaxy catalog file in txt format (wigglez) or FITS "
         "format (gama or cmass)."
)
parser.add_argument(
    "--LoadTF",
    action="store_true",
    help="If passed, load an existing transfer function from disk."
)
parser.add_argument(
    "--do2DTF",
    action="store_true",
    help="Compute the two-dimensional foreground transfer function."
)
parser.add_argument(
    "--doHIauto",
    action="store_true",
    help="Compute the autocorrlation power spectrum, i.e. HI x HI."
)
parser.add_argument(
    "--doMock",
    action="store_true",
    help="Use mock data for consistency checks."
)
parser.add_argument(
    "--mockindx",
    type=int,
    help="Mock file index.  If you pass '--doMock' and no value for "
         "'--mockindx' is passed, a random index will be used."
)
parser.add_argument(
    "--Nmock",
    type=int,
    help="Number of mock files."
)
parser.add_argument(
    "--mockfilepath-HI",
    type=str,
    help="Path and base name for a set of numpy-readable, indexed mock HI "
         "files.  For example, if the mock data are stored in "
         "'/path/to/mocks_{index}.npy', you would pass "
         "'--mockfilepath-HI /path/to/mocks' and exclude the '_{index}.npy' "
         "suffix."
)
parser.add_argument(
    "--mockfilepath-g",
    type=str,
    help="Path and base name for a set of numpy-readable, indexed mock galaxy "
         "files.  For example, if the mock data are stored in "
         "'/path/to/mocks_{index}.npy', you would pass "
         "'--mockfilepath-gal /path/to/mocks' and exclude hte '_{index}.npy' "
         "suffix."
)
parser.add_argument(
    "--Nfg",
    dest="N_fg",
    type=int,
    default=10,
    help="Number of PCA FG modes.  Defaults to 10."
)
parser.add_argument(
    "--gamma",
    type=float,
    default=1.4,
    help="Resmoothing factor.  Defaults to 1.4."
)
parser.add_argument(
    "--tukey-alphas",
    type=List[float],
    default=[0.5, 0.1, 0.2, 0.8, 1],
    help="List of Tukey window shape parameter values.  Passed as a list of "
         "floats with no spaces between commas.  A single value can be passed "
         "as '--tukey-alphas [0.1]'.  Multiple values would be passed as "
         "'--tukey-alphas [0.1,0.5,1]'.  Defaults to [0.5, 0.1, 0.2, 0.8, 1]."
)
parser.add_argument(
    "--grid-seed",
    type=int,
    default=834515,
    help="Random seed for the regridding step.  Fixes the random locations "
         "of the sampling particles within pixels."
)
parser.add_argument(
    "--out-dir",
    type=str,
    help="Path to a directory for output files."
)
parser.add_argument(
    "--config",
    action=ActionConfigFile
)

args = parser.parse_args()
print("Command Line Arguments:")
pprint(args.__dict__)

sys.path.insert(1, (Path(args.meerpower_path) / 'meerpower').as_posix())
import Init
import plot

def RunPipeline(
    survey,
    filepath_HI,
    gal_cat,
    filepath_g,
    N_fg,
    gamma=1.4,
    kcuts=None,
    LoadTF=False,
    do2DTF=False,
    doHIauto=False,
    doMock=False,
    mockindx=None,
    Nmock=500,
    mockfilepath_HI=None,
    mockfilepath_g=None,
    out_dir="./",
    tukey_alpha=0.1
):
    '''
    Use for looping over full pipeline with different choices of inputs for purposes
    of transfer function building. Input choices from below:
    # survey = '2019' or '2021'
    # map_file = Path to a HI map file in FITS format
    # gal_cat = 'wigglez', 'cmass', or 'gama'
    # filepath_g = Path to a galaxy catalog file in txt format (wigglez) or FITS format (gama or cmass).
    # N_fg = int (the number of PCA components to remove)
    # gamma = float or None (resmoothing parameter)
    # kcuts = [kperpmin,kparamin,kperpmax,kparamax] or None (exclude areas of k-space from spherical average)]
    # do2DTF = Compute the two-dimensional foreground transfer function
    # doHIauto = Compute the autocorrlation power spectrum, i.e. HI x HI
    # doMock = Use mock data for consistency checks
    # mockindx = Mock file index.  If None (default), choose a random index
    # mockfilepath_HI = Path and base name for a set of numpy-readable, indexed files containing mock HI data.
    #                   For example, if the mock data are stored in '/path/to/mocks_{index}.npy', you would pass
    #                   'mockfilepath_HI=/path/to/mocks' and exclude the '_{index}.npy' suffix.
    # mockfilepath_g = Path and base name for a set of numpy-readable, indexed files containing mock galaxy data.
    #                  For example, if the mock data are stored in '/path/to/mocks_{index}.npy', you would pass
    #                  'mockfilepath_g=/path/to/mocks' and exclude the '_{index}.npy' suffix.
    # out_dir = Path to the meerpower
    # tukey_alpha = Tukey window shape parameter
    # mp_path = Path to meerpower repository
    '''
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)

    # Load data and run some pre-processing steps:
    doMock = False # Set True to load mock data for consistency checks

    if survey=='2019':
        # filestem = '/idia/projects/hi_im/raw_vis/katcali_output/level6_output/p0.3d/p0.3d_sigma2.5_iter2/'
        # map_file = filestem + 'Nscan366_Tsky_cube_p0.3d_sigma2.5_iter2.fits'
        numin,numax = 971,1023.2
    if survey=='2021':
        # filestem = '/idia/users/jywang/MeerKLASS/calibration2021/level6/0.3/sigma4_count40/re_cali1_round5/'
        # map_file = filestem + 'Nscan961_Tsky_cube_p0.3d_sigma4.0_iter2.fits'
        numin,numax = 971,1023.8 # default setting in Init.ReadIn()
    MKmap,w_HI,W_HI,counts_HI,dims,ra,dec,nu,wproj = Init.ReadIn(filepath_HI,numin=numin,numax=numax)
    if doMock==True:
        if mockindx is None:
            mockindx = np.random.randint(100)
        # MKmap_mock = np.load('/idia/projects/hi_im/meerpower/'+survey+'Lband/mocks/dT_HI_p0.3d_wBeam_%s.npy'%mockindx)
        MKmap_mock = np.load(mockfilepath_HI + '_%s.npy'%mockindx)
    nx,ny,nz = np.shape(MKmap)

    ### Remove incomplete LoS pixels from maps:
    MKmap,w_HI,W_HI,counts_HI = Init.FilterIncompleteLoS(MKmap,w_HI,W_HI,counts_HI)

    ### IM weights (averaging of counts along LoS so not to increase rank of the map for FG cleaning):
    w_HI = np.repeat(np.mean(counts_HI,2)[:, :, np.newaxis], nz, axis=2)

    # Initialise some fiducial cosmology and survey parameters:
    import cosmo
    nu_21cm = 1420.405751 #MHz
    #from astropy.cosmology import Planck15 as cosmo_astropy
    zeff = (nu_21cm/np.median(nu)) - 1 # Effective redshift - defined as redshift of median frequency
    zmin = (nu_21cm/np.max(nu)) - 1 # Minimum redshift of band
    zmax = (nu_21cm/np.min(nu)) - 1 # Maximum redshift of band
    cosmo.SetCosmology(builtincosmo='Planck18',z=zeff,UseCLASS=True)
    Pmod = cosmo.GetModelPk(zeff,kmax=25,UseCLASS=True) # high-kmax needed for large k-modes in NGP alisasing correction
    f = cosmo.f(zeff)
    sig_v = 0
    b_HI = 1.5
    OmegaHIbHI = 0.85e-3 # MKxWiggleZ constraint
    OmegaHI = OmegaHIbHI/b_HI
    import HItools
    import telescope
    Tbar = HItools.Tbar(zeff,OmegaHI)
    r = 0.9 # cross-correlation coefficient
    D_dish = 13.5 # Dish-diameter [metres]
    theta_FWHM,R_beam = telescope.getbeampars(D_dish,np.median(nu))
    # gamma = 1.4 # resmoothing factor - set = None to have no resmoothing
    #gamma = None

    ### Map resmoothing:
    MKmap_unsmoothed = np.copy(MKmap)
    if gamma is not None:
        w_HI_orig = np.copy(w_HI)
        MKmap,w_HI = telescope.weighted_reconvolve(MKmap,w_HI_orig,W_HI,ra,dec,nu,D_dish,gamma=gamma)
        if doMock==True: MKmap_mock,null = telescope.weighted_reconvolve(MKmap_mock,w_HI_orig,W_HI,ra,dec,nu,D_dish,gamma=gamma)

    ### Trim map edges:
    doTrim = True
    if doTrim==True:
        if survey=='2019':
            #raminMK,ramaxMK = 152,173
            #decminMK,decmaxMK = -1,8
            raminMK,ramaxMK = 149,190
            decminMK,decmaxMK = -5,20

        if survey=='2021':
            raminMK,ramaxMK = 334,357
            #decminMK,decmaxMK = -35,-26.5
            decminMK,decmaxMK = np.min(dec[np.mean(W_HI,2)>0]),np.max(dec[np.mean(W_HI,2)>0])
        ### Before trimming map, show contour of trimmed area:
        MKmap_untrim,W_HI_untrim = np.copy(MKmap),np.copy(W_HI)
        MKmap,w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,MKmap,w_HI,W_HI,counts_HI,ramin=raminMK,ramax=ramaxMK,decmin=decminMK,decmax=decmaxMK)

        if survey=='2019':
            cornercut_lim = 146 # set low to turn off
            cornercut = ra - dec < cornercut_lim
            MKmap[cornercut],w_HI[cornercut],W_HI[cornercut],counts_HI[cornercut] = 0,0,0,0

    # Spectral analysis for possible frequency channel flagging:
    #Also remove some corners/extreme temp values

    MKmap_flag,w_HI_flag,W_HI_flag = np.copy(MKmap),np.copy(w_HI),np.copy(W_HI)

    if survey=='2019':
        extreme_temp_LoS = np.zeros(np.shape(ra))
        extreme_temp_LoS[MKmap[:,:,0]>3530] = 1
        extreme_temp_LoS[MKmap[:,:,0]<3100] = 1
        MKmap_flag[extreme_temp_LoS==1] = 0
        w_HI_flag[extreme_temp_LoS==1] = 0
        W_HI_flag[extreme_temp_LoS==1] = 0

    import model
    nra,ndec = np.shape(ra)
    offsets = np.zeros((nra,ndec,len(nu)))
    for i in range(nra):
        for j in range(ndec):
            if W_HI_flag[i,j,0]==0: continue
            poly = model.FitPolynomial(nu,MKmap_flag[i,j,:],n=2)
            offsets[i,j,:] = np.abs((MKmap_flag[i,j,:] - poly)/MKmap_flag[i,j,:])
    offsets = 100*np.mean(offsets,axis=(0,1))

    if survey=='2019': offsetcut = 0.029 # Set to zero for no additional flagging
    #if survey=='2019': offsetcut = None # Set to None for no additional flagging
    if survey=='2021': offsetcut = None

    if offsetcut is None: flagindx = []
    else: flagindx = np.where(offsets>offsetcut)[0]

    flags = np.full(nz,False)
    flags[flagindx] = True

    MKmap_flag[:,:,flags] = 0
    w_HI_flag[:,:,flags] = 0
    W_HI_flag[:,:,flags] = 0

    # Principal component analysis:
    import foreground # PCA clean and transfer function calculations performed here
    import model

    # Foreground clean:
    MKmap,w_HI,W_HI = np.copy(MKmap_flag),np.copy(w_HI_flag),np.copy(W_HI_flag) # Propagate flagged maps for rest of analysis
    if doMock==False:
        MKmap_clean = foreground.PCAclean(MKmap,N_fg=N_fg,W=W_HI,w=w_HI,MeanCentre=True) # weights included in covariance calculation
    if doMock==True:
        MKmap_mock[W_HI==0] = 0 # apply same flags, trims, cuts as data
        #MKmap_clean = MKmap_mock
        MKmap_clean = foreground.PCAclean(MKmap + MKmap_mock ,N_fg=N_fg,W=W_HI,w=w_HI,MeanCentre=True) # weights included in covariance calculation

    W_HI_untrim,w_HI_untrim = np.copy(W_HI),np.copy(w_HI)
    if gal_cat=='wigglez': # obtained from min/max of wigglez catalogue
        ramin_gal,ramax_gal = 152.906631, 172.099625
        decmin_gal,decmax_gal = -1.527391, 8.094599
    if gal_cat=='gama':
        ramin_gal,ramax_gal = 339,351
        decmin_gal,decmax_gal = -35,-30
    '''
    if gal_cat=='wigglez' or gal_cat=='gama':
        MKmap_clean,w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,MKmap_clean,w_HI,W_HI,counts_HI,ramin=ramin_gal,ramax=ramax_gal,decmin=decmin_gal,decmax=decmax_gal)
    '''
    if gal_cat=='wigglez':
        MKmap_clean,w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,MKmap_clean,w_HI,W_HI,counts_HI,ramin=ramin_gal,ramax=ramax_gal,decmin=decmin_gal,decmax=decmax_gal)
    if gal_cat=='gama':
        w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,w_HI,W_HI,counts_HI,ramin=ramin_gal,ramax=ramax_gal,decmin=decmin_gal,decmax=decmax_gal)
        # Re-do clean with trimmed mask:
        MKmap_clean = foreground.PCAclean(MKmap,N_fg=N_fg,W=W_HI,w=w_HI,MeanCentre=True) # weights included in covariance calculation


    # Read-in overlapping galaxy survey:
    from astropy.io import fits
    if survey=='2019':
        if gal_cat=='wigglez':
            if doMock==False: # Read-in WiggleZ galaxies (provided by Laura):
                # galcat = np.genfromtxt('/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11data.dat', skip_header=1)
                galcat = np.genfromtxt(filepath_g, skip_header=1)
                ra_g,dec_g,z_g = galcat[:,0],galcat[:,1],galcat[:,2]
            # if doMock==True: ra_g,dec_g,z_g = np.load('/idia/projects/hi_im/meerpower/2019Lband/mocks/mockWiggleZcat_%s.npy'%mockindx)
            if doMock==True: ra_g,dec_g,z_g = np.load(mockfilepath_g + '_%s.npy'%mockindx)
            z_Lband = (z_g>zmin) & (z_g<zmax) # Cut redshift to MeerKAT IM range:
            ra_g,dec_g,z_g = ra_g[z_Lband],dec_g[z_Lband],z_g[z_Lband]
        if gal_cat=='cmass':
            if doMock==False: # Read-in BOSS-CMASS galaxies (in Yi-Chao's ilifu folder - also publically available from: https://data.sdss.org/sas/dr12/boss/lss):
                # filename = '/idia/users/ycli/SDSS/dr12/galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz'
                # hdu = fits.open(filename)
                hdu = fits.open(filepath_g)
                ra_g,dec_g,z_g = hdu[1].data['RA'],hdu[1].data['DEC'],hdu[1].data['Z']
            # if doMock==True: ra_g,dec_g,z_g = np.load('/idia/projects/hi_im/meerpower/2019Lband/mocks/mockCMASScat_%s.npy'%mockindx)
            if doMock==True: ra_g,dec_g,z_g = np.load(mockfilepath_g + '_%s.npy'%mockindx)
            ra_g,dec_g,z_g = Init.pre_process_2019Lband_CMASS_galaxies(ra_g,dec_g,z_g,ra,dec,zmin,zmax,W_HI)

    if survey=='2021':
        if doMock==False: # Read-in GAMA galaxies:
            # Fits = '/idia/projects/hi_im/GAMA_DR4/G23TilingCatv11.fits'
            # hdu = fits.open(Fits)
            hdu = fits.open(filepath_g)
            ra_g,dec_g,z_g = hdu[1].data['RA'],hdu[1].data['DEC'],hdu[1].data['Z']
        # if doMock==True: ra_g,dec_g,z_g = np.load('/idia/projects/hi_im/meerpower/2021Lband/mocks/mockGAMAcat_%s.npy'%mockindx)
        if doMock==True: ra_g,dec_g,z_g = np.load(mockfilepath_g + '_%s.npy'%mockindx)
        # Remove galaxies outside bulk GAMA footprint so they don't bias the simple binary selection function
        GAMAcutmask = (ra_g>ramin_gal) & (ra_g<ramax_gal) & (dec_g>decmin_gal) & (dec_g<decmax_gal) & (z_g>zmin) & (z_g<zmax)
        ra_g,dec_g,z_g = ra_g[GAMAcutmask],dec_g[GAMAcutmask],z_g[GAMAcutmask]

    print('Number of overlapping ', gal_cat,' galaxies: ', str(len(ra_g)))

    # Assign galaxy bias:
    if gal_cat=='wigglez': b_g = np.sqrt(0.83) # for WiggleZ at z_eff=0.41 - from https://arxiv.org/pdf/1104.2948.pdf [pg.9 rhs second quantity]
    if gal_cat=='cmass':b_g = 1.85 # Mentioned in https://arxiv.org/pdf/1607.03155.pdf
    if gal_cat=='gama': b_g = 2.35 # tuned by eye in GAMA auto-corr
    b_g = 1.9  # Change made to match the value in the notebook ./galaxy_cross.ipynb

    # Gridding maps and galaxies to Cartesian field:
    import grid # use this for going from (ra,dec,freq)->(x,y,z) Cartesian-comoving grid
    cell2vox_factor = 1.5 # increase for lower resolution FFT Cartesian grid
    Np = 5 # number of Monte-Carlo sampling particles per map voxel used in regridding
    window = 'ngp'
    compensate = True
    interlace = False
    nxmap,nymap,nzmap = np.shape(MKmap)
    ndim_rg = int(nxmap/cell2vox_factor),int(nymap/cell2vox_factor),int(nzmap/cell2vox_factor)
    nzcell2vox = int(nzmap/cell2vox_factor)
    if nzcell2vox % 2 != 0: nzcell2vox += 1 # Ensure z-dimension is even for FFT purposes
    ndim_rg = int(nxmap/cell2vox_factor),int(nymap/cell2vox_factor),nzcell2vox
    dims_rg,dims0_rg = grid.comoving_dims(ra,dec,nu,wproj,ndim_rg,W=W_HI_untrim,dobuffer=True) # dimensions of Cartesian grid for FFT
    lx,ly,lz,nx_rg,ny_rg,nz_rg = dims_rg

    # Regrid cleaned map, IM mask and weights to Cartesian field:
    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=MKmap_clean,W=W_HI,Np=Np,seed=args.grid_seed)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
    MKmap_clean_rg,null,null = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=False)
    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=W_HI,W=W_HI,Np=Np,seed=args.grid_seed)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
    W_HI_rg,null,null = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)
    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=w_HI,W=W_HI,Np=Np,seed=args.grid_seed)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
    w_HI_rg = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]

    # Grid galaxies straight to Cartesian field:
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_g,dec_g,z_g,ramean_arr=ra,decmean_arr=dec,doTile=False)
    n_g_rg = grid.mesh(xp,yp,zp,dims=dims0_rg,window=window,compensate=compensate,interlace=interlace,verbose=False)[0]

    # Construct galaxy selection function:
    if survey=='2019':
        if gal_cat=='wigglez': # grid WiggleZ randoms straight to Cartesian field for survey selection:
            BuildSelFunc = False
            if BuildSelFunc==True:
                nrand = 1000 # number of WiggleZ random catalogues to use in selection function (max is 1000)
                W_g_rg = np.zeros(np.shape(n_g_rg))
                for i in range(1,nrand):
                    plot.ProgressBar(i,nrand)
                    # galcat = np.genfromtxt( '/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11rand%s.dat' %'{:04d}'.format(i), skip_header=1)
                    galcat = np.genfromtxt(filepath_g.replace('data.dat', 'rand%s.dat' %'{:04d}'.format(i)), skip_header=1)
                    ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]
                    z_Lband = (z_g_rand>zmin) & (z_g_rand<zmax) # Cut redshift to MeerKAT IM range:
                    ra_g_rand,dec_g_rand,z_g_rand = ra_g_rand[z_Lband],dec_g_rand[z_Lband],z_g_rand[z_Lband]
                    xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
                    W_g_rg += grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]
                W_g_rg /= nrand
                np.save('/idia/projects/hi_im/meerpower/2019Lband/wigglez/data/wiggleZ_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor,W_g_rg)
            W_g_rg = np.load('/idia/projects/hi_im/meerpower/2019Lband/wigglez/data/wiggleZ_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)

        if gal_cat=='cmass':
        # Data obtained from untarrting DR12 file at: https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-DR12NGC-COMPSAM_V6C.tar.gz
            BuildSelFunc = False
            if BuildSelFunc==True:
                nrand = 2048 # number of WiggleZ random catalogues to use in selection function (max is 1000)
                W_g_rg = np.zeros(np.shape(n_g_rg))
                for i in range(1,nrand):
                    plot.ProgressBar(i,nrand)
                    galcat = np.genfromtxt( '/idia/projects/hi_im/meerpower/2019Lband/cmass/sdss/Patchy-Mocks-DR12NGC-COMPSAM_V6C_%s.dat' %'{:04d}'.format(i+1), skip_header=1)
                    ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]
                    ra_g_rand,dec_g_rand,z_g_rand = Init.pre_process_2019Lband_CMASS_galaxies(ra_g_rand,dec_g_rand,z_g_rand,ra,dec,zmin,zmax,W_HI)
                    xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
                    W_g_rg += grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]
                W_g_rg /= nrand
                np.save('/idia/projects/hi_im/meerpower/2019Lband/cmass/data/cmass_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor,W_g_rg)
            W_g_rg = np.load('/idia/projects/hi_im/meerpower/2019Lband/cmass/data/cmass_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)

    if survey=='2021': # grid uncut pixels to obtain binary mask in comoving space in absence of GAMA mocks for survey selection:
        ra_p,dec_p,nu_p = grid.SkyPixelParticles(ra,dec,nu,wproj,Np=Np,seed=args.grid_seed)
        '''
        if doTrim==True:
            MKcutmask = (ra_p>ramin_gal) & (ra_p<ramax_gal) & (dec_p>decmin_gal) & (dec_p<decmax_gal)
            xp,yp,zp = grid.SkyCoordtoCartesian(ra_p[MKcutmask],dec_p[MKcutmask],HItools.Freq2Red(nu_p[MKcutmask]),ramean_arr=ra,decmean_arr=dec,doTile=False)
            null,W_HI_rg,counts = grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)
        '''
        GAMAcutmask = (ra_p>ramin_gal) & (ra_p<ramax_gal) & (dec_p>decmin_gal) & (dec_p<decmax_gal)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p[GAMAcutmask],dec_p[GAMAcutmask],HItools.Freq2Red(nu_p[GAMAcutmask]),ramean_arr=ra,decmean_arr=dec,doTile=False)
        null,W_g_rg,null = grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)

    # Calculate FKP weigts:
    '''
    W_g01_rg = np.ones(np.shape(W_g_rg)) # Binary window function for galaxies to mark dead pixels
    W_g01_rg[W_g_rg==0] = 0
    W_g_rg = W_g_rg/np.sum(W_g_rg) # normalised window function for FKP weight calculation
    P0 = 5000 # at k~0.1
    nbar = np.sum(n_g_rg)/(lx*ly*lz) # Calculate number density inside survey footprint
    w_g_rg = 1/(1 + W_g_rg*(nx*ny*nz)*nbar*P0)
    w_g_rg[W_g01_rg==0] = 0 # zero weight for dead pixels
    '''
    #w_g_rg = np.ones(np.shape(W_g_rg))
    w_g_rg = np.copy(W_g_rg)

    MKmap_clean_rg_notaper,w_HI_rg_notaper,W_HI_rg_notaper = np.copy(MKmap_clean_rg),np.copy(w_HI_rg),np.copy(W_HI_rg)
    n_g_rg_notaper,w_g_rg_notaper,W_g_rg_notaper = np.copy(n_g_rg),np.copy(w_g_rg),np.copy(W_g_rg)

    # Footprint tapering/apodisation:
    ### Chose no taper:
    #taper_HI,taper_g = 1,1

    ### Chose to use Blackman window function along z direction as taper:
    blackman = np.reshape( np.tile(np.blackman(nz_rg), (nx_rg,ny_rg)) , (nx_rg,ny_rg,nz_rg) ) # Blackman function along every LoS
    tukey = np.reshape( np.tile(signal.windows.tukey(nz_rg, alpha=tukey_alpha), (nx_rg,ny_rg)) , (nx_rg,ny_rg,nz_rg) ) # Blackman function along every LoS


    #taper_HI = blackman
    #taper_g = blackman
    taper_HI = tukey
    #taper_g = tukey
    #taper_HI = 1
    taper_g = 1


    # Multiply tapering windows by all fields that undergo Fourier transforms:
    #MKmap_clean_rg,w_HI_rg,W_HI_rg = taper_HI*MKmap_clean_rg_notaper,taper_HI*w_HI_rg_notaper,taper_HI*W_HI_rg_notaper
    #n_g_rg,W_g_rg,w_g_rg = taper_g*n_g_rg_notaper,taper_g*W_g_rg_notaper,taper_g*w_g_rg_notaper
    w_HI_rg = taper_HI*w_HI_rg_notaper
    w_g_rg = taper_g*w_g_rg_notaper

    # Power spectrum measurement and modelling (without signal loss correction):
    import power
    import model
    nkbin = 16
    kmin,kmax = 0.07,0.3
    kbins = np.linspace(kmin,kmax,nkbin+1) # k-bin edges [using linear binning]

    import model
    sig_v = 200  # Changed to match the notebook ./galaxy_cross.ipynb
    dpix = 0.3
    d_c = cosmo.d_com(HItools.Freq2Red(np.min(nu)))
    s_pix = d_c * np.radians(dpix)
    s_para = np.mean( cosmo.d_com(HItools.Freq2Red(nu[:-1])) - cosmo.d_com(HItools.Freq2Red(nu[1:])) )

    ### Galaxy Auto-power (can use to constrain bias and use for analytical errors):
    Pk_g,k,nmodes = power.Pk(n_g_rg,n_g_rg,dims_rg,kbins,corrtype='Galauto',w1=w_g_rg,w2=w_g_rg,W1=W_g_rg,W2=W_g_rg)
    W_g_rg /= np.max(W_g_rg)
    Vfrac = np.sum(W_g_rg)/(nx_rg*ny_rg*nz_rg)
    nbar = np.sum(n_g_rg)/(lx*ly*lz*Vfrac) # Calculate number density inside survey footprint
    P_SN = np.ones(len(k))*1/nbar # approximate shot-noise for errors (already subtracted in Pk estimator)

    # Calculate power specs (to get k's for TF):
    if doHIauto==False:
        Pk_gHI,k,nmodes = power.Pk(MKmap_clean_rg,n_g_rg,dims_rg,kbins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts)
        if gamma is not None: 
            theta_FWHM_max,R_beam_max = telescope.getbeampars(D_dish,np.min(nu))
            R_beam_gam = R_beam_max * np.sqrt(gamma)
        else: R_beam_gam = np.copy(R_beam)
        pkmod,k = model.PkMod(Pmod,dims_rg,kbins,b_HI,b_g,f,sig_v,Tbar1=Tbar,Tbar2=1,r=r,R_beam1=R_beam_gam,R_beam2=0,w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts,s_pix1=s_pix,s_para1=s_para,interpkbins=True,MatterRSDs=True,gridinterp=True)[0:2]
    if doHIauto==True:
        Pk_HI,k,nmodes = power.Pk(MKmap_clean_rg,MKmap_clean_rg,dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
    # Calculate P_SN and sig_err()
    sig_err = 1/np.sqrt(2*nmodes) * np.sqrt( Pk_gHI**2 + Pk_HI*( Pk_g + P_SN ) ) # Error estimate

    # LoadTF = False
    # Nmock = 500
    if gamma is None: gamma_label = 'None'
    else: gamma_label = str(gamma)
    if kcuts is None: kcuts_label = 'nokcuts'
    else: kcuts_label = 'withkcuts'
    # mockfilepath_HI = '/idia/projects/hi_im/meerpower/'+survey+'Lband/mocks/dT_HI_p0.3d_wBeam'
    # if gal_cat=='wigglez': mockfilepath_g = '/idia/projects/hi_im/meerpower/2019Lband/mocks/mockWiggleZcat'
    # if gal_cat=='cmass': mockfilepath_g = '/idia/projects/hi_im/meerpower/2019Lband/mocks/mockCMASScat'
    # if gal_cat=='gama': mockfilepath_g = '/idia/projects/hi_im/meerpower/2021Lband/mocks/mockGAMAcat'

    out_dir_tf = out_dir / f'{survey}Lband' / gal_cat / 'TFdata'
    out_dir_tf.mkdir(exist_ok=True, parents=True)
    if do2DTF==False:
        if doHIauto==False:
            #TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)+kcuts_label
            # TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)+kcuts_label+'_tukeyHI=%s'%tukey_alpha
            TFfile = (out_dir_tf / f'T_Nfg={N_fg}_gamma={gamma_label}_{kcuts_label}_tukeyHI={tukey_alpha}_Nmock={Nmock}').as_posix()
            if args.grid_seed is not None:
                TFfile += f'_seed_{args.grid_seed}'
            T_wsub_i, T_nosub_i,k  = foreground.TransferFunction(MKmap_unsmoothed,Nmock,N_fg,'Cross',kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,
                                                        Np,window,compensate,interlace,mockfilepath_HI,mockfilepath_g,gal_cat=gal_cat,
                                                        gamma=gamma,D_dish=D_dish,w_HI=w_HI,W_HI=W_HI,doWeightFGclean=True,PCAMeanCentre=True,
                                                        w_HI_rg=w_HI_rg,W_HI_rg=W_HI_rg,w_g_rg=w_g_rg,W_g_rg=W_g_rg,kcuts=kcuts,
                                                        taper_HI=taper_HI,taper_g=taper_g,LoadTF=LoadTF)
        if doHIauto==True:
            # TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T_HIauto_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)+kcuts_label
            TFfile = (out_dir_tf / ('T_HIauto_Nfg=%s_gamma=%s_'%(N_fg,gamma_label) + kcuts_label)).as_posix()
            T_wsub_i, T_nosub_i,k  = foreground.TransferFunction(MKmap_unsmoothed,Nmock,N_fg,'HIauto',kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,
                                                        Np,window,compensate,interlace,mockfilepath_HI,mockfilepath_g,gal_cat=gal_cat,
                                                        gamma=gamma,D_dish=D_dish,w_HI=w_HI,W_HI=W_HI,doWeightFGclean=True,PCAMeanCentre=True,
                                                        w_HI_rg=w_HI_rg,W_HI_rg=W_HI_rg,w_g_rg=w_g_rg,W_g_rg=W_g_rg,kcuts=kcuts,
                                                        taper_HI=taper_HI,taper_g=taper_g,LoadTF=LoadTF)
    if do2DTF==True:
        kperpbins = np.linspace(0.008,0.3,34)
        kparabins = np.linspace(0.003,0.6,22)
        if doHIauto==False:
            # TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T2D_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)
            TFfile = (out_dir_tf / ('T2D_Nfg=%s_gamma=%s_'%(N_fg,gamma_label))).as_posix()
            T2d_wsub_i, T2d_nosub_i,k2d  = foreground.TransferFunction(MKmap_unsmoothed,Nmock,N_fg,'Cross',kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,
                                                        Np,window,compensate,interlace,mockfilepath_HI,mockfilepath_g,gal_cat=gal_cat,
                                                        gamma=gamma,D_dish=D_dish,w_HI=w_HI,W_HI=W_HI,doWeightFGclean=True,PCAMeanCentre=True,
                                                        w_HI_rg=w_HI_rg,W_HI_rg=W_HI_rg,w_g_rg=w_g_rg,W_g_rg=W_g_rg,kcuts=kcuts,
                                                        taper_HI=taper_HI,taper_g=taper_g,LoadTF=LoadTF,TF2D=True,kperpbins=kperpbins,kparabins=kparabins)
        if doHIauto==True:
            # TFfile = '/idia/projects/hi_im/meerpower/'+survey+'Lband/'+gal_cat+'/TFdata/T2D_HIauto_Nfg=%s_gamma=%s_'%(N_fg,gamma_label)
            TFfile = (out_dir_tf / ('T2D_HIauto_Nfg=%s_gamma=%s_'%(N_fg,gamma_label))).as_posix()
            T2d_wsub_i, T2d_nosub_i,k2d  = foreground.TransferFunction(MKmap_unsmoothed,Nmock,N_fg,'HIauto',kbins,k,TFfile,ra,dec,nu,wproj,dims0_rg,
                                                        Np,window,compensate,interlace,mockfilepath_HI,mockfilepath_g,gal_cat=gal_cat,
                                                        gamma=gamma,D_dish=D_dish,w_HI=w_HI,W_HI=W_HI,doWeightFGclean=True,PCAMeanCentre=True,
                                                        w_HI_rg=w_HI_rg,W_HI_rg=W_HI_rg,w_g_rg=w_g_rg,W_g_rg=W_g_rg,kcuts=kcuts,
                                                        taper_HI=taper_HI,taper_g=taper_g,LoadTF=LoadTF,TF2D=True,kperpbins=kperpbins,kparabins=kparabins)
    
    if not do2DTF:
        # Apply transfer function and save power spectrum plots
        # Code copied from https://github.com/meerklass/meerpower/blob/main/allLband/galaxy_cross.ipynb
        # Transfer function arrays have shape (Nmock, Nk)
        T_wsub_i = np.array(T_wsub_i, dtype=float)
        T_nosub_i = np.array(T_nosub_i, dtype=float)
        k = np.array(k, dtype=float)
        
        T_i = np.copy(T_nosub_i)
        T = np.mean(T_i, 0)
        deltaT_i = T_i - T
        Pk_rec = Pk_gHI/T

        fig_kwargs = {'facecolor': 'w'}
        suffix = f'Nfg_{N_fg}_gamma_{gamma}_{kcuts_label}_talpha_{tukey_alpha}_Nmock_{Nmock}'
        if args.grid_seed is not None:
            suffix += f'_seed_{args.grid_seed}'
        ps_dir = out_dir_tf.parent

        ### TF variance:
        fig = plt.figure()
        plt.axhline(0,lw=0.8,color='black')
        plt.axhline(1,lw=0.8,color='black')
        plt.errorbar(k,np.mean(T_wsub_i,0),np.std(T_wsub_i,0),label='with [] sub',zorder=1)
        plt.errorbar(k+0.002,np.mean(T_nosub_i,0),np.std(T_nosub_i,0),label='no [] sub',ls='--',zorder=1)
        for i in range(Nmock):
            if i==0: plt.plot(k,T_nosub_i[i],lw=0.2,color='gray',label='no [] sub realisations', zorder=0)
            else: plt.plot(k,T_nosub_i[i],lw=0.2,color='gray', zorder=0)
        plt.legend(fontsize=12)
        plt.title('MK X ' + gal_cat + r' with $N_{\rm fg}=%s$ (%s mocks)'%(N_fg,Nmock))
        plt.xlabel(r'$k$')
        plt.ylabel(r'$T(k)$')
        plt.ylim(-0.5,1.5)
        fig.savefig(ps_dir / f'tf_variance_{suffix}.pdf', **fig_kwargs)

        # Propagate error on TF into error on power:
        deltaPk_i =  Pk_rec * (deltaT_i/T) 
        Pk_rec_i = Pk_rec + deltaPk_i # corrected power uncertainty distribution
        np.save(ps_dir / f'k_{suffix}.npy', k)
        np.save(ps_dir / f'Pk_rec_i_{suffix}.npy', Pk_rec_i)
        # Calculate 68th percentile regions for non-symmetric/non-Gaussian errors:
        ### 68.27%/2 = 34.135%. So 50-34.135 -> 50+34.135 covers 68th percentile region:
        lower_error = np.abs(np.percentile(deltaPk_i,15.865,axis=0))
        upper_error = np.abs(np.percentile(deltaPk_i,84.135,axis=0))
        asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        np.save(ps_dir / f'Pk_rec_i_errorbars_{suffix}.npy', asymmetric_error)

        # k-bin covariance/correlation matrices from TF:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,9))
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        kgrid = k * k[:,np.newaxis]
        C = kgrid**3*np.cov(Pk_rec_i,rowvar=False)
        im = ax[0].imshow(C,origin='lower',extent=[kbins[0],kbins[-1],kbins[0],kbins[-1]])
        ax[0].set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        ax[0].set_ylabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax[0].set_title(r'$(k_ik_j)^3\times$ Covariance ($N_{\rm fg}=%s$)'%N_fg,fontsize=20)
        # Noramlised k-bin correlation matrix:
        R = np.corrcoef(Pk_rec_i,rowvar=False)
        im = ax[1].imshow(R,origin='lower',cmap='RdBu',vmin=-1,vmax=1,extent=[kbins[0],kbins[-1],kbins[0],kbins[-1]])
        ax[1].set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        ax[1].set_ylabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax[1].set_title(r'Normalised covariance "correlation matrix" ($N_{\rm fg}=%s$)'%N_fg,fontsize=20)
        plt.subplots_adjust(wspace=0.25)
        fig.savefig(ps_dir / f'kbin_cov_corr_matrices_{suffix}.pdf', **fig_kwargs)

        # Plot results, correcting for signal loss and with TF-based errors:
        # Chose factorisation of P(k) in plotting:
        norm = np.ones(nkbin)
        fig = plt.figure()
        for i in range(Nmock):
            plt.plot(k,k**2*Pk_rec_i[i],lw=0.2,color='gray',zorder=-1)
        plt.errorbar(k,k**2*Pk_rec,k**2*asymmetric_error,ls='none',marker='o',label=r'$N_{\rm fg}=%s$ (percentile)'%N_fg,markersize=10)
        plt.plot(k,k**2*pkmod,color='black',ls='--',label=r'Model [$\Omega_{\rm HI}b_{\rm HI} = %s \times 10^{-3}]$'%np.round(OmegaHI*b_HI*1e3,2))
        plt.axhline(0,lw=0.8,color='black')
        plt.legend(fontsize=12,loc='upper right',frameon=False)
        plt.title('MeerKAT x ' + gal_cat)
        plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        plt.ylabel(r'$k^2\,P_{\rm g,HI}(k)\,[{\rm mK}\,h^{-1}{\rm Mpc}]$')
        plt.ylim(-3*np.abs(np.min(k**2*Pk_rec)),3*np.max(k**2*Pk_rec))
        fig.savefig(ps_dir / f'pspec_{suffix}.pdf', **fig_kwargs)

        ### Histogram of TF mock distributions to show error profile for each k-bin:
        nrows = int(nkbin/4)
        fig, ax = plt.subplots(nrows=nrows, ncols=4, figsize=(20,12))
        k_ind = 0
        for row in ax:
            for col in row:
                if k_ind==nkbin: break
                dum, = col.plot(0,label=r'$k=%s$'%np.round(k[k_ind],3),color='white') # dummy line to show k-value in legend
                legend1 = col.legend([dum], [r'$k=%s$'%np.round(k[k_ind],3)], loc='upper left',fontsize=16,handlelength=0,handletextpad=0,frameon=False,borderaxespad=0.1)
                col.add_artist(legend1)
                #mod = col.axvline(pkmod[k_ind],color='black',ls='--',lw=2,zorder=8)
                mod = col.axvline(0,color='black',ls='--',lw=2,zorder=8)
                median = col.axvline(np.median(pkmod[k_ind]-Pk_rec_i[:,k_ind]),color='tab:blue',ls='-',lw=1,zorder=9)
                span = col.axvspan(np.percentile(pkmod[k_ind]-Pk_rec_i[:,k_ind],50-34.1,axis=0), np.percentile(pkmod[k_ind]-Pk_rec_i[:,k_ind],50+34.1,axis=0), alpha=0.4, color='tab:blue',zorder=-10,label='68th percentile')
                if k_ind==0:
                    legend2 = col.legend([mod,median,span], ['Model','Median','68th percentile'], loc='upper right',fontsize=12,frameon=False,handlelength=1.5,handletextpad=0.4,borderaxespad=0.1)
                    col.add_artist(legend2)
                hist,bins = np.histogram(pkmod[k_ind]-Pk_rec_i[:,k_ind],bins=30)
                bins = (bins[1:] + bins[:-1])/2
                col.plot(bins,hist,ds='steps')
                col.set_xlim(bins[0],bins[-1])
                col.set_ylim(0)
                col.set_yticks([])
                if k_ind>=12: col.set_xlabel(r'$P_{\rm g,HI}(k)\,[{\rm mK}\,h^{-1}{\rm Mpc}]$')
                k_ind += 1
        plt.subplots_adjust(hspace=0.4)
        plt.suptitle(r'Power distribution from %s TF mocks for each $k$ ($N_{\rm fg}=%s$)'%(Nmock,N_fg))
        fig.savefig(ps_dir / f'pspec_distributions_{suffix}.pdf', **fig_kwargs)

        if gal_cat=='cmass': kbin_cut = (k>0.09) & (k<0.25)
        if gal_cat=='gama':
            # Remove the two lowest k bins which are consistent with zero
            # Remove the highest k bin which is negative
            kmincut,kmaxcut = 0.1,0.285

        kbin_cut = (k>kmincut) & (k<kmaxcut) # untrusted k-bins to cut

        sig_v = 400  # sig_v updated here for consistency with the model fitting in the notebook ./galaxy_cross.ipynb

        ### Chi-squared and detection significance for fiducial OmegaHI and diagonal covariance:
        print('\n----------------------------------------------\n---- (fiducial OmegaHI & diagonal covariance):')
        print('\nFiducial OmegaHIbHIr x 10^3 = ',str(np.round(1e3*OmegaHIbHI,3)))
        print('\nReduced Chi^2: ' + str(model.ChiSquare(Pk_rec[kbin_cut],pkmod[kbin_cut],np.std(Pk_rec_i,0)[kbin_cut],dof=len(k[kbin_cut]))))
        det_sig = model.DetectionSigma(Pk_rec[kbin_cut],pkmod[kbin_cut],np.std(Pk_rec_i,0)[kbin_cut])
        OmegaHI = OmegaHIbHI/b_HI
        Tbar = HItools.Tbar(zeff,OmegaHI)
        pkmod,k = model.PkMod(Pmod,dims_rg,kbins,b_HI,b_g,f,sig_v,Tbar1=Tbar,Tbar2=1,r=1,R_beam1=R_beam_gam,R_beam2=0,w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts,s_pix1=s_pix,s_pix2=0,s_para1=s_para,s_para2=0,interpkbins=True,MatterRSDs=True,gridinterp=True)[0:2]

        ### Now get best-fit OmegaHI:
        # Diagonal covariance:
        OmHIbHIr_fit, sigma_OmHIbHIr = model.LSqFitPkAmplitude(Pk_rec,np.std(Pk_rec_i,0),Pmod,zeff,dims_rg,kbins,corrtype='Cross',P_N_=0,kmin=kmincut,kmax=kmaxcut,b2=b_g,f=f,sig_v=sig_v,R_beam1=R_beam_gam,R_beam2=0,w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts,s_pix=s_pix,s_para=s_para)
        OmegaHI_fit = OmHIbHIr_fit/b_HI
        Tbar_fit = HItools.Tbar(zeff,OmegaHI_fit)
        pkmod_fit,k = model.PkMod(Pmod,dims_rg,kbins,b_HI,b_g,f,sig_v,Tbar1=Tbar_fit,Tbar2=1,r=1,R_beam1=R_beam_gam,R_beam2=0,w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts,s_pix1=s_pix,s_pix2=0,s_para1=s_para,s_para2=0,interpkbins=True,MatterRSDs=True,gridinterp=True)[0:2]
        print('\n----------------------------------------------\n---- (fitted OmegaHI & diagonal covariance):')
        print('\nOmegaHIbHIr x 10^3 = ',str(np.round(1e3*OmHIbHIr_fit,3)), '+/-', str(np.round(1e3*sigma_OmHIbHIr,3)))
        print('\nReduced Chi^2 :' + str(model.ChiSquare(Pk_rec[kbin_cut],pkmod_fit[kbin_cut],np.std(Pk_rec_i,0)[kbin_cut],dof=len(k[kbin_cut]))))
        det_sig = model.DetectionSigma(Pk_rec[kbin_cut],pkmod_fit[kbin_cut],np.std(Pk_rec_i,0)[kbin_cut])

        # Full covariance (fiducial OmegaHI):
        print('\n----------------------------------------------\n---- (fiducial OmegaHI & full covariance):')
        print('\nFiducial OmegaHIbHIr x 10^3 = ',str(np.round(1e3*OmegaHIbHI,3)))
        C = np.cov(Pk_rec_i,rowvar=False)
        C = C[kbin_cut]
        C = C[:,kbin_cut]
        print('\nReduced Chi^2 :' + str(model.ChiSquare(Pk_rec[kbin_cut],pkmod[kbin_cut],C,dof=len(k[kbin_cut]))))
        det_sig = model.DetectionSigma(Pk_rec[kbin_cut],pkmod[kbin_cut],C)

        # Full covariance (fitted OmegaHI):
        OmHIbHIr_fit, sigma_OmHIbHIr = model.LSqFitPkAmplitude(Pk_rec,np.cov(Pk_rec_i,rowvar=False),Pmod,zeff,dims_rg,kbins,corrtype='Cross',P_N_=0,kmin=kmincut,kmax=kmaxcut,b2=b_g,f=f,sig_v=sig_v,R_beam1=R_beam_gam,R_beam2=0,w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts,s_pix=s_pix,s_para=s_para)
        OmegaHI_fit = OmHIbHIr_fit/b_HI
        Tbar_fit = HItools.Tbar(zeff,OmegaHI_fit)
        pkmod_fit,k = model.PkMod(Pmod,dims_rg,kbins,b_HI,b_g,f,sig_v,Tbar1=Tbar_fit,Tbar2=1,r=1,R_beam1=R_beam_gam,R_beam2=0,w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts,s_pix1=s_pix,s_pix2=0,s_para1=s_para,s_para2=0,interpkbins=True,MatterRSDs=True,gridinterp=True)[0:2]
        print('\n----------------------------------------------\n---- (fitted OmegaHI & full covariance):')
        print('\nOmegaHIbHIr x 10^3 = ',str(np.round(1e3*OmHIbHIr_fit,3)), '+/-', str(np.round(1e3*sigma_OmHIbHIr,3)))
        print('\nReduced Chi^2 :' + str(model.ChiSquare(Pk_rec[kbin_cut],pkmod_fit[kbin_cut],C,dof=len(k[kbin_cut]))))
        det_sig = model.DetectionSigma(Pk_rec[kbin_cut],pkmod_fit[kbin_cut],C)

        fig = plt.figure()
        plt.errorbar(k,norm*np.abs(Pk_rec),norm*asymmetric_error,ls='none',marker='o',label=r'$N_{\rm fg}=%s$'%N_fg,markersize=10)
        plt.scatter(k[Pk_rec<0],norm[Pk_rec<0]*np.abs(Pk_rec[Pk_rec<0]),marker='o',facecolors='white',color='tab:blue',zorder=10,s=50)
        plt.errorbar(k+0.001,norm*np.abs(Pk_rec),norm*sig_err,ls='none',marker='o',label='Analytical errors',markersize=10,color='tab:orange')
        plt.scatter(k[Pk_rec<0]+0.001,norm[Pk_rec<0]*np.abs(Pk_rec[Pk_rec<0]),marker='o',facecolors='white',color='tab:orange',zorder=10,s=50)
        plt.plot(k,norm*pkmod,color='gray',ls='--',label=r'Model [$\Omega_{\rm HI}b_{\rm HI} = %s \times 10^{-3}]$'%np.round(OmegaHIbHI*1e3,2))
        plt.plot(k,norm*pkmod_fit,color='black',ls='--',label=r'Model [$\Omega_{\rm HI}b_{\rm HI} = %s \times 10^{-3}]$'%np.round(OmHIbHIr_fit*1e3,2))
        if norm[0]==1.0: plt.yscale('log')
        plt.axhline(0,lw=0.8,color='black')
        plt.axvspan(kbins[0],kmincut,color='red',alpha=0.4)
        plt.axvspan(kmaxcut,kbins[-1],color='red',alpha=0.4)
        plt.xlim(kbins[0],kbins[-1])
        plt.ylim(bottom=np.min(norm*pkmod_fit))
        plt.legend(fontsize=16,loc='upper right',frameon=True,framealpha=1)
        plt.title('MeerKAT x ' + gal_cat)
        plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        if norm[0]==1.0: plt.ylabel(r'$P_{\rm g,HI}(k)\,[{\rm mK}\,h^{-3}{\rm Mpc}^{3}]$')
        else: plt.ylabel(r'$k^2\,P_{\rm g,HI}(k)\,[{\rm mK}\,h^{-1}{\rm Mpc}]$')
        fig.savefig(ps_dir / f'omegab_fits_{suffix}.pdf', **fig_kwargs)


# [kperpmin,kparamin,kperpmax,kparamax] (exclude areas of k-space from spherical average)
kcuts = [0.052,0.031,0.175,None]

for i in range(len(args.tukey_alphas)):
    RunPipeline(
        args.survey,
        args.filepath_HI,
        args.gal_cat,
        args.filepath_g,
        args.N_fg,
        gamma=args.gamma,
        kcuts=kcuts,
        LoadTF=args.LoadTF,
        do2DTF=args.do2DTF,
        doHIauto=args.doHIauto,
        doMock=args.doMock,
        mockindx=args.mockindx,
        Nmock=args.Nmock,
        mockfilepath_HI=args.mockfilepath_HI,
        mockfilepath_g=args.mockfilepath_g,
        out_dir=args.out_dir,
        tukey_alpha=args.tukey_alphas[i]
    )
