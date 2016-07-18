# -*- coding: utf-8 -*-
"""A set of routines for estimating the temporal components, given the spatial components and temporal components
@author: agiovann
"""
from scipy.sparse import spdiags,coo_matrix#,csgraph
import scipy
import numpy as np
from deconvolution import constrained_foopsi
from utilities import update_order
import sys
import time
#%%
def make_G_matrix(T,g):
    ''' create matrix of autoregression to enforce indicator dynamics
    Inputs: 
    T: positive integer
        number of time-bins
    g: nd.array, vector p x 1
        Discrete time constants
        
    Output:
    G: sparse diagonal matrix
        Matrix of autoregression
    '''    
    if type(g) is np.ndarray:    
        if len(g) == 1 and g < 0:
            g=0
    
#        gs=np.matrix(np.hstack((-np.flipud(g[:]).T,1)))
        gs=np.matrix(np.hstack((1,-(g[:]).T)))
        ones_=np.matrix(np.ones((T,1)))
        G = spdiags((ones_*gs).T,range(0,-len(g)-1,-1),T,T)    
        
        return G
    else:
        raise Exception('g must be an array')
#%%
def constrained_foopsi_parallel(arg_in):
    """ necessary for parallel computation of the function  constrained_foopsi
    """  
   
    Ytemp, nT, jj_, bl, c1, g, sn, argss = arg_in
    T=np.shape(Ytemp)[0]
    #cc_,cb_,c1_,gn_,sn_,sp_ = constrained_foopsi(Ytemp/nT, bl = bl,  c1 = c1, g = g,  sn = sn, **argss)
    cc_,cb_,c1_,gn_,sn_,sp_ = constrained_foopsi(Ytemp, bl = bl,  c1 = c1, g = g,  sn = sn, **argss)
    gd_ = np.max(np.roots(np.hstack((1,-gn_.T))));  
    gd_vec = gd_**range(T)   
    

    C_ = cc_[:].T + cb_ + np.dot(c1_,gd_vec)
    Sp_ = sp_[:T].T
    #Ytemp_ = Ytemp - np.dot(nT,C_).T
    Ytemp_ = Ytemp - C_.T
    
    return C_,Sp_,Ytemp_,cb_,c1_,sn_,gn_,jj_
    
     
#%%
def update_temporal_components(Y, A, b, Cin, fin, bl = None,  c1 = None, g = None,  sn = None, ITER=2, method_foopsi='constrained_foopsi', n_processes=1, backend='single_thread',memory_efficient=False, debug=False, **kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.
    
    Parameters
    -----------    

    Y: np.ndarray (2D)
        input data with time in the last axis (d x T)
    A: sparse matrix (crc format)
        matrix of temporal components (d x K)
    b: ndarray (dx1)
        current estimate of background component
    Cin: np.ndarray
        current estimate of temporal components (K x T)   
    fin: np.ndarray
        current estimate of temporal background (vector of length T)
    g:  np.ndarray
        Global time constant (not used)
    bl: np.ndarray
       baseline for fluorescence trace for each column in A
    c1: np.ndarray
       initial concentration for each column in A
    g:  np.ndarray       
       discrete time constant for each column in A
    sn: np.ndarray
       noise level for each column in A       
    ITER: positive integer
        Maximum number of block coordinate descent loops. 
    method_foopsi: string
        Method of deconvolution of neural activity. constrained_foopsi is the only method supported at the moment.               
    n_processes: int
        number of processes to use for parallel computation. Should be less than the number of processes started with ipcluster.
    backend: 'str'
        single_thread no parallelization
        ipyparallel, parallelization using the ipyparallel cluster. You should start the cluster (install ipyparallel and then type 
        ipcluster -n 6, where 6 is the number of processes). 
        SLURM: using SLURM scheduler
    memory_efficient: Bool
        whether or not to optimize for memory usage (longer running times). nevessary with very large datasets  
    **kwargs: dict
        all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation). Some useful parameters are      
    p: int
        order of the autoregression model
    method: [optional] string
        solution method for constrained foopsi. Choices are
            'cvx':      using cvxopt and picos (slow especially without the MOSEK solver)
            'cvxpy':    using cvxopt and cvxpy with the ECOS solver (faster, default)
    
    solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers to be used with cvxpy, default is ['ECOS','SCS']
            
    Note
    --------

    The temporal components are updated in parallel by default by forming of sequence of vertex covers.  
    
    Returns
    --------
    
    C:   np.ndarray
            matrix of temporal components (K x T)
    f:   np.array
            vector of temporal background (length T) 
    S:   np.ndarray            
            matrix of merged deconvolved activity (spikes) (K x T)
    bl:  float  
            same as input    
    c1:  float
            same as input    
    g:   float
            same as input    
    sn:  float
            same as input 
    YrA: np.ndarray
            matrix of spatial component filtered raw data, after all contributions have been removed.            
            YrA corresponds to the residual trace for each component and is used for faster plotting (K x T)
    """
    if not kwargs.has_key('p') or kwargs['p'] is None:
        raise Exception("You have to provide a value for p")

    d,T = np.shape(Y);    
    nr = np.shape(A)[-1]
    
    
    if  bl is None:
        bl=np.repeat(None,nr)
        
    if  c1 is None:
        c1=np.repeat(None,nr)

    if  g is None:
        g=np.repeat(None,nr)

    if  sn is None:
        sn=np.repeat(None,nr)                        
    
    A = scipy.sparse.hstack((A,coo_matrix(b)))
    S = np.zeros(np.shape(Cin));
    Cin =  np.vstack((Cin,fin));
    C = Cin;
    nA = np.squeeze(np.array(np.sum(np.square(A.todense()),axis=0)))
    #import pdb
    #pdb.set_trace()
    Cin=coo_matrix(Cin)
    #YrA = ((A.T.dot(Y)).T-Cin.T.dot(A.T.dot(A)))
    YA = (A.T.dot(Y).T)*spdiags(1./nA,0,nr+1,nr+1)
    AA = ((A.T.dot(A))*spdiags(1./nA,0,nr+1,nr+1)).tocsr()
    YrA = YA - Cin.T.dot(AA)
    #YrA = ((A.T.dot(Y)).T-Cin.T.dot(A.T.dot(A)))*spdiags(1./nA,0,nr+1,nr+1)
    
    if backend == 'ipyparallel' or backend == 'SLURM':
        try: # if server is not running and raise exception if not installed or not started        
            from ipyparallel import Client
            if backend is 'SLURM':
                if 'IPPPDIR' in os.environ and 'IPPPROFILE' in os.environ:
                    pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
                else:
                    raise Exception('envirnomment variables not found, please source slurmAlloc.rc')
        
                c = Client(ipython_dir=pdir, profile=profile)
                print 'Using '+ str(len(c)) + ' processes'
            else:
                c = Client()
            
        except:
            print "this backend requires the installation of the ipyparallel (pip install ipyparallel) package and  starting a cluster (type ipcluster start -n 6) where 6 is the number of nodes"
            raise
    
        if len(c) <  n_processes:
            print len(c)
            raise Exception("the number of nodes in the cluster are less than the required processes: decrease the n_processes parameter to a suitable value")            
        
        dview=c[:n_processes] # use the number of processes
    
    Cin=np.array(Cin.todense())    
    for iter in range(ITER):
        O,lo = update_order(A.tocsc()[:,:nr])
        P_=[];
        for count,jo_ in enumerate(O):
            jo=np.array(list(jo_))           
            #Ytemp = YrA[:,jo.flatten()] + (np.dot(np.diag(nA[jo]),Cin[jo,:])).T
            Ytemp = YrA[:,jo.flatten()] + Cin[jo,:].T
            Ctemp = np.zeros((np.size(jo),T))
            Stemp = np.zeros((np.size(jo),T))
            btemp = np.zeros((np.size(jo),1))
            sntemp = btemp.copy()
            c1temp = btemp.copy()
            gtemp = np.zeros((np.size(jo),kwargs['p']));
            nT = nA[jo]            
                        
#            args_in=[(np.squeeze(np.array(Ytemp[:,jj])), nT[jj], jj, bl[jo[jj]], c1[jo[jj]], g[jo[jj]], sn[jo[jj]], kwargs) for jj in range(len(jo))]
            args_in=[(np.squeeze(np.array(Ytemp[:,jj])), nT[jj], jj, None, None, None, None, kwargs) for jj in range(len(jo))]
#            import pdb
#            pdb.set_trace()
            if backend == 'ipyparallel' or backend == 'SLURM':                    
                #
                if debug:                
                    results = dview.map_async(constrained_foopsi_parallel,args_in)  
                    results.get()
                    for outp in results.stdout:   
                        print outp[:-1]  
                        sys.stdout.flush()                                                 
                    for outp in results.stderr:   
                        print outp[:-1]  
                        sys.stderr.flush()            
                    
                else:
                    
                    results = dview.map_sync(constrained_foopsi_parallel,args_in)
                
            elif backend == 'single_thread':
                
                results = map(constrained_foopsi_parallel,args_in)            
                
            else:
                
                raise Exception('Backend not defined. Use either single_thread or ipyparallel or SLURM')
                
            for chunk in results:
                pars=dict()
                C_,Sp_,Ytemp_,cb_,c1_,sn_,gn_,jj_=chunk                    
                Ctemp[jj_,:] = C_[None,:]
                                
                Stemp[jj_,:] = Sp_               
                Ytemp[:,jj_] = Ytemp_[:,None]            
                btemp[jj_] = cb_
                c1temp[jj_] = c1_
                sntemp[jj_] = sn_   
                gtemp[jj_,:] = gn_.T  
                   
                bl[jo[jj_]] = cb_
                c1[jo[jj_]] = c1_
                sn[jo[jj_]] = sn_
                g[jo[jj_]]  = gn_.T if kwargs['p'] > 0 else [] #gtemp[jj,:]
                                             
                pars['b'] = cb_
                pars['c1'] = c1_                 
                pars['neuron_sn'] = sn_
                pars['gn'] = gtemp[jj_,np.abs(gtemp[jj,:])>0] 
                pars['neuron_id'] = jo[jj_]
                P_.append(pars)
            
            YrA -= (Ctemp-C[jo,:]).T*AA[jo,:]
            #YrA[:,jo] = Ytemp
            C[jo,:] = Ctemp.copy()            
            S[jo,:] = Stemp
            
#            if (np.sum(lo[:jo])+1)%1 == 0:
            print str(np.sum(lo[:count+1])) + ' out of total ' + str(nr) + ' temporal components updated'
        
        ii=nr        

        #YrA[:,ii] = YrA[:,ii] + np.atleast_2d(Cin[ii,:]).T
        #cc = np.maximum(YrA[:,ii],0)        
        cc = np.maximum(YrA[:,ii] + np.atleast_2d(Cin[ii,:]).T,0)
        YrA -= (cc-np.atleast_2d(Cin[ii,:]).T)*AA[ii,:]      
        C[ii,:] = cc.T
        #YrA = YA - C.T.dot(AA)
        #YrA[:,ii] = YrA[:,ii] - np.atleast_2d(C[ii,:]).T                
        
        if backend == 'ipyparallel' or  backend == 'SLURM':       
            dview.results.clear()   
            c.purge_results('all')
            c.purge_everything()

        if scipy.linalg.norm(Cin - C,'fro')/scipy.linalg.norm(C,'fro') <= 1e-3:
            # stop if the overall temporal component does not change by much
            print "stopping: overall temporal component not changing significantly"
            break
        else:
            Cin = C
        
    f = C[nr:,:]
    C = C[:nr,:]
    YrA = np.array(YrA[:,:nr]).T    
    P_ = sorted(P_, key=lambda k: k['neuron_id']) 
    if backend == 'ipyparallel' or  backend == 'SLURM':      
        c.close()
    
    return C,f,S,bl,c1,sn,g,YrA #,P_