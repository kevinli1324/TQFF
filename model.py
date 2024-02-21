import numpy as np
import torch
import torch.nn as nn
from linear_operator.operators import DiagLinearOperator, LowRankRootLinearOperator, CholLinearOperator
from gpytorch.distributions import MultivariateNormal
from tqdm import tqdm
import matplotlib.pyplot as plt
from linear_operator.utils.cholesky import psd_safe_cholesky
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution, TrilNaturalVariationalDistribution
from torch.distributions.kl import kl_divergence
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import scipy
import itertools as it
import qmcpy 
from gpytorch.models import ApproximateGP
from sklearn.cluster import KMeans
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution

def genORF(n, d):
    b = int((n + d)/d)
    chi2 = torch.distributions.chi2.Chi2(d)
    G = torch.randn((b, d, d))
    Q, R = torch.linalg.qr(G)
    S = chi2.rsample((b, d)).sqrt()
    G = (np.sqrt(2))*S[:, :, None]*Q
    Gf = G.reshape((G.shape[0]*G.shape[1], G.shape[-1]))
    return Gf[:n]
    
def tensorproduct(nodes, weights, d, cosine = True):
    if d == 1:
        return nodes, weights
    if cosine:
        nodes = torch.cat([nodes, -nodes], axis = 0)
        weights = torch.cat([weights, weights])*.5
    
    nodes = nodes.cpu().double().numpy().squeeze()
    weights = weights.cpu().double().numpy()

    node_list = [nodes for i in range(d)]
    weight_list = [weights for i in range(d)]
    return_array = np.asarray(list(it.product(*node_list)))

    
    weights = np.asarray(list(it.product(*weight_list))).prod(-1)

    return_array = torch.from_numpy(return_array).cuda().float()
    weights = torch.from_numpy(weights).cuda().float()

    if cosine:
        negativeset = set([])
        rel_idx = []
        for idx in range(return_array.shape[0]):
            neghash = tuple((-i.item() for i in return_array[idx]))
            hash = tuple((i.item() for i in return_array[idx]))
            if not neghash in negativeset:
                rel_idx.append(idx)
                negativeset.add(hash)

        return_array = return_array[rel_idx]
        weights = weights[rel_idx]*2
    
    return return_array, weights


def safe_inverse(psd):
    return CholLinearOperator(psd_safe_cholesky(psd, jitter = 1e-4)).inverse().to_dense()

class quadGP(nn.Module):

    def __init__(self, nodes, weights, num_c = 1, theta_init = .5, nu_init = 1.,
                 device = "cuda", N = 1, siginit = .2, dtype = torch.float32, jitter = 1e-4, 
                 gamma = 1):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.theta = nn.Parameter( torch.log((torch.ones((num_c, nodes.shape[1]), device = device, dtype= dtype)*theta_init).square())  )
        self.nu = nn.Parameter(torch.ones(num_c, device = device, dtype = dtype)*nu_init)
        self.sig = nn.Parameter(torch.tensor([siginit], device = device, dtype = dtype))
        self.jitter = jitter
        self.gamma = gamma

        self.nodes = nodes
        self.weights = weights
        self.N = N

    def init_svi(self, N):
        nnodes = self.nodes.shape[0]
        #self.variational_dist = CholeskyVariationalDistribution(nnodes*2).cuda()
        self.variational_dist = NaturalVariationalDistribution(nnodes*2).cuda()
        self.prior = MultivariateNormal(torch.zeros(nnodes*2).cuda(), DiagLinearOperator(torch.ones(nnodes*2).cuda()))
        self.N = N

        
    def get_features(self, X , gl = False, rff = False):
        #X: n x num_c x d
        #nodes: q x d
        #weights q
        if gl:
            exp_feature = torch.exp(-(self.gamma*self.nodes).square().sum(-1)/2)[None, None]
        else:
            exp_feature =  1
        
        c = (2*np.pi**2)
        X = self.gamma*(2*np.pi)*X/torch.sqrt(c*self.theta.exp()[None])
        inner = (self.nodes[None, None, :]*X[:,:, None]).sum(-1)
        self.cosf = torch.cos(inner)
        cos_features = torch.cos(inner)*torch.sqrt(self.weights[None, None]*self.gamma)*exp_feature
        sin_features = torch.sin(inner)*torch.sqrt(self.weights[None, None]*self.gamma)*exp_feature
        F = torch.sqrt(self.nu.square())[None, :, None]*torch.cat([cos_features, sin_features], axis = -1)

        if not rff:
            F  = F/((np.pi)**(.25*self.d))
        
        F = F.flatten(start_dim = 1)

        return F
        
        
    def calculate_covar(self, X, Y, gl = False):
        
        Fx = self.get_features(X, gl = gl)
        Fy = self.get_features(Y, gl = gl)
        
        
        print(Fx.shape)
        print(Fy.shape)
        return Fx.T @ Fy

    def svi_loss(self, X, Y, gl = False, rff = False):
        self.d = X.shape[-1]
        F = self.get_features(X, gl = gl, rff = rff)
        D = torch.ones(X.shape[0], device = self.device, dtype = self.dtype )*self.sig.square() + torch.ones(X.shape[0], device= self.device, dtype = self.dtype)*1e-6
        self.F = F
        cv =self.variational_dist.forward().lazy_covariance_matrix.to_dense()
        
        Exp_quad = (Y[:, None] - F @ self.variational_dist.forward().mean[:, None]).square().sum() + torch.trace(cv @ F.T @ F)
        Exp_quad = Exp_quad*(-1/(2*self.sig.square())) - .5*torch.log(2*torch.pi*self.sig.square())*Y.shape[0]
        
        kl = kl_divergence(self.variational_dist.forward(), self.prior)*(Y.shape[0]/self.N)

        return -(Exp_quad - kl)/(Y.shape[0])

    def loss(self, X, Y, gl = False, rff= False):
        self.d = X.shape[1]
        F = self.get_features(X, gl = gl, rff = rff)
        D = torch.ones(X.shape[0], device = self.device, dtype = self.dtype)*self.sig.square() + torch.ones(X.shape[0], device= self.device, dtype = self.dtype)*self.jitter
        
        Knn = LowRankRootLinearOperator(F) + DiagLinearOperator(D)  
        
        logprob = MultivariateNormal(torch.zeros(X.shape[0], device = self.device), Knn).log_prob(Y)

        return -logprob/Y.shape[0]

        
    def pred(self, Xtest, Xtrain, Ytrain):
        Ftest = self.get_features(Xtest)
        Ftrain = self.get_features(Xtrain)

        D = torch.ones(Xtrain.shape[0], device = self.device)*self.sig.square() + torch.ones(Xtrain.shape[0], device= self.device, dtype = self.dtype)*1e-4
        Knn = LowRankRootLinearOperator(Ftrain) + DiagLinearOperator(D)  
        Knn.add_jitter(1e-4)

        Knni_y = torch.linalg.solve(Knn, Ytrain)

        Knm = Ftest @ Ftrain.T

        Ktt = torch.diag(Ftest @ Ftest.T)
        var = Ktt - Knn.inv_quad_logdet(Knm.T, reduce_inv_quad = False)[0]

        return Ftest @ Ftrain.T @ Knni_y, var

    def bpred(self, Xtest, Xtrain, Ytrain, gl = False, rff = False, batch = 100):
        Ftrain = self.get_features(Xtrain, gl = gl, rff = rff)
        Ftest = self.get_features(Xtest, gl = gl, rff = rff)


        D = torch.ones(Ftrain.shape[1], device = self.device, dtype = self.dtype)
        
        var_term = (Ftrain.T @ Ftrain)/(self.sig.square() + self.jitter) + torch.diag(D)
        var_term = CholLinearOperator(psd_safe_cholesky(var_term, jitter = 1e-5))
        var_termi = torch.linalg.solve(var_term, torch.eye(Ftrain.shape[1], device = self.device))
        #var_termi = safe_inverse(var_term)
        self.var_term = var_term
        #beta = torch.linalg.solve(var_term, Ftrain.T @ Ytrain[:, None])
        beta = var_termi @ Ftrain.T @ Ytrain[:, None]/(self.sig.square() + self.jitter)
        mu = (Ftest @ beta).squeeze()

        var = var_term.inv_quad_logdet(Ftest.T, reduce_inv_quad = False)[0] + self.sig.square()
        
        return mu.detach().cpu().numpy(), var.detach().cpu().numpy()
    
    def svi_pred(self, Xtest, gl = False, rff = False):
        Ftest = self.get_features(Xtest, gl = gl, rff = rff)
        cv =self.variational_dist.forward().lazy_covariance_matrix.to_dense()

        mu = (Ftest @ self.variational_dist.forward().mean[:, None]).squeeze()
        var = self.variational_dist.forward().lazy_covariance_matrix.inv_quad_logdet(Ftest.T, reduce_inv_quad = False)[0]

        return mu, var

    def batchsvi_pred(self, Xtest, Ytest = None, batch = 500, gl = False, rff = False):

        if not Ytest:
            Ytest = torch.zeros(Xtest.shape[0], device = self.device)
        with torch.no_grad():
            test_dataset = TensorDataset(Xtest, Ytest)
            test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
            
            
            mean_vec = torch.tensor([0.], device= self.device)
            var_vec = torch.tensor([0.], device = self.device)
            nll_vec = torch.tensor([0.], device=  self.device)
            for xbatch, ybatch in test_loader:
                mu, var = self.svi_pred(xbatch, rff = rff, gl = gl)
                
                mean_vec = torch.cat([mean_vec, mu])
                var_vec = torch.cat([var_vec, var])
       
        mean_vec=  mean_vec[1:].cpu().detach().numpy()
        var_vec = var_vec[1:].cpu().detach().numpy()

        return mean_vec, var_vec
    def svi_fit(self, X, Y, N, iters = 1000, lr = .005, nlr = .1, gl= False, rff = False, batch = 1024):
            
        train_dataset = TensorDataset(X, Y)
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

        self.init_svi(N)
        param_optimizer = torch.optim.Adam([self.nu, self.theta, self.sig], lr = lr)
        var_optimizer = gpytorch.optim.NGD(self.variational_dist.parameters(), lr = nlr, num_data = X.shape[0])
        epochs_iter = tqdm(range(iters), desc="Epoch")
        for i in epochs_iter:
            for x_batch, y_batch in train_loader:
                param_optimizer.zero_grad()
                var_optimizer.zero_grad()
                loss = self.svi_loss(x_batch, y_batch, gl = gl, rff = rff)
                loss.backward()
                param_optimizer.step()
                var_optimizer.step()
            epochs_iter.set_postfix(loss=loss.item())

    def fit(self,X, Y, iters = 1000, lr = .005, gl = False,rff = False, update = 250):
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
            
        epochs_iter = tqdm(range(iters), desc="Epoch", miniters = update)
        for i in epochs_iter:
            optimizer.zero_grad()
            loss = self.loss(X, Y, gl = gl, rff = rff)
            loss.backward()
            optimizer.step()
            if i % update == 0:
                epochs_iter.set_postfix(loss=loss.item())


def get_covxy(X, Y, theta, nu):
    inner = ((X[:, None] - Y[None])/(np.sqrt(2)*theta[None, None]))

    return nu*torch.exp(-inner.square().sum(-1))



def nodes_weights(method, n, d, usescipy = False, gl_trunc = np.pi, gamma = 1):
    #methods
    #qmc, rff, gl, gh, trig
    gamma_dict = {1: "_1", 1.15: "_15", .85: "_085", 1.25: "_25"}
    dist =  torch.distributions.normal.Normal(0, 1/np.sqrt(2), validate_args=None)
    soboleng = torch.quasirandom.SobolEngine(dimension=d, scramble = True)

    sampler = scipy.stats.qmc.Halton(d=d, scramble=True)
    dnb2 = qmcpy.discrete_distribution.digital_net_b2.digital_net_b2.DigitalNetB2(d, seed = 7)
    
    with open("quad_weights/quad_dict{}.pickle".format(gamma_dict[gamma]), "rb") as openfile:
        quad_obj = pickle.load(openfile)
    

    if method == "trigl":
        with open("quad_weights/quad_dict_gl.pickle".format(gamma_dict[gamma]), "rb") as openfile:
            quad_obj = pickle.load(openfile)
        nodes = quad_obj[n]['nodes'][:, None]
        weights = quad_obj[n]['weights']/(np.sqrt(np.pi))
        nodes, weights = torch.from_numpy(nodes).float().cuda(), torch.from_numpy(weights).float().cuda()
        nodes, weights = tensorproduct(nodes, weights, d)
        return nodes, weights


    if method == "orf":
        nodes  = genORF(n, d).cuda()
        weights= torch.ones(nodes.shape[0]).cuda()/nodes.shape[0]
        return nodes, weights
    
    if method == "qmc":
        samples = torch.from_numpy(sampler.random(n = n)).cuda().float()
        #samples = torch.from_numpy(dnb2.gen_samples(n)).cuda().float()
        nodes = dist.icdf(samples)
        weights = torch.ones(nodes.shape[0]).cuda()/nodes.shape[0]
        
        return nodes, weights

    if method == "rff":
        nodes = dist.rsample((n, d)).cuda()
        weights = torch.ones(nodes.shape[0]).cuda()/nodes.shape[0]

        return nodes, weights

    if method == "gl":  
        glnodes, glweights = np.polynomial.legendre.leggauss(2*n)
        keep_idx = np.where(glnodes >= 0)[0]
        glnodes, glweights = glnodes[keep_idx], 2*glweights[keep_idx]
        #glweights[0] = glweights[0]/2
        nodes, weights = torch.from_numpy(np.pi*glnodes).float().cuda()[:, None], np.pi*torch.from_numpy(glweights).float().cuda()
        n2, w2 = tensorproduct(nodes, weights, d, cosine = True)
        return n2, w2
    
    if method == "gh":
        ghnodes, ghweights = scipy.special.roots_hermite(2*n)
        keep_idx = np.where(ghnodes >= 0)[0]
        ghnodes, ghweights = ghnodes[keep_idx], 2*ghweights[keep_idx]
        #ghweights[0] = ghweights[0]/2
        nodes, weights = torch.from_numpy(ghnodes).float().cuda()[:, None], torch.from_numpy(ghweights).float().cuda()
        n2, w2 = tensorproduct(nodes, weights, d, cosine = True)

        return n2, w2

    if method == "trig":
        nodes = quad_obj[n]['nodes'][:, None]
        weights = quad_obj[n]['weights']
        nodes, weights = torch.from_numpy(nodes).float().cuda(), torch.from_numpy(weights).float().cuda()
        nodes, weights = tensorproduct(nodes, weights, d)
        return nodes, weights

def nll(Ytest, mu, var):
    Ytest = Ytest.cpu().numpy()
    nll = .5*np.log(2*np.pi*var) + np.square(Ytest - mu)/(2*var)
    return nll.mean()
     
def fit_method(Xtrain, Ytrain, Xtest, Ytest, method, n_nodes,iters = 500, lr = .0025, theta_init = .5, siginit = .25, update = 10, verbose = False, opt = True, svi = False, nlr = .1, batch = 1024, jitter = 1e-4,
              gl_trunc = np.pi, nu_init = 1., return_model = False, gamma = 1):
    gl = "gl" in method
    rff = method in ["rff", "qmc", "orf"]

    nodes, weights = nodes_weights(method, n_nodes, Xtrain.shape[1], gl_trunc = gl_trunc, gamma = gamma)
    print(nodes.shape)
    gp = quadGP(nodes, weights, num_c = 1, theta_init = theta_init, siginit = siginit, dtype = torch.float32, jitter = jitter, nu_init = nu_init)

    if method in ["gl", "trig"]:
        gp.gamma = gamma
    else:
        gp.gamma = 1
    
    if svi:
        gp.svi_fit(X = Xtrain[:, None, :], Y = Ytrain, N = Xtrain.shape[0], iters = iters, lr = lr, nlr = nlr, gl= gl, rff = rff, batch = batch)
        mu, var = gp.batchsvi_pred(Xtest = Xtest[:, None, :], Ytest = None, batch = 500, gl = gl, rff = rff)
    else:
        if opt:
            gp.fit(Xtrain[:, None, :], Ytrain, iters = iters, gl = gl, rff = rff,lr = lr, update = update)
        else:
            gp.d = Xtrain.shape[-1]
        mu, var = gp.bpred(Xtest[:, None, :], Xtrain[:, None, :], Ytrain, gl = gl, rff = rff)

    if return_model:
        return mu, var, gp
    if verbose:
        return mu, var
    
    rmse = np.square(Ytest.cpu().numpy() - mu).mean()**(.5)
    tnll = nll(Ytest, mu, var)
    mtheta = gp.theta.detach().cpu().numpy().mean()
    del gp
    torch.cuda.empty_cache()
   
    return rmse, tnll, mtheta



def fit_GP(Xtrain, Ytrain, Xtest, Ytest):

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims =Xtrain.shape[1]))
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(Xtrain, Ytrain, likelihood)
    model = model.cuda().double()


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Includes GaussianLikelihood parameters
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    epochs_iter = tqdm(range(1000), desc="Epoch")

    for i in epochs_iter:
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(Xtrain)
        # Calc loss and backprop gradients
        loss = -mll(output, Ytrain)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    f_preds = model(Xtest)
    y_preds = likelihood(model(Xtest))
    
    f_mean = y_preds.mean
    f_var = y_preds.variance

    return f_mean.detach().cpu().numpy(), f_var.detach().cpu().numpy()


def gpygp(Xtrain, Ytrain, Xtest, Ytest, init_theta = 1., iters = 1000):
    Xc = Xtrain.cpu().double().numpy()
    Yc = Ytrain.cpu().double().numpy()
    ker = gp.kern.RBF(input_dim=Xc.shape[1], ARD = False, lengthscale = init_theta) + gp.kern.White(input_dim=Xc.shape[1])
    gpy_mod = gp.models.GPRegression(Xc,Yc[:, None], ker)
    
    gpy_mod.optimize("lbfgs", start = None, max_iters = iters)
    
    Xtc = Xtest.cpu().numpy()
    Ytc = Ytest.cpu().numpy()
    
    mu2, var2 = gpy_mod.predict(Xtc)
    mu2 = mu2[:, 0]
    var2 = var2[:, 0]

    return mu2, var2

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = inducing_points.shape[1], poop = False))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_SVGP(Xtrain, Ytrain, Xtest, Ytest, n_m = 1100, ppgr = False, iters = 200, verbose = False, lr = .01):
    Xtrain, Ytrain, Xtest, Ytest = Xtrain.float(), Ytrain.float(), Xtest.float(), Ytest.float()
    
    
    
    train_dataset = TensorDataset(Xtrain.float(), Ytrain.float())
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


    Xnp = Xtrain.cpu().numpy()
    Xnp = Xnp[np.random.choice(Xnp.shape[0], np.min([Xnp.shape[0], 19999]), replace = False)]
    
    #kmeans = KMeans(n_clusters=n_m, random_state=0, n_init = 'auto').fit(Xnp)
    
    #inducing_points = torch.from_numpy(kmeans.cluster_centers_).float().cuda()
    inducing_points = Xtrain[:n_m]
    
    
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        
    

    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    #scheduler = MultiStepLR(optimizer, [225, 275], .25)

    # Our loss object. We're using the VariationalELBO
    #mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Ytrain.size(0))
    #mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Ytrain.size(0))
    
    if ppgr:
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=Ytrain.size(0))
    else:
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Ytrain.size(0))
    
    epochs_iter = tqdm(range(iters), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        #minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
        #scheduler.step()
        epochs_iter.set_postfix(loss=loss.item())
        #print(loss.item())
    
    
    model.eval()
    likelihood.eval()
    
    test_dataset = TensorDataset(Xtest.float(), Ytest.float())
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    
    mean_vec = torch.tensor([0.]).float()
    var_vec = torch.tensor([0.]).float()
    nll_vec = torch.tensor([0.]).float()
    for xbatch, ybatch in test_loader:
        preds = likelihood(model(xbatch.float()))
        mu= preds.mean.cpu()
        var =preds.variance.cpu()
        
        mean_vec = torch.cat([mean_vec, mu])
        var_vec = torch.cat([var_vec, var])
        #nll_vec = torch.cat([nll_vec, -preds.log_prob(ybatch).cpu()[None]])
    mean_vec=  mean_vec[1:].detach().cpu().numpy()
    var_vec = var_vec[1:].detach().cpu().numpy()
    nll_vec = nll_vec[1:].detach().cpu().numpy()
        
    return_nll = nll(Ytest.float().cpu(), mean_vec, var_vec).mean()
    return_rmse = ((Ytest.float().cpu().detach().numpy() - mean_vec)**2).mean()**(.5)
    
    del model
    del likelihood
    del Xtrain
    del Ytrain
    del Ytest
    del Xtest
    del test_dataset
    del test_loader
    torch.cuda.empty_cache()
    
    if verbose:
        return  mean_vec, var_vec
    else:
        return return_nll, return_rmse

    
    