
import math
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, ConstantKernel


class DirichletGPModel(ExactGP):
    def __init__(self, x_train, y_train, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )
        self.likelihood = likelihood

    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def fit(self, x_train, y_train, lr=0.1, training_steps=50, verbose=False):
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
        # Loss for GPs - The marginal log-likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.forward(x_train)
            loss = -mll(output, self.likelihood.transformed_targets).sum()
            loss.backward()
            if verbose and i % 10 == 0:
                print(f"Iter {i}/{training_steps} - Loss: {loss.item()} - lengthscale: {self.covar_module.base_kernel.lengthscale.mean().item()} - Noise: {self.likelihood.second_noise_covar.noise.mean().item()}")
            optimizer.step()


    def predict(self, x_test):
        self.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            test_dist = self.forward(x_test)
            pred_means = test_dist.loc
            pred_var = test_dist.variance

        return pred_means, pred_var

    
    def predict_proba(self, x_test, num_samples=2000):
        self.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            test_dist = self.forward(x_test)
            pred_samples = test_dist.sample(torch.Size((num_samples,))).exp()
            probs = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)

        return probs




class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, x_train, y_train, training_iter=50):
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.forward(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

    
    def predict(self, x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.forward(x)
        
        return pred


        


        


