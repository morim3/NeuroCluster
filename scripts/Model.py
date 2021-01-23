import torch
import torch.nn
import gpytorch
import pyro

from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    IndependentMultitaskVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
import torch.nn.utils.rnn as rnn

class NeuroCluster(gpytorch.models.ApproximateGP):
    def __init__(self, ave_arrivals, max_time, class_num=4, num_inducing=32, name_prefix="cox_gp_model"):
        self.name_prefix = name_prefix
        self.max_time = max_time


        # Define the variational distribution and strategy of the GP
        # We will initialize the inducing points to lie on a grid from 0 to T
        inducing_points = torch.linspace(0, max_time, num_inducing).unsqueeze(-1)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing)
        variational_strategy = IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution),
            num_tasks=class_num
        )

        # Define model
        super().__init__(variational_strategy=variational_strategy)

        self.mean_intensity = torch.nn.Parameter(torch.Tensor([ave_arrivals / max_time]))

        # Define mean and kernel
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, times):
        mean = self.mean_module(times)
        covar = self.covar_module(times)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, arrival_times, quadrature_times):
        function_distribution = self.pyro_guide(torch.cat([arrival_times.data, quadrature_times], -1))

        # Draw samples from q(f) at arrival_times
        # Also draw samples from q(f) at evenly-spaced points (quadrature_times)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            pyro.sample(
                self.name_prefix + ".function_samples",
                function_distribution
            )

    def model(self, arrival_times, quadrature_times):
        pyro.module(self.name_prefix + ".gp", self)
        function_distribution = self.pyro_model(torch.cat([arrival_times.data, quadrature_times], -1))

        # Draw samples from p(f) at arrival times
        # Also draw samples from p(f) at evenly-spaced points (quadrature_times)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            function_samples = pyro.sample(
                self.name_prefix + ".function_samples",
                function_distribution
            )

        ####
        # Convert function samples into intensity samples, using the function above
        ####
        intensity_samples = (function_samples.exp() * self.mean_intensity).transpose(-2, -1)

        # Divide the intensity samples into arrival_intensity_samples and quadrature_intensity_samples

        arrival_intensity_samples, quadrature_intensity_samples = intensity_samples.split(np.cat([
            arrival_times.batch_sizes.tolist(), [quadrature_times.size(-1)]
        ]), dim=-1)

        ####
        # Compute the log_likelihood, using the method described above
        ####

        arrival_log_intensities = arrival_intensity_samples.log().sum(dim=-1)

        est_num_arrivals = quadrature_intensity_samples.mean(dim=-1).mul(self.max_time)

        log_likelihood = arrival_log_intensities - est_num_arrivals

        pyro.factor(self.name_prefix + ".log_likelihood", log_likelihood)
