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
        self.cluster_ratio = torch.ones(class_num) / class_num

        # Define the variational distribution and strategy of the GP
        # We will initialize the inducing points to lie on a grid from 0 to T
        inducing_points = torch.linspace(0, max_time, num_inducing).unsqueeze(-1)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([class_num])
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution),
            num_tasks=class_num
        )

        # Define model
        super().__init__(variational_strategy=variational_strategy)

        #self.mean_intensity = torch.nn.Parameter(torch.Tensor([ave_arrivals / max_time]))
        self.mean_intensity = ave_arrivals / max_time

        # Define mean and kernel
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([class_num]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([class_num]), lengthscale_prior=gpytorch.priors.GammaPrior(1, 2)),
            batch_shape=torch.Size([class_num]))

    def forward(self, times):
        mean = self.mean_module(times)
        covar = self.covar_module(times)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, arrival_times, quadrature_times, estimated_cluster=None):
        function_distribution = self.pyro_guide(torch.cat([arrival_times.data, quadrature_times], -1))

        # Draw samples from q(f) at arrival_times
        # Also draw samples from q(f) at evenly-spaced points (quadrature_times)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            pyro.sample(
                self.name_prefix + ".function_samples",
                function_distribution
            )

    def model(self, arrival_times, quadrature_times, estimated_cluster=None):
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
        cluster_log_likelihood = self.estimated_likelihood(arrival_times, function_samples, quadrature_times)

        log_likelihood = (cluster_log_likelihood * estimated_cluster.unsqueeze(1)).sum(dim=-1)

        pyro.factor(self.name_prefix + ".log_likelihood", log_likelihood.t())

    def expectation(self, arrival_times, quadrature_times):

        with torch.no_grad():
            function_samples = self(torch.cat([arrival_times.data, quadrature_times], -1))(torch.Size([1000]))
            # cluster_log_likelihood : (batch, particle, cluster)
            cluster_log_likelihood = self.estimated_likelihood(arrival_times, function_samples, quadrature_times)

            log_ratio = cluster_log_likelihood.mean(dim=1) + self.cluster_ratio.log().unsqueeze(0)
            # estimated_cluster : (batch, cluster)
            estimated_cluster = (log_ratio - torch.logsumexp(log_ratio, dim=1, keepdim=True)).exp()
            self.cluster_ratio = estimated_cluster.mean(dim=0)

        return estimated_cluster

    def estimated_likelihood(self, arrival_times, function_distribution, quadrature_times):
        intensity_samples = (function_distribution.exp() * self.mean_intensity).transpose(-2, -1)
        # Divide the intensity samples into arrival_intensity_samples and quadrature_intensity_samples
        arrival_intensity_samples, quadrature_intensity_samples = intensity_samples.split([
            arrival_times.data.size(-1), quadrature_times.size(-1)
        ], dim=-1)
        data_likelihood = torch.stack([each_data.log().sum(dim=-1)
                                       for each_data
                                       in arrival_intensity_samples.split(arrival_times.batch_sizes, dim=-1)],
                                      )
        ####
        # Compute the log_likelihood, using the method described above
        ####
        est_num_arrivals = quadrature_intensity_samples.mean(dim=-1).mul(self.max_time)
        if data_likelihood.ndim == 3:
            # data_likelihood : (batch, particle, cluster)
            # cluster_ratio : (batch, cluster)
            # est_num_arrivals : (particle, cluster)
            cluster_log_likelihood = data_likelihood - est_num_arrivals.unsqueeze(0)

        else:
            # data_likelihood : (batch, cluster)
            # cluster_ratio : (batch, cluster)
            # est_num_arrivals : (cluster)
            cluster_log_likelihood = (data_likelihood - est_num_arrivals.unsqueeze(0)).unsqueeze(1)

        return cluster_log_likelihood

