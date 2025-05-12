import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np



def entry_stop_gradients(target, mask):
    mask_stop = tf.logical_not(mask)
    mask = tf.cast(mask, dtype=target.dtype)
    mask_stop = tf.cast(mask_stop, dtype=target.dtype)
    return tf.stop_gradient(mask_stop * target) + mask * target


class affine_coupling(layers.Layer):
    def __init__(self, name, n_split_at, n_width=32, flow_coupling=0, **kwargs):
        super(affine_coupling, self).__init__(name=name, **kwargs)
        self.n_split_at = n_split_at
        self.flow_coupling = flow_coupling
        self.n_width = n_width

    def build(self, input_shape):
        n_length = input_shape[-1]
        if self.flow_coupling == 0:
            self.f = NN2('a2b', self.n_width, n_length-self.n_split_at)
        elif self.flow_coupling == 1:
            self.f = NN2('a2b', self.n_width, (n_length-self.n_split_at)*2)
        else:
            raise Exception()
        self.log_gamma = self.add_weight(name='log_gamma', shape=(
            1, n_length-self.n_split_at), initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=True)

    def call(self, x, logdet=None):
        z = x
        n_split_at = self.n_split_at
        alpha = 0.6

        z1, z2 = z[:, :n_split_at], z[:, n_split_at:]

        if self.flow_coupling == 0:
            shift = self.f(z1)
            shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
            z2 += shift
        elif self.flow_coupling == 1:
            h = self.f(z1)
            shift = h[:, ::2]
            scale = alpha*tf.nn.tanh(h[:, 1::2])
            shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
            z2 = z2 + scale*z2 + shift
        else:
            raise ValueError(
                "Invalid flow_coupling value. Use 0 (additive) or 1 (affine).")
        if logdet is not None:
            logdet += tf.reduce_sum(tf.math.log(scale +
                                    tf.ones_like(scale)), axis=[1], keepdims=True)

        z = tf.concat([z1, z2], 1)
        if logdet is not None:
            return z, logdet

        return z

    def inverse(self, z, logdet=None):
        n_split_at = self.n_split_at
        z1, z2 = z[:, :n_split_at], z[:, n_split_at:]
        alpha = 0.6

        if self.flow_coupling == 0:
            shift = self.f(z1)
            shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
            z2 -= shift
        elif self.flow_coupling == 1:
            h = self.f(z1)
            shift = h[:, ::2]
            scale = alpha*tf.nn.tanh(h[:, 1::2])
            shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
            z2 = (z2 - shift) / (tf.ones_like(scale) + scale)
        else:
            raise ValueError(
                "Invalid flow_coupling value. Use 0 (additive) or 1 (affine).")
        if logdet is not None:
            logdet -= tf.reduce_sum(tf.math.log(scale +
                                    tf.ones_like(scale)), axis=[1], keepdims=True)

        z = tf.concat([z1, z2], 1)
        if logdet is not None:
            return z, logdet

        return z


class NN2(layers.Layer):
    def __init__(self, name, n_width=32, n_out=None, **kwargs):
        super(NN2, self).__init__(name=name, **kwargs)
        self.n_width = n_width
        self.n_out = n_out

    def build(self, input_shape):
        self.l_1 = layers.Dense(units=self.n_width, activation=None, name='h1')
        self.l_2 = layers.Dense(units=self.n_width, activation=None, name='h2')
        n_out = self.n_out or int(input_shape[-1])
        self.l_f = layers.Dense(units=n_out, activation=None, name='last')

    def call(self, inputs):
        x = self.l_1(inputs)
        x = tf.nn.relu(x)
        x = self.l_2(x)
        x = tf.nn.relu(x)
        x = self.l_f(x)
        # #Option for Tanh with high regularity (commented out)
        # x = tf.nn.tanh(self.l_1(inputs))
        # x = tf.nn.tanh(self.l_2(x))
        return x


class NN2v(layers.Layer):
    def __init__(self, name, n_width=32, n_out=None, **kwargs):
        super(NN2v, self).__init__(name=name, **kwargs)
        self.n_width = n_width
        self.n_out = n_out

    def build(self, input_shape):
        self.l_1 = layers.Dense(units=self.n_width, activation=None, name='h1')
        self.l_2 = layers.Dense(units=self.n_width//2,
                                activation=None, name='h2')
        self.l_3 = layers.Dense(units=self.n_width//2,
                                activation=None, name='h3')
        self.l_4 = layers.Dense(units=self.n_width, activation=None, name='h4')
        n_out = self.n_out or int(input_shape[-1])
        self.l_f = layers.Dense(units=n_out, activation=None, name='last')

    def call(self, inputs):
        x = tf.nn.relu(self.l_1(inputs))
        x = tf.nn.relu(self.l_2(x))
        x = tf.nn.relu(self.l_3(x))
        x = tf.nn.relu(self.l_4(x))
        x = self.l_f(x)
        return x


class squeezing(layers.Layer):
    def __init__(self, name, n_dim, n_cut=1, **kwargs):
        super(squeezing, self).__init__(name=name, **kwargs)
        self.n_dim = n_dim
        self.n_cut = n_cut
        self.x = None

    def call(self, inputs):
        z = inputs
        n_length = z.get_shape()[-1]

        if self.n_length < self.n_cut and not self.x:
            raise Exception()
        if self.n_dim == n_length:
            if n_length > self.n_cut:
                if self.x:
                    raise Exception()
                else:
                    self.x = z[:, n_length - self.n_cut:]
                    z = z[:, :n_length - self.n_cut]
            else:
                self.x = None
        elif self.n_length <= self.n_cut:
            z = tf.concat([z, self.x], 1)
            self.x = None
        else:
            cut = z[:, n_length - self.n_cut:]
            self.x = tf.concat([cut, self.x], 1)
            z = z[:, :n_length - self.n_cut]
        return z

    def inverse(self, inputs):
        z = inputs
        n_length = z.get_shape()[-1]

        if self.n_dim == n_length:
            n_start = self.n_dim % self.n_cut
            n_start += self.n_cut if n_start == 0 else 0
            self.x = z[:, n_start:]
            z = z[:, :n_start]
        else:
            x_length = self.x.get_shape()[-1]
            if x_length < self.n_cut:
                raise Exception()
            cut = self.x[:, :self.n_cut]
            z = tf.concat([z, cut], 1)
            if x_length - self.n_cut == 0:
                self.x = None
            else:
                self.x = self.x[:, self.n_cut:]
        return z


class squeezing2(layers.Layer):
    def __init__(self, name, n_dim, n_cut=1, **kwargs):
        super(squeezing2, self).__init__(name=name, **kwargs)
        self.n_dim = n_dim
        self.n_cut = n_cut
        self.x = None

    def call(self, inputs):
        z = inputs
        # print(z.get_shape())
        n_length = z.get_shape()[-1]

        if n_length < self.n_cut and not self.x:
            raise Exception()

        if self.n_dim == n_length:
            if n_length > 2*self.n_cut:
                if self.x != None:
                    raise Exception()
                else:
                    self.x = z[:, n_length - self.n_cut:]
                    z = z[:, :n_length - self.n_cut]
            else:
                self.x = None
        elif n_length <= 2*self.n_cut:
            z = tf.concat([z, self.x], 1)
            self.x = None
        else:
            cut = z[:, n_length - self.n_cut:]
            self.x = tf.concat([cut, self.x], 1)
            z = z[:, :n_length - self.n_cut]
        return z

    def inverse(self, inputs):
        z = inputs
        n_length = z.get_shape()[-1]
        if self.n_dim == n_length:
            n_start = self.n_dim % self.n_cut
            if n_start == 0:
                n_start += self.n_cut
            self.x = z[:, n_start:]
            z = z[:, :n_start]

        x_length = self.x.get_shape()[-1]
        if x_length < self.n_cut:
            raise Exception()
        cut = self.x[:, :self.n_cut]
        z = tf.concat([z, cut], 1)
        if x_length - self.n_cut == 0:
            self.x = None
        else:
            self.x = self.x[:, self.n_cut:]
        return z


class W_LU(layers.Layer):
    def __init__(self, name, **kwargs):
        super(W_LU, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.n_length = input_shape[-1]
        self.LU = self.add_weight(name='LU', shape=(self.n_length, self.n_length),initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=True)
        self.LU_init = self.add_weight(name="LU_init", shape=(
            self.n_length, self.n_length), initializer=tf.keras.initializers.Identity(), trainable=False, dtype=tf.float32)

    def call(self, inputs, logdet=None, reverse=False):
        x = inputs
        n_dim = x.shape[-1]
        LU = self.LU_init + self.LU

        U = tf.linalg.band_part(LU, 0, -1)
        U_diag = tf.linalg.tensor_diag_part(U)
        U_mask = (tf.linalg.band_part(tf.ones([n_dim, n_dim]), 0, -1) >= 1)
        U = entry_stop_gradients(U, U_mask)

        I = tf.eye(self.n_length, dtype=tf.float32)
        L = tf.linalg.band_part(I+LU, -1, 0)-tf.linalg.band_part(LU, 0, 0)
        L_mask = (tf.linalg.band_part(tf.ones(
            [n_dim, n_dim]), -1, 0) - tf.linalg.band_part(tf.ones([n_dim, n_dim]), 0, 0) >= 1)
        L = entry_stop_gradients(L, L_mask)

        if not reverse:
            x = tf.transpose(x)
            x = tf.linalg.matmul(U, x)
            x = tf.linalg.matmul(L, x)
            x = tf.transpose(x)
        else:
            x = tf.transpose(x)
            x = tf.linalg.matmul(tf.linalg.inv(L), x)
            x = tf.linalg.matmul(tf.linalg.inv(U), x)
            x = tf.transpose(x)

        if logdet is not None:
            dlogdet = tf.reduce_sum(tf.math.log(tf.math.abs(U_diag)))
            if reverse:
                dlogdet *= -1.0
            return x, logdet + dlogdet
        return x


class flow_mapping(layers.Layer):
    def __init__(self, name, n_depth, n_split_at, n_width=32, flow_coupling=0, n_bins=16, **kwargs):
        super(flow_mapping, self).__init__(name=name, **kwargs)
        self.n_depth = n_depth
        self.n_split_at = n_split_at
        self.n_width = n_width
        self.flow_coupling = flow_coupling
        self.n_bins = n_bins
        assert n_depth % 2 == 0

    def build(self, input_shape):
        self.n_length = input_shape[-1]
        self.affine_layers = []
        self.scale_layers = []

        sign = -1
        for i in range(self.n_depth):
            self.scale_layers.append(actnorm('actnorm'+str(i)))
            sign *= -1
            i_split_at = (self.n_split_at*sign + self.n_length) % self.n_length
            self.affine_layers.append(affine_coupling('af_coupling_' + str(i),i_split_at,n_width=self.n_width,flow_coupling=self.flow_coupling))
            # if self.n_bins > 0:
            #   self.cdf_layer = CDF_quadratic('cdf_layer', self.n_bins)

    def call(self, inputs, logdet=None, reverse=False):
        z = inputs
        if not reverse:
            for i in range(self.n_depth):
                z = self.scale_layers[i](z,logdet)
                if logdet is not None:
                    z, logdet = z
                
                z = self.affine_layers[i](z, logdet)
                if logdet is not None:
                    z, logdet = z
                z = z[:, ::-1]
        else:
            for i in reversed(range(self.n_depth)):
                z = z[:, ::-1]
                z = self.affine_layers[i].inverse(z, logdet)
                if logdet is not None:
                    z, logdet = z
                
                z = self.scale_layers[i](z, logdet, reverse=True)
                if logdet is not None:
                    z, logdet = z
        if logdet is not None:
            return z, logdet
        return z
    def actnorm_data_initialization(self):
        for i in range(self.n_depth):
            self.scale_layers[i].reset_data_initialization()

class actnorm(layers.Layer):
    def __init__(self,name,scale = 1.0,logscale_factor = 3.0,**kwargs):
        super(actnorm, self).__init__(name=name,**kwargs)
        self.scale = scale
        self.logscale_factor = logscale_factor
        self.data_init = True
    def build(self, input_shape):
        self.n_length = input_shape[-1]
        self.b = self.add_weight(name='b', shape=(1, self.n_length),initializer=tf.zeros_initializer(),dtype=tf.float32, trainable=True)
        self.b_init = self.add_weight(name='b_init', shape=(1, self.n_length),initializer=tf.zeros_initializer(),dtype=tf.float32, trainable=False)
        self.logs  = self.add_weight(name='logs', shape=(1, self.n_length),initializer=tf.zeros_initializer(),dtype=tf.float32, trainable=True)
        self.logs_init = self.add_weight(name='logs_init', shape=(1, self.n_length),initializer=tf.zeros_initializer(),dtype=tf.float32, trainable=False)
        
    def call(self, inputs, logdet = None, reverse = False):
        # data initialization
        # # by default, no data initialization is implemented.
        if not self.data_init:
            x_mean = tf.reduce_mean(inputs, [0], keepdims=True)
            x_var = tf.reduce_mean(tf.square(inputs-x_mean), [0], keepdims=True)
            
            self.b_init.assign(-x_mean)
            self.logs_init.assign(tf.math.log(self.scale/(tf.sqrt(x_var)+1e-6))/self.logscale_factor)
            
            self.data_init = True
        if not reverse:
            x = inputs + (self.b + self.b_init)
            x = x * tf.exp(self.logs + self.logs_init)
        else:
            x = inputs * tf.exp(-self.logs - self.logs_init)
            x = x - (self.b + self.b_init)
        
        if logdet is not None:
            dlogdet = tf.reduce_sum(self.logs + self.logs_init)
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet
        
        return x
    def reset_data_initialization(self):
        self.data_init = False



class krnet(tf.keras.Model):
    def __init__(self, name, n_dim, n_step, n_depth, n_width=32, shrink_rate=1.0, flow_coupling=0, n_bins=16, rotation=False, **kwargs):
        super(krnet, self).__init__(name=name, **kwargs)
        assert n_depth % 2 == 0
        # tfd = tfp.distributions
        # prior = tfd.MultivariateNormalDiag(loc=tf.zeros(n_dim),scale_diag=tf.ones(n_dim))

        self.n_dim = n_dim
        self.Max = tf.convert_to_tensor(np.array([[-3,0],[0.00,0],[3,0]]),dtype=tf.float32)
        self.n_step = n_step
        self.n_depth = n_depth
        self.n_width = n_width
        self.n_bins = n_bins
        self.shrink_rate = shrink_rate
        self.flow_coupling = flow_coupling
        self.rotation = rotation
        ######
        self.weights_1 = [0.5, 0.2,0.3]
        self.means_1 = [-3, 0, 3]
        self.stds_1 = [1.0, 0.5, 0.8]

        ######

        self.n_stage = n_dim // n_step
        if n_dim % n_step == 0:
            self.n_stage -= 1
        self.n_rotation = self.n_stage

        if rotation:
            self.rotations = []
            for i in range(self.n_rotation):
                self.rotations.append(W_LU('rotation'+str(i)))

        self.flow_mappings = []
        for i in range(self.n_stage):
            n_split_at = n_dim - (i+1) * n_step
            self.flow_mappings.append(flow_mapping('flow_mapping'+str(i),
                                                   n_depth,
                                                   n_split_at,
                                                   n_width=n_width,
                                                   flow_coupling=flow_coupling,
                                                   n_bins=n_bins))
            n_width = int(n_width*self.shrink_rate)

        self.squeezing_layer = squeezing2('squeezing', n_dim, n_step)
        # self.log_prior = self.prior.log_prob

    def call(self, inputs):
        # print('call invoke')
        objective = tf.zeros_like(inputs, dtype='float32')[:, 0]
        objective = tf.reshape(objective, [-1, 1])
        z, objective = self.mapping_to_prior(inputs, objective)
        objective += tf.reshape(self.compute_prior_log_prob(z), [-1, 1])
        return objective

    def compute_prior_log_prob(self, samples):
        """
        Compute the log probability under the mixture Gaussian prior.
        :param samples: Tensor of shape (n_samples, n_dim).
        :return: Tensor of log probabilities of shape (n_samples,).
        """
        dim_1 = samples[:, 0]
        # dim_2 = samples[:, 1]
        # dim_3 = samples[:, 2]
        gaussian_dims = samples[:, 1:]

        # Compute mixture Gaussian log probabilities for first three dimensions
        log_pdf_1 = self._compute_mixture_pdf(
            dim_1, self.weights_1, self.means_1, self.stds_1)
        # log_pdf_2 = self._compute_mixture_pdf(
        #     dim_2, self.weights_2, self.means_2, self.stds_2)
        # log_pdf_3 = self._compute_mixture_pdf(
        #     dim_3, self.weights_3, self.means_3, self.stds_3)

        # Compute Gaussian log PDF for remaining dimensions
        log_pdf_gaussian = tf.reduce_sum(-0.5 * gaussian_dims **
                                         2 - tf.math.log(tf.sqrt(2 * np.pi)), axis=1)

        # Combine log probabilities
        log_prior = log_pdf_1 + log_pdf_gaussian #+ log_pdf_2 + log_pdf_3
        return log_prior

    def _compute_mixture_pdf(self, samples, weights, means, stds):
        """
        Compute the log PDF for a mixture of Gaussians in a numerically stable way.
        :param samples: Input tensor of shape (n_samples,).
        :param weights: List or Tensor of mixture weights.
        :param means: List or Tensor of means for each Gaussian component.
        :param stds: List or Tensor of standard deviations for each Gaussian component.
        :return: Log PDF of the mixture for each sample.
        """
        weights = tf.constant(weights, dtype=tf.float32)
        means = tf.constant(means, dtype=tf.float32)
        stds = tf.constant(stds, dtype=tf.float32)

        # Expand dimensions to match samples shape for broadcasting
        samples = tf.expand_dims(samples, axis=-1)  # Shape: (n_samples, 1)
        means = tf.expand_dims(means, axis=0)       # Shape: (1, n_components)
        stds = tf.expand_dims(stds, axis=0)         # Shape: (1, n_components)

        # Compute log probabilities for each component
        log_probs = -0.5 * ((samples - means) / stds) ** 2 - \
            tf.math.log(stds * tf.sqrt(2 * np.pi))
        log_probs += tf.math.log(weights)  # Add log weights

        # Combine log probabilities using log-sum-exp for numerical stability
        return tf.reduce_logsumexp(log_probs, axis=-1)  # Shape: (n_samples,)

    def mapping_to_prior(self, inputs, logdet=None):
        z = inputs
        # print("Mapping to prior, input shape:", z.shape)
        for i in range(self.n_stage):
            # print(f"Stage {i}, input shape before squeezing: {z.shape}")
            if logdet is not None:
                if self.rotation and i < self.n_rotation:
                    z, logdet = self.rotations[i](z, logdet)
                    # print(f"Applied rotation {i}, output shape: {z.shape}")
                z, logdet = self.flow_mappings[i](z, logdet)
                # print(f"Applied flow mapping {i}, output shape: {z.shape}")
            else:
                if self.rotation and i < self.n_rotation:
                    z = self.rotations[i](z)
                z = self.flow_mappings[i](z)
            z = self.squeezing_layer(z)
        # print("Final shape after mapping to prior:", z.shape)
        if logdet is not None:
            return z, logdet
        return z
    
    def actnorm_data_initialization(self):
        for i in range(self.n_stage):
            self.flow_mappings[i].actnorm_data_initialization()

    def mapping_from_prior(self, inputs):
        z = inputs
        for i in reversed(range(self.n_stage)):
            z = self.squeezing_layer.inverse(z)
            z = self.flow_mappings[i](z, reverse=True)
            if self.rotation and i < self.n_rotation:
                z = self.rotations[i](z, reverse=True)
        return z

    def draw_samples_from_prior(self, n_samples):
        """
        Generate samples from the prior distribution.
        :param n_samples: Number of samples to generate.
        :return: TensorFlow tensor of shape (n_samples, n_dim).
        """
        # First 3 dimensions: Mixture Gaussians
        dim_1 = self._generate_mixture_samples(
            n_samples, self.weights_1, self.means_1, self.stds_1)
        # dim_2 = self._generate_mixture_samples(
        #     n_samples, self.weights_2, self.means_2, self.stds_2)
        # dim_3 = self._generate_mixture_samples(
        #     n_samples, self.weights_3, self.means_3, self.stds_3)

        # Last (n_dim - 3) dimensions: Standard Gaussian
        gaussian_dims = tf.random.normal(
            shape=(n_samples, self.n_dim - 1), mean=0.0, stddev=1.0)

        # Combine all dimensions into a single tensor of shape (n_samples, n_dim)
        random_variable = tf.concat([
            tf.expand_dims(dim_1, axis=-1),  # Shape: (n_samples, 1)
            gaussian_dims                     # Shape: (n_samples, n_dim - 3)
        ], axis=1)

        return self.mapping_from_prior(random_variable)

    def _generate_mixture_samples(self, n_samples, weights, means, stds):
        """
        Generate samples from a 1D mixture of Gaussians.
        :param n_samples: Number of samples to generate.
        :param weights: List of mixture weights.
        :param means: List of means for each Gaussian component.
        :param stds: List of standard deviations for each Gaussian component.
        :return: TensorFlow tensor of shape (n_samples,).
        """
        # Convert weights, means, and stds to TensorFlow tensors
        weights = tf.constant(weights, dtype=tf.float32)
        means = tf.constant(means, dtype=tf.float32)
        stds = tf.constant(stds, dtype=tf.float32)

        # Number of components in the mixture
        n_components = tf.shape(weights)[0]

        # Sample component indices based on the mixture weights
        components = tf.random.categorical(tf.math.log(
            [weights]), n_samples)  # Shape: (1, n_samples)
        components = tf.reshape(components, [-1])  # Shape: (n_samples,)

        # Gather means and standard deviations for the sampled components
        sampled_means = tf.gather(means, components)  # Shape: (n_samples,)
        sampled_stds = tf.gather(stds, components)    # Shape: (n_samples,)

        # Sample from the corresponding Gaussian components
        samples = tf.random.normal(shape=(
            n_samples,), mean=sampled_means, stddev=sampled_stds)  # Shape: (n_samples,)

        return samples

    # def draw_samples_from_prior(self, n_samples):
    #   z = self.prior.sample((n_samples,))
    #   x = self.mapping_from_prior(z)
    #   return x




def draw_samples_from_prior(n_samples):
        """
        Generate samples from the prior distribution.
        :param n_samples: Number of samples to generate.
        :return: TensorFlow tensor of shape (n_samples, n_dim).
        """
        # First 3 dimensions: Mixture Gaussians
        dim_1 = flow._generate_mixture_samples(
            n_samples, flow.weights_1, flow.means_1, flow.stds_1)
    
        # Last (n_dim - 3) dimensions: Standard Gaussian
        gaussian_dims = tf.random.normal(
            shape=(n_samples, flow.n_dim - 1), mean=0.0, stddev=1)

        # Combine all dimensions into a single tensor of shape (n_samples, n_dim)
        random_variable = tf.concat([
            tf.expand_dims(dim_1, axis=-1),  # Shape: (n_samples, 1)
            gaussian_dims
        ], axis=1)

        return random_variable





flow = krnet('krnet',2,1,32,n_width=128,n_bins= 0,shrink_rate=1,flow_coupling=1,rotation=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
dummy_input = tf.random.normal((1, 2))  
flow(dummy_input) 

flow.actnorm_data_initialization()
_ = flow(dummy_input)

checkpoint = tf.train.Checkpoint(optimizer=optimizer,net = flow )
checkpoint_dir = './checkpoints'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

if latest_checkpoint:
    checkpoint.restore(latest_checkpoint).expect_partial()  
    print(optimizer.learning_rate.numpy())
    print(f"Restored checkpoint from {latest_checkpoint}")
else:
    print("No checkpoint found. Starting from scratch.")
    flow.actnorm_data_initialization()

# Training loop

penalty = tf.Variable(0.01, dtype=tf.float32)
penalty_1 = tf.Variable(0.001, dtype=tf.float32)

@tf.function
def train_step(x_data, lambda_margin=0, lambda_grad=0.001):
    """
    Performs one step of training:
    - Computes the negative log-likelihood loss
    - Applies a margin penalty to enforce logP_max > logP_neighbors
    - Enforces gradient penalty to make g_Max a local maximum
    """
    # Dynamically compute g_Max from flow.max using mapping_from_prior
    g_Max = flow.mapping_from_prior(flow.Max)
    with tf.GradientTape() as tape:
        # Compute log PDF for the batch data
        log_pdf = flow(x_data)
        loss = -tf.reduce_mean(log_pdf)  # Negative log-likelihood loss

        # Compute logP at nearby perturbed points (to enforce local max constraint)
        g_Max_expanded = tf.expand_dims(g_Max, axis=0)
        g_Max_tiled = tf.tile(g_Max_expanded, [1000, 1, 1])
        noise = tf.random.normal(shape=tf.shape(g_Max_tiled), mean=0.0, stddev=0.05)
        g_neighbour = g_Max_tiled + noise
        g_neighbour = tf.reshape(g_neighbour, [-1, 2])
        logP_max = flow(tf.reshape(g_Max_tiled,[-1,2]))
        logP_neighbour = flow(g_neighbour)
    
        
        penalty_loss = tf.reduce_sum(tf.maximum(0.0,logP_neighbour - logP_max))

        #Repulsive Loss
        x_max = flow.mapping_from_prior(flow.Max)  # Map maxima in z-space to x-space
        dists = tf.norm(x_max[:, None] - x_max[None, :], axis=-1)
        mask = tf.linalg.band_part(tf.ones_like(dists), 0, -1) - tf.eye(tf.shape(x_max)[0])
        repulsion_loss = tf.reduce_sum(tf.maximum(0.0, 0.8 - dists)**2 * mask)

        # Gradient penalty: Ensure ∇logP(g_Max) ≈ 0
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(g_Max)
            logP_max_value = flow(g_Max)
        grad_logP = grad_tape.gradient(logP_max_value, g_Max)
        # grad_penalty = tf.reduce_sum(tf.abs(grad_logP))  # Penalize nonzero gradients
        grad_penalty = tf.reduce_sum(tf.abs(grad_logP) + 1e-5)

        # Total loss (negative log-likelihood + penalties)
        total_loss = loss + lambda_margin * penalty_loss + lambda_grad * grad_penalty + repulsion_loss

    # Compute gradients and apply optimizer updates
    gradients = tape.gradient(total_loss, flow.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 0.01)
    optimizer.apply_gradients(zip(gradients, flow.trainable_variables))

    return loss, penalty_loss, grad_penalty,repulsion_loss

for t in range(100001):
    samples = np.load('./data/set_'+str(int(t%20))+'.npy')
    samples = tf.convert_to_tensor(samples, dtype=tf.float32)

    # Perform one training step
    loss, penalty_loss, grad_penalty_loss,repulsion_loss = train_step(samples,lambda_margin=penalty_1,lambda_grad=penalty)

    # Print progress
    if t % 500 == 0:
        print(f"Iteration {t}: Loss = {loss.numpy():.3f}, Penalty = {penalty_loss.numpy():.3f}, Grad Loss = {grad_penalty_loss.numpy():.3f}, Repulsion_loss = {repulsion_loss.numpy():.3f}")
    if t and t%5000 == 0:
        manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoints', max_to_keep=5)
        manager.save()
    if t%5000 == 0:
        new_lr = optimizer.learning_rate.numpy() * 0.9
        
        new_penalty = penalty.numpy()*1.05
        penalty.assign(new_penalty)
        
        new_penalty_1 = penalty_1.numpy()*1.05
        penalty_1.assign(new_penalty_1)
        optimizer.learning_rate.assign(new_lr)











