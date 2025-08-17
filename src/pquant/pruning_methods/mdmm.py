# -*- coding: utf-8 -*-
# @Author: Arghya Ranjan Das
# file: src/pquant/pruning_methods/mdmm.py
# modified by:


import keras
from keras import ops
import abc

@ops.custom_gradient    
def flip_gradient(x, scale=-1.0):
    def grad(*args, upstream=None):
        if upstream is None:     
            (upstream,) = args
        scale_t = ops.convert_to_tensor(scale, dtype=upstream.dtype)
        return (ops.multiply(upstream, scale_t),) # ops.abs()

    return x, grad


# Abstract base class for constraints
@keras.saving.register_keras_serializable(package="pquant.pruning_methods", name="Constraint")
class Constraint(keras.layers.Layer):
    def __init__(self, lmbda_init=0.0,scale=1.0, damping=1.0, **kwargs):
        self.use_grad_ = bool(kwargs.pop("use_grad", True))
        self.lr_ = float(kwargs.pop("lr", 0.0))
        super().__init__(**kwargs)
        
        
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(scale, dtype=dtype),
            trainable=False
        )
        self.damping = self.add_weight(
            name='damping',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(damping, dtype=dtype),
            trainable=False
        )
        self.lmbda = self.add_weight(
            name=f'{self.name}_lmbda',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(lmbda_init, dtype=dtype),
            trainable=self.use_grad_
        )
        
        if not self.use_grad_:
            self.prev_infs = self.add_weight(
                name=f'{self.name}_prev_infs',
                shape=(),
                initializer=lambda shape, dtype: ops.convert_to_tensor(0.0, dtype=dtype),
                trainable=False
            )

    def call(self, weight):
        """Calculates the penalty from a given infeasibility measure."""
        raw_infeasibility = self.get_infeasibility(weight)
        infeasibility = self.pipe_infeasibility(raw_infeasibility)
       
        if self.use_grad_:
            ascent_lmbda = flip_gradient(self.lmbda)
            # ascent_lmbda = ops.maximum(ascent_lmbda, 0.0)
        else:
            lmbda_step = self.lr_ * self.scale * self.prev_infs
            ascent_lmbda = self.lmbda + lmbda_step
            self.lmbda.assign_add(lmbda_step)
            self.prev_infs.assign(infeasibility)
        
        l_term =  ascent_lmbda * infeasibility
        damp_term = self.damping * ops.square(infeasibility) / 2
        penalty = self.scale * (l_term + damp_term)
        
        return penalty

    @abc.abstractmethod
    def get_infeasibility(self, weight):
        """Must be implemented by subclasses to define the violation."""
        raise NotImplementedError
    
    def pipe_infeasibility(self, infeasibility):
        """Optional transformation of raw infeasibility.
        Default is identity. Subclasses may override."""
        return infeasibility

    def turn_off(self):
        if not self.use_grad_:
            self.lr_ = 0.0
        self.scale.assign(0.0)
        self.lmbda.assign(0.0)
        
#-------------------------------------------------------------------
#               Generic Constraint Classes
#-------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="pquant.pruning_methods", name="EqualityConstraint")
class EqualityConstraint(Constraint):
    """Constraint for g(w) == target_value."""
    def __init__(self, metric_fn, target_value = 0.0,**kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value
        
    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = metric_value - self.target_value
        # return ops.abs(infeasibility)
        return infeasibility
    
@keras.saving.register_keras_serializable(package="pquant.pruning_methods", name="LessThanOrEqualConstraint")
class LessThanOrEqualConstraint(Constraint):
    """Constraint for g(w) <= target_value."""
    def __init__(self, metric_fn, target_value = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value
        
    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = metric_value - self.target_value
        return ops.maximum(infeasibility, 0.0)
    
@keras.saving.register_keras_serializable(package="pquant.pruning_methods", name="GreaterThanOrEqualConstraint")
class GreaterThanOrEqualConstraint(Constraint):
    """Constraint for g(w) >= target_value."""
    def __init__(self, metric_fn, target_value = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value
        
    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = self.target_value - metric_value
        return ops.maximum(infeasibility, 0.0)
    
#-------------------------------------------------------------------
#                   Metric Functions
#-------------------------------------------------------------------

class UnstructuredSparsityMetric:
    """L0-L1 based metric """
    """Calculates the ratio of non-zero weights in a tensor."""
    def __init__(self, l0_mode='coarse', scale_mode = "mean", epsilon=1e-3, target_sparsity=0.8, alpha=100.0):
        # Note: scale_mode:"sum" give very high losses for large model
        assert l0_mode in ['coarse', 'smooth'], "Mode must be 'coarse' or 'smooth'"
        assert scale_mode in ['sum', 'mean'], "Scale mode must be 'sum' or 'mean'"
        assert 0 <= target_sparsity <= 1, "target_sparsity must be between 0 and 1"
        self.l0_mode = l0_mode
        self.scale_mode = scale_mode
        self.target_sparsity = float(target_sparsity)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        
        self.l0_fn = None  
        self._scaling = None
        
        self.build()
        
    def build(self):
        # l0 term -> number of zero weights/number of weights
        if self.l0_mode == 'coarse':
            self.l0_fn = self._coarse_l0
        elif self.l0_mode == 'smooth':
            self.l0_fn = self._smooth_l0
        
        if self.scale_mode == 'mean':
            self._scaling = self._mean_scaling
        elif self.scale_mode == 'sum':
            self._scaling = self._sum_scaling
    
    def _sum_scaling(self, fn_value, num):
        return fn_value
    
    def _mean_scaling(self, fn_value, num):
        return fn_value / num

    def _coarse_l0(self, weight_vector):
        return ops.mean(ops.cast(ops.abs(weight_vector) <= self.epsilon, "float32"))

    def _smooth_l0(self, weight_vector):
        """Differentiable approximation of L0 norm using Keras ops."""
        return ops.mean(ops.exp(-self.alpha * ops.square(weight_vector)))

    def __call__(self, weight):
        num_weights = ops.cast(ops.size(weight), weight.dtype)
        weights_vector = ops.reshape(weight, [-1])

        l0_term = self.l0_fn(weights_vector)
        l1_term = ops.sum(ops.abs(weights_vector))
        
        # farctor by constrction goes to zero when l0_term == target_sparsiity
        factor = ops.square(self.target_sparsity) - ops.square(l0_term)
        fn_value = factor * l1_term
        fn_value = self._scaling(fn_value, num_weights)
            
        return fn_value

class FPGAAwareSparsityMetric:
    def __init__(self, rf=1,  precision=16, target_resource='DSP', 
                 bram_width=36, epsilon=1e-3):
        assert target_resource in ['DSP', 'BRAM'], "target_resource must be 'DSP' or 'BRAM'."
        self.rf = rf
        self.precision = precision
        self.target_resource = target_resource
        self.bram_width = bram_width
        self.epsilon = epsilon
        
        self.c = self._calculate_c()

        
    def _calculate_c(self):
        """Calculates 'C', the number of consecutive DSP groups that are packed into a single BRAM block"""
        if self.bram_width % self.precision == 0:
            return self.bram_width // self.precision
        else:
            return (2 * self.bram_width) // self.precision
        
    def _prepare_weights(self, weight):
        """
        Reshapes and pads the weight tensor to align with the Reuse Factor (RF).
        => Makes the tensor divisible into DSP-sized groups.
        """
        original_shape = weight.shape
        # For mulit-dim lyers (eg. Conv2D) => Flatten
        if len(original_shape) > 2:
            weight_reshaped = ops.reshape(weight, (original_shape[0], -1))
        else:
            weight_reshaped = weight
            
        num_weights = ops.shape(weight_reshaped)[1]
        padding_needed = (self.rf - num_weights % self.rf) % self.rf
        weight_padded = ops.pad(weight_reshaped, [[0, 0], [0, padding_needed]])

        return weight_padded
    
        
    def __call__(self, weight):
        
        prepared_weights = self._prepare_weights(weight)
        dsp_groups = ops.reshape(prepared_weights, (prepared_weights.shape[0], -1, self.rf))
        
        if self.target_resource == 'DSP':
            return self._calculate_dsp_sparsity(dsp_groups)
        elif self.target_resource == 'BRAM':
            return self._calculate_bram_sparsity(dsp_groups)

    def _calculate_dsp_sparsity(self, dsp_groups):
        """
        Calculates sparsity at the DSP level.

        A DSP block is considered "pruned" if the L2-norm of the weight group
        it processes is below the 'epsilon' threshold.
        """
        group_norms = ops.sqrt(ops.sum(ops.square(dsp_groups), axis=-1)) # Calculate the L2 norm for each DSP group
        zero_groups = ops.less_equal(group_norms, self.epsilon) # Identify which groups are effectively zero
        num_groups = ops.cast(ops.size(group_norms), "float32")

        # Return the fraction of pruned groups
        return ops.sum(ops.cast(zero_groups, "float32")) / num_groups # TODO Align with some target

    def _calculate_bram_sparsity(self, dsp_groups):
        """
        Calculates sparsity at the BRAM level.

        This involves further grouping the DSP-level structures into BRAM-sized
        chunks before calculating the norm. A BRAM block is "pruned" if the norm
        of all weights stored within it is below the 'epsilon' threshold.
        """
        # Further group the DSP structures into BRAM-sized chunks
        num_dsp_groups = ops.shape(dsp_groups)[1]
        bram_padding = (self.c - num_dsp_groups % self.c) % self.c
        dsp_groups_padded = ops.pad(dsp_groups, [[0, 0], [0, bram_padding], [0, 0]])
        bram_groups = ops.reshape(dsp_groups_padded, (dsp_groups.shape[0], -1, self.c, self.rf))

        # Calculate the L2 norm for each BRAM group
        bram_group_norms = ops.sqrt(ops.sum(ops.square(bram_groups), axis=(-1, -2)))
        zero_bram_groups = ops.less_equal(bram_group_norms, self.epsilon)
        num_bram_groups = ops.cast(ops.size(bram_group_norms), "float32")
        return ops.sum(ops.cast(zero_bram_groups, "float32")) / num_bram_groups


from .utils import patterns
    
class PACAPatternMetric:
    def __init__(self, num_patterns_to_keep=16, beta=0.75, 
                 epsilon=1e-5, distance_metric='valued_hamming'):
        self.alpha = num_patterns_to_keep
        self.beta = beta
        self.distance_metric = distance_metric
        self.epsilon = epsilon
        self.dominant_patterns = None
        self.projection_mask = None
        self.src = "OIHW"                                   

    def __call__(self, weight):
        if len(weight.shape) != 4:
            return ops.convert_to_tensor(0.0, dtype=weight.dtype)

        if self.dominant_patterns is None:
            _, all_patterns, _ = patterns._get_kernels_and_patterns(weight, self.src, self.epsilon)
            unique_patterns, counts = patterns._get_unique_patterns_with_counts(all_patterns)
            self.dominant_patterns = patterns._select_dominant_patterns(all_patterns, unique_patterns, counts, 
                                                                    alpha = self.alpha, beta = self.beta, dtype=weight.dtype)

        if self.dominant_patterns is None or self.dominant_patterns.shape[0] == 0:
             return ops.convert_to_tensor(0.0, dtype=weight.dtype)

        w_kernels, distances = patterns._pattern_distances(weight, self.dominant_patterns, 
                                                           self.src, 
                                                           self.epsilon, 
                                                           self.distance_metric)
        min_distances = ops.min(distances, axis=1)
        return ops.mean(min_distances)

    def get_projection_mask(self, weight):
        if self.projection_mask is None:
            self.projection_mask = patterns._get_projection_mask(weight, self.dominant_patterns, self.src, self.epsilon, self.distance_metric)
        return self.projection_mask

    
    def get_dominant_patterns(self):
        return self.dominant_patterns
#-------------------------------------------------------------------
#                   MDMM Layer
#-------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="pquant.pruning_methods", name="MDMM")
class MDMM(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.layer_type = layer_type
        self.metric_fn = None
        self.constraint_layer = None
        self.penalty_loss = None
        self.built = False
        self.is_finetuning = False
    
    def build(self, input_shape):
        metric_type = self.config["pruning_parameters"].get("metric_type", "UnstructuredSparsity")
        constraint_type = self.config["pruning_parameters"].get("constraint_type", "GreaterThanOrEqual")
        target_value = self.config["pruning_parameters"].get("target_value", 0.0)
        
        if metric_type == "UnstructuredSparsity":
            self.metric_fn = UnstructuredSparsityMetric(
                epsilon=self.config["pruning_parameters"].get("epsilon", 1e-5),
                target_sparsity=self.config["pruning_parameters"].get("target_sparsity", 0.9),
                l0_mode=self.config["pruning_parameters"].get("l0_mode", "coarse"),
                scale_mode=self.config["pruning_parameters"].get("scale_mode", "mean")
            )
        elif metric_type == "FPGAAwareSparsity":
            self.metric_fn = FPGAAwareSparsityMetric(
                rf=self.config["pruning_parameters"].get("rf", 1),
                precision=self.config["pruning_parameters"].get("precision", 16),
                target_resource=self.config["pruning_parameters"].get("target_resource", "DSP")
            )
        elif metric_type == "PACAPatternSparsity":
            constraint_type = "Equality"
            target_value = 0.0
            self.metric_fn = PACAPatternMetric(
                num_patterns_to_keep=self.config["pruning_parameters"].get("num_patterns_to_keep", 16),
                beta=self.config["pruning_parameters"].get("beta", 0.75),
                epsilon=self.config["pruning_parameters"].get("epsilon", 1e-5),
                distance_metric=self.config["pruning_parameters"].get("distance_metric", "valued_hamming")
            )
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        common_args = {
            "metric_fn": self.metric_fn,
            "target_value": target_value,
            "scale": self.config["pruning_parameters"].get("scale", 1.0),
            "damping": self.config["pruning_parameters"].get("damping", 1.0),
            "use_grad": self.config["pruning_parameters"].get("use_grad", True),
            "lr": self.config.get("lr", 0.0),
        }
        
        if constraint_type == "Equality":
            self.constraint_layer = EqualityConstraint(**common_args)
        elif constraint_type == "LessThanOrEqual":
            self.constraint_layer = LessThanOrEqualConstraint(**common_args)
        elif constraint_type == "GreaterThanOrEqual":
            self.constraint_layer = GreaterThanOrEqualConstraint(**common_args)
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")
        
        self.mask = ops.ones(input_shape)
        self.constraint_layer.build(input_shape)

        
        super().build(input_shape)
        self.built = True
                    
    def call(self, weight):
        if not self.built:
            self.build(weight.shape)
        
        if self.is_finetuning:
            self.penalty_loss = 0.0
            if isinstance(self.metric_fn, PACAPatternMetric):
                self.mask = self.metric_fn.get_projection_mask(weight)
            weight = weight * self.mask
        else:
            self.penalty_loss = self.constraint_layer(weight)

        return weight 
    
    def get_hard_mask(self, weight):
        return self.mask
    
    def get_layer_sparsity(self, weight):
        epsilon = self.config["pruning_parameters"].get("epsilon", 1e-5)
        sparse_mask = ops.cast(ops.abs(weight) > epsilon, weight.dtype)
        return ops.sum(sparse_mask) / ops.size(weight) # Should this be subtracted from 1.0?

    def calculate_additional_loss(self):
        if self.penalty_loss is None:
            raise ValueError("Penalty loss has not been calculated. Call the layer with weights first.")
        else:
            penalty_loss = ops.sum(self.penalty_loss)
                    
        return penalty_loss
    
    
    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def pre_finetune_function(self):
        # Freeze the weights
        # Set lmbda(s) to zero
        self.is_finetuning = True
        
        if hasattr(self.constraint_layer, 'module'):
            self.constraint_layer.module.turn_off()
        else:
            self.constraint_layer.turn_off()

    def post_epoch_function(self, epoch, total_epochs):
        pass

    def post_pre_train_function(self):
        pass

    def post_round_function(self):
        pass
    
    
    
    
