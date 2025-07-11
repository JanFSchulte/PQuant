import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def soft_quantize(x, k):

    levels = np.array([-1.0, -0.5, 0.0, 0.5])
    
    x_reshaped = x[:, np.newaxis]      
    levels_reshaped = levels[np.newaxis, :]
    dist = np.square(x_reshaped - levels_reshaped)
    exp_term = np.exp(-k * dist)
    weights = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    return np.sum(weights * levels, axis=1)

def soft_quantize_derivative(x, k):
    levels = np.array([-1.0, -0.5, 0.0, 0.5])
    
    x_reshaped = x[:, np.newaxis]
    levels_reshaped = levels[np.newaxis, :]
    
    dist = np.square(x_reshaped - levels_reshaped)
    exp_term = np.exp(-k * dist)
    weights = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    quantized_x = np.sum(weights * levels, axis=1)
    quantized_x_sq = np.sum(weights * (levels**2), axis=1)
    
    return 2 * k * (quantized_x_sq - quantized_x**2)


fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.25) 

x_input = np.linspace(-1.5, 1.5, 500)
initial_k = 1.0

identity_line, = ax.plot(x_input, x_input, 'k--', alpha=0.5, label='Identity (y=x)')
soft_quant_line, = ax.plot(x_input, soft_quantize(x_input, initial_k), 'b-', linewidth=2.5, label='Soft Quantization (Trainable)')
derivative_line, = ax.plot(x_input, soft_quantize_derivative(x_input, initial_k), 'r-', linewidth=2, label='Derivative')



ax.grid(True, linestyle=':')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
k_slider = Slider(
    ax=ax_slider,
    label='Annealing Temp (k)',
    valmin=0,    # 10^0 = 1
    valmax=3,    # 10^3 = 1000
    valinit=0,   # Start at 10^0 = 1
    color='#007ACC'
)

def update(val):
    k = 10**k_slider.val
    new_y = soft_quantize(x_input, k)
    soft_quant_line.set_ydata(new_y)
    
    new_derivative_y = soft_quantize_derivative(x_input, k)
    derivative_line.set_ydata(new_derivative_y)
    
    fig.canvas.draw_idle()

k_slider.on_changed(update)

plt.show()


class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  def call(self, inputs):
      if self.training:
          return soft_quantize(inputs)
      else:
          return hard_quantize(inputs)

layer = MyDenseLayer(10)