import jax.numpy as jnp
from jax import grad, jit

# 1. Datos de ejemplo
x = jnp.array([1.0, 2.0, 3.0, 4.0])
y = jnp.array([2.0, 4.0, 6.0, 8.0])  # y = 2x

# 2. Modelo (una recta)
def modelo(params, x):
    w, b = params
    return w * x + b

# 3. Función de error (qué tan mal predice el modelo)
def perdida(params, x, y):
    y_pred = modelo(params, x)
    return jnp.mean((y - y_pred) ** 2)

# 4. Gradiente de la pérdida
grad_perdida = grad(perdida)

# 5. Compilamos con JIT
perdida_jit = jit(perdida)
grad_perdida_jit = jit(grad_perdida)

# 6. Parámetros iniciales
params = jnp.array([0.0, 0.0])  # w = 0, b = 0
learning_rate = 0.1

# 7. Entrenamiento
for i in range(100):
    grads = grad_perdida_jit(params, x, y)
    params = params - learning_rate * grads

# 8. Resultado final
print("Parámetros aprendidos (w, b):", params)
print("Predicción para x=5:", modelo(params, 5.0))
