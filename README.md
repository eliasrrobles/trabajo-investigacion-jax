# Trabajo de Investigación: JAX

***Elías Robles Ruiz** - **CPIFP Alan Turing** - Máster de IA y BigData*

## ¿Qué es JAX?

JAX, que significa *Just Another XLA*, es una librería de Python pensada para hacer cálculos numéricos de forma rápida, sobre todo en tareas de **aprendizaje automático**.
Fue creada por Google y está diseñada para aprovechar mejor el hardware potente como las **GPU** y las **TPU**, lo que hace que los programas se ejecuten mucho más rápido que usando solo la CPU.

Una de las ventajas principales de JAX es que se parece mucho a NumPy, una librería muy usada en Python, por lo que no es tan difícil de aprender si ya se tiene algo de experiencia previa.

---

## Principales características de JAX

### API NumPy (`jax.numpy`)

JAX tiene una API muy parecida a NumPy, que permite escribir código casi igual al de NumPy.

### Compilación Just-In-Time (JIT)

El compilador de JAX se llama **XLA (Accelerated Linear Algebra)** que optimiza el código.
Usando `@jit` en el código este se compila y se ejecuta mucho más rápido, sobretodo en GPUs y TPUs.

### Diferenciación automática (`grad`)

Sirve para calcular derivadas de manera automática. Esto es muy útil para entrenar modelos de redes neuronales, ya que facilita el cálculo de gradientes sin hacerlo a mano.

### Vectorización automática (`vmap`)

La función `vmap` permite que una función que trabaja con un solo dato pueda aplicarse automáticamente a muchos datos a la vez (lotes o batches), haciendo el código más simple y eficiente.

### Programación funcional

Se pueden hacer uso de funciones puras y datos inmutables. Esto ayuda a que el código sea más fácil de optimizar y paralelizar.

### Paralelización (`pmap`)

Con `pmap`, JAX puede repartir los cálculos entre varios dispositivos (por ejemplo, varias GPUs o TPUs), lo que permite entrenar modelos más grandes y rápidos.

---

## Comparación de JAX con TensorFlow y PyTorch

A diferencia de **TensorFlow** o **PyTorch**, JAX no es un framework completo para crear modelos desde cero con muchas herramientas integradas.
En cambio, JAX se centra en ofrecer **alto rendimiento y flexibilidad**, por lo que es muy usado en investigación y experimentos avanzados.

**JAX**: Usando principalmente para investigación, experimentación matemática y alto rendimiento.

**PyTorch**: Más equilibrado entre investigación y producción.

**TensorFlow**: El más fuerte en producción y despliegue industrial.

---

## Ecosistema de JAX

Aunque JAX no es un framework de redes neuronales como tal, existen varias librerías que se construyen sobre él:

- **Flax**: Librería para crear redes neuronales de forma clara y organizada.
- **Haiku**: Otra opción para construir modelos, muy usada en investigación.
- **Optax**: Librería para optimizadores (como Adam o SGD).
- **Chex**: Herramientas para depuración y validación de código en JAX.

Estas librerías permiten usar JAX de una manera más parecida a otros frameworks conocidos.

---

## Conclusión

Trás investigar sobre JAX se puede afirmar que es una herramienta muy potente para el aprendizaje automático, especialmente útil en investigación.

Su gran ventaja es la velocidad y la posibilidad de usar hardware avanzado, aunque puede ser un poco más complejo al inicio comparado con otras opciones.

Además cuenta con el respaldo de algunas empresas muy importantes como Google, Nvidia, Anthropic, xIA o Apple en modelos de fundacionales.

---

## Bibliografía

La información utilizada para este trabajo fue obtenida a partir de las siguientes fuentes:

- [EITCA – Introducción a JAX](https://es.eitca.org/artificial-intelligence/eitc-ai-gcml-google-cloud-machine-learning/google-cloud-ai-platform/introduction-to-jax/)

- [Documentación oficial de Google Cloud sobre JAX y TPU](https://docs.cloud.google.com/tpu/docs/jax-ai-stack?hl=es)

- [Documentación oficial de JAX](https://jax.readthedocs.io/)

- [Gemini (Google AI)](https://gemini.google.com/)

- [ChatGPT](https://chatgpt.com/)

- [Artículo de WALL STREET MARKETING](https://thewallstreetmarketing.com/2025/11/building-ai-with-jax-tpus/)
