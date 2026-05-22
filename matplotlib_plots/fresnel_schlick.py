"""
Compare Fresnel's equation vs. Schlick's approximation.

References:
https://graphicscompendium.com/raytracing/11-fresnel-beer
https://psgraphics.blogspot.com/2020/03/fresnel-equations-schlick-approximation.html
"""

from numpy import radians, abs, sin, cos, sqrt, array
import matplotlib.pyplot as plt

# Indices of refraction.
n_1 = 1
n_2 = 1.5

# Reflectance at normal incidence.
f_0 = ((n_1 - n_2) / (n_1 + n_2)) ** 2


def schlick_approximation(f_0, cos_theta):
    return f_0 + (1 - f_0) * (1 - cos_theta) ** 5


def fresnel(r_i):
    # S-polarized light.
    r_s = abs((n_1 * cos(r_i) - n_2 * sqrt(1 - (n_1 / n_2 * sin(r_i)) ** 2)) /
              (n_1 * cos(r_i) + n_2 * sqrt(1 - (n_1 / n_2 * sin(r_i)) ** 2))) ** 2
    
    # P-polarized light.
    r_p = abs((n_1 * sqrt(1 - (n_1 / n_2 * sin(r_i)) ** 2) - n_2 * cos(r_i)) /
              (n_1 * sqrt(1 - (n_1 / n_2 * sin(r_i)) ** 2) + n_2 * cos(r_i))) ** 2

    return (r_s + r_p) / 2
    

r_i = [i for i in range(90)]
schlick_r = array([schlick_approximation(f_0, cos(radians(n))) for n in r_i])
fresnel_r = array([fresnel(radians(n)) for n in r_i])

plt.plot(r_i, schlick_r, "r-", label="Schlick Approximation")
plt.plot(r_i, fresnel_r, "b-", label="Fresnel Equation")

# Label x and y axes with LaTeX
plt.xlabel(r"Angle of Incidence ($\theta_i$°)")
plt.ylabel(r"Reflectance ($R$)")

plt.legend()
plt.show()