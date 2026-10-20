import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# es necesario instalar las extensiones, por tanto hay que ejecutar "py -m pip install numpy scipy matplotlib" en el CMD

# ============================================================
# CONDICIONES INICIALES
# ============================================================

m1 = 2.0       # masa de m1 [kg]
m2 = 0.5       # masa de m2 [kg]

k = 20.0      # constante del muelle [N/m]

l = 0.0005       # longitud del péndulo [m]
l0 = 1.5       # longitud natural del muelle [m]

g = 9.81       # gravedad [m/s^2]


x0 = l0 + 0.3     # posición inicial de m1 [m]
v0 = 0.0                # velocidad inicial de m1 [m/s]

theta0 = np.deg2rad(20) # ángulo inicial [rad]
omega0 = 0.0            # velocidad angular inicial [rad/s]


# Vector inicial
y0 = [
    x0,
    v0,
    theta0,
    omega0
]


# ============================================================
# TIEMPO DE LA SIMULACIÓN
# ============================================================

t_inicio = 0.0
t_final = 15.0

num_puntos = 3000

t_eval = np.linspace(
    t_inicio,
    t_final,
    num_puntos
)


# ============================================================
# ECUACIONES DEL MOVIMIENTO
# ============================================================

def ecuaciones(t, y):

    x = y[0]
    x_dot = y[1]

    theta = y[2]
    theta_dot = y[3]


    # Sistema de ecuaciones para x_ddot y theta_ddot

    A = np.array([
        [m1 + m2, m2 * l * np.cos(theta)],
        [np.cos(theta), l]
    ])


    b = np.array([
        -k * (x - l0)
        + m2 * l * np.sin(theta) * theta_dot**2,

        -g * np.sin(theta)
    ])


    # Resolver el sistema

    x_ddot, theta_ddot = np.linalg.solve(A, b)


    return [
        x_dot,
        x_ddot,
        theta_dot,
        theta_ddot
    ]


# ============================================================
# INTEGRACIÓN NUMÉRICA
# ============================================================

sol = solve_ivp(
    ecuaciones,
    [t_inicio, t_final],
    y0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)


if not sol.success:
    raise RuntimeError("La integración numérica ha fallado.")


# Extraemos la solución

t = sol.t

x = sol.y[0]
theta = sol.y[2]


# ============================================================
# POSICIÓN DE m2
# ============================================================

# Posición de m2:
#
# x2 = x + l*sin(theta)
# y2 = -l*cos(theta)

x2 = x + l * np.sin(theta)
y2 = -l * np.cos(theta)


# ============================================================
# CREACIÓN DE LA ANIMACIÓN
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))


# Límites de la pantalla

margen = 0.8

xmin = min(
    -0.5,
    np.min(x) - margen,
    np.min(x2) - margen
)

xmax = max(
    l0 + 1,
    np.max(x) + margen,
    np.max(x2) + margen
)

ymin = -l - 0.5
ymax = 1.0


ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_aspect("equal")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

ax.set_title("Sistema masa-muelle-péndulo")

ax.grid()


# ============================================================
# ELEMENTOS DE LA ANIMACIÓN
# ============================================================

# Suelo

suelo, = ax.plot(
    [xmin, xmax],
    [0, 0],
    linewidth=3
)


# Pared

pared, = ax.plot(
    [0, 0],
    [0, 0.7],
    linewidth=3
)


# Muelle

muelle, = ax.plot(
    [],
    [],
    linewidth=2
)


# Masa m1

masa1, = ax.plot(
    [],
    [],
    "s",
    markersize=18
)


# Péndulo + masa m2

pendulo, = ax.plot(
    [],
    [],
    "o-",
    linewidth=2,
    markersize=10
)


# ============================================================
# TRAYECTORIA DE m2
# ============================================================

trayectoria, = ax.plot(
    [],
    [],
    "--",
    linewidth=1.5,
    alpha=0.7
)


# Texto del tiempo

texto_tiempo = ax.text(
    0.02,
    0.95,
    "",
    transform=ax.transAxes
)


# ============================================================
# ACTUALIZAR LA ANIMACIÓN
# ============================================================

def actualizar(frame):

    # --------------------------------------------------------
    # Posición actual de m1
    # --------------------------------------------------------

    x_actual = x[frame]


    # --------------------------------------------------------
    # Posición actual de m2
    # --------------------------------------------------------

    x2_actual = x2[frame]
    y2_actual = y2[frame]


    # --------------------------------------------------------
    # MUELLE
    # --------------------------------------------------------

    numero_espiras = 12

    xs = np.linspace(
        0,
        x_actual,
        numero_espiras * 2 + 1
    )

    amplitud = 0.08

    ys = np.zeros_like(xs)

    for i in range(1, len(xs) - 1):

        if i % 2 == 0:
            ys[i] = amplitud
        else:
            ys[i] = -amplitud


    muelle.set_data(xs, ys)


    # --------------------------------------------------------
    # MASA m1
    # --------------------------------------------------------

    masa1.set_data(
        [x_actual],
        [0]
    )


    # --------------------------------------------------------
    # PÉNDULO
    # --------------------------------------------------------

    pendulo.set_data(
        [x_actual, x2_actual],
        [0, y2_actual]
    )


    # --------------------------------------------------------
    # TRAYECTORIA DE m2
    # --------------------------------------------------------

    trayectoria.set_data(
        x2[:frame + 1],
        y2[:frame + 1]
    )


    # --------------------------------------------------------
    # TIEMPO
    # --------------------------------------------------------

    texto_tiempo.set_text(
        f"t = {t[frame]:.2f} s"
    )


    return (
        muelle,
        masa1,
        pendulo,
        trayectoria,
        texto_tiempo
    )


# ============================================================
# ANIMACIÓN
# ============================================================

salto = 5

frames = range(
    0,
    len(t),
    salto
)


animacion = FuncAnimation(
    fig,
    actualizar,
    frames=frames,
    interval=20,
    blit=True
)


plt.show()