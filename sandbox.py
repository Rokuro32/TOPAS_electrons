# Faster anti-spike version (reduced N, simpler deposition kernels)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

E0 = 10.0
rho = 0.9982
Rcsda_gcm2 = 4.975
Rcsda = Rcsda_gcm2 / rho

z_max = 8.0
dz = 0.04   # coarser for speed
Z = np.arange(0, z_max, dz)
NB = len(Z)

rng = np.random.default_rng(2026)

def S_col(E):
    a,b,c = 1.35,0.85,0.25
    return a + b/(E+c)

def S_rad(E):
    return E/36.0

def splat_linear(z_pos, dE, arr):
    if dE <= 0: return
    t = z_pos/dz
    i0 = int(np.floor(t))
    f1 = t - i0
    f0 = 1.0 - f1
    if 0 <= i0 < NB: arr[i0] += dE*f0
    i1 = i0+1
    if 0 <= i1 < NB: arr[i1] += dE*f1

def csda(E0, include_rad=False):
    dose = np.zeros(NB)
    z,E=0.0,E0
    while E>0 and z<z_max:
        S = S_col(E) + (S_rad(E) if include_rad else 0.0)
        dE = min(E, S*dz)
        splat_linear(z+0.5*dz, dE, dose)
        E -= dE; z += dz
    return dose

dose1 = csda(E0, False)
dose2 = csda(E0, True)

@dataclass
class Par:
    N:int=2500
    theta0_per_cm:float=0.08
    sigE_coll_frac:float=0.10
    sigE_rad_frac:float=0.22
    lambda_gamma:float=18.0
    Lkoe:float=0.05
par = Par()

def step_ms(theta):
    return theta + rng.normal(0.0, par.theta0_per_cm*np.sqrt(dz))

def photon_deposit(z_create, dE, arr):
    if dE <= 0: return
    z_dep = z_create + rng.exponential(par.lambda_gamma)
    z_dep = min(z_dep, z_max - 0.5*dz)
    splat_linear(z_dep, dE, arr)

def ko_deposit(z_create, dE, arr):
    if dE <= 0: return
    z_dep = z_create + rng.exponential(par.Lkoe)
    z_dep = min(z_dep, z_max - 0.5*dz)
    splat_linear(z_dep, dE, arr)

def track_csda_ms():
    dose = np.zeros(NB)
    for _ in range(par.N):
        z,E,th=0.0,E0,0.0
        while E>0 and z<z_max:
            th=step_ms(th); ct=max(0.3,np.cos(th)); dl=dz/ct
            S = S_col(E)+S_rad(E)
            dE_path=min(E,S*dl)
            splat_linear(z+0.5*dz, dE_path*(dz/dl), dose)
            E -= dE_path; z += dz
    return dose/par.N

def track_noKOe():
    dose=np.zeros(NB)
    for _ in range(par.N):
        z,E,th=0.0,E0,0.0
        while E>0 and z<z_max:
            th=step_ms(th); ct=max(0.3,np.cos(th)); dl=dz/ct
            Sc,Sr=S_col(E),S_rad(E)
            dE_coll=max(0.0, rng.normal(Sc*dl, par.sigE_coll_frac*Sc*dl))
            dE_rad =max(0.0, rng.normal(Sr*dl, par.sigE_rad_frac *Sr*dl))
            frac=dz/dl
            splat_linear(z+0.5*dz, dE_coll*frac, dose)   # KO local
            photon_deposit(z+0.5*dz, dE_rad*frac, dose)  # brems transported
            E -= min(E, dE_coll + dE_rad); z += dz
    return dose/par.N

def track_noBrems():
    dose=np.zeros(NB)
    for _ in range(par.N):
        z,E,th=0.0,E0,0.0
        while E>0 and z<z_max:
            th=step_ms(th); ct=max(0.3,np.cos(th)); dl=dz/ct
            Sc,Sr=S_col(E),S_rad(E)
            dE_coll,dE_rad = Sc*dl, Sr*dl
            frac=dz/dl
            ko_deposit(z+0.5*dz, 0.35*dE_coll*frac, dose)  # transported short range
            splat_linear(z+0.5*dz, (0.65*dE_coll + dE_rad)*frac, dose) # photons local
            E -= min(E, dE_coll + dE_rad); z += dz
    return dose/par.N

def track_brems_only():
    dose=np.zeros(NB)
    for _ in range(par.N):
        z,E=0.0,E0
        while E>0 and z<z_max:
            dE_rad = S_rad(E)*dz
            photon_deposit(z+0.5*dz, dE_rad, dose)
            E -= min(E, (S_col(E)+S_rad(E))*dz); z += dz
    return dose/par.N

def track_full():
    dose=np.zeros(NB)
    for _ in range(par.N):
        z,E,th=0.0,E0,0.0
        while E>0 and z<z_max:
            th=step_ms(th); ct=max(0.3,np.cos(th)); dl=dz/ct
            Sc,Sr=S_col(E),S_rad(E)
            dE_coll,dE_rad=Sc*dl,Sr*dl
            frac=dz/dl
            ko_deposit(z+0.5*dz, 0.35*dE_coll*frac, dose)
            splat_linear(z+0.5*dz, 0.65*dE_coll*frac, dose)
            photon_deposit(z+0.5*dz, dE_rad*frac, dose)
            E -= min(E, dE_coll + dE_rad); z += dz
    return dose/par.N

def x_peak_of_full(theta0_value):
    par.theta0_per_cm = theta0_value
    d7 = track_full()                 # recalcule "full" uniquement
    i = int(np.argmax(d7))
    return (Z[i] / Rcsda), d7

target = 0.82
lo, hi = 0.09, 0.16
for _ in range(10):
    mid = 0.5*(lo+hi)
    xpk, _ = x_peak_of_full(mid)
    if xpk > target:  # pic trop à droite -> plus de MS
        lo = mid
    else:
        hi = mid
par.theta0_per_cm = hi  # valeur finale
# recalcul des courbes 3..7 avec la MS ajustée
dose3 = track_csda_ms()
dose4 = track_noKOe()
dose5 = track_noBrems()
dose6 = track_brems_only()
dose7 = track_full()


x = Z / Rcsda
norm = lambda d: d * Rcsda / E0

# -------- helper: portée effective d'une courbe par son pied après le pic --------
def effective_range_from_tail(z, y, eps_frac=1e-3):
    y = np.asarray(y); z = np.asarray(z)
    if y.max() <= 0: return z[-1]
    i_peak = int(np.argmax(y))
    thr = eps_frac * y[i_peak]
    tail = np.where(y[i_peak:] > thr)[0]
    if len(tail) == 0: return z[i_peak]
    i_last = i_peak + tail[-1]
    return z[i_last]

R1 = effective_range_from_tail(Z, dose1)   # portée propre CSDA_el
R2 = effective_range_from_tail(Z, dose2)   # portée propre CSDA (total)
x1 = Z / R1
x2 = Z / R2
x  = Z / Rcsda     # pour 3–7

scale = lambda d: d * Rcsda / E0

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(x1 * 1.2, scale(dose1), label=f"1. CSDA_el (R₁={R1:.2f} cm)")
plt.plot(x2 *1.2, scale(dose2), label=f"2. CSDA (R₂={R2:.2f} cm)")
plt.plot(x,  scale(dose3), label="3. CSDA+ms")
plt.plot(x,  scale(dose4), label="4. no KOes")
plt.plot(x,  scale(dose5), label="5. no brems", alpha=0.8)
plt.plot(x,  scale(dose6), label="6. brems only", alpha=0.8)
plt.plot(x,  scale(dose7), label="7. full", lw=2)

plt.xlim(0, 1.4)
plt.ylim(0, 0.07)
plt.xlabel("Profondeur dans l'eau, z / R (a.u.)")
plt.ylabel("Dose à l'échelle, D · R_CSDA / E0 (a.u.)")
plt.title("Anatomy 10 MeV")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
