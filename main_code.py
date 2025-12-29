import os

os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"

import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import datetime

from amuse.units import units, constants
from amuse.units import nbody_system
from amuse.lab import Particles, Particle, new_plummer_model
from amuse.community.ph4 import Ph4
from amuse.community.seba import Seba
from amuse.couple import bridge
from amuse.ext.galactic_potentials import MWpotentialBovy2015

MEASURED_VELOCITY = 55.0 | units.kms
DISTANCE_TO_GALACTIC_CENTER = 100 | units.parsec
CLUSTER_RADIUS = 0.2| units.parsec
ECCENTRICITY = 0.6
INCLINATION = 5.0

SIMULATION_END = 0.03 | units.Myr
DIAGNOSTIC_DT = 0.001| units.Myr
BRIDGE_TIMESTEP = 0.001 | units.Myr
PH4_WORKERS = 3

COLLISION_RADIUS_FACTOR = 2.0
ACCRETION_RADIUS_FACTOR = 15.0
MAX_FRAMES = 250

C_MASS_LOSS_MS = 0.30

Q_BREAK = 0.4
C_LOW_Q = 0.243
C_HIGH_Q = 0.30

BH_BASE_PROB = 0.03
BH_Q_SLOPE = 0.05

MIN_NS_MASS = 8.0
MAX_NS_MASS = 2.2
NS_BH_TRANSITION = 5.0
MIN_BH_MASS_SINGLE = 20.0
MIN_BH_MASS_MERGER = 40.0
MAX_WD_MASS = 1.4

HYDROGEN_RETAIN_O = 0.7
HYDROGEN_RETAIN_WN = 0.3
HYDROGEN_RETAIN_DEFAULT = 0.5

GAS_FRICTION_ENABLED = True
GAS_DENSITY_PROFILE = 'central_molecular_zone'
DRAG_COEFFICIENT = 1
COULOMB_LOGARITHM = 8.0
MIN_FRICTION_VELOCITY = 0.1 | units.kms

CHANDRASEKHAR_LIMIT = 1.44
OPPENHEIMER_VOLKOFF_LIMIT = 2.5


NS_UPPER_LIMIT = 25.0
DIRECT_COLLAPSE_THRESHOLD = 60.0
PISN_LOWER_THRESHOLD = 130.0
PISN_UPPER_THRESHOLD = 250.0

MAX_AGE_FACTOR = 0.95
COLLAPSE_CHECK_INTERVAL = 10

def get_bh_thresholds(metallicity):
    if metallicity < 0.001:
        return 15.0, 30.0
    elif metallicity < 0.01:
        return 18.0, 36.0
    elif metallicity < 0.02:
        return 22.0, 44.0
    else:
        return 25.0, 50.0

def can_collapse_to_bh_now(star_age, star_mass, current_time, stellar_type=""):
    if "Merged" in stellar_type or "Collision" in stellar_type:
        return True

    if star_mass.value_in(units.MSun) > 0:
        lifetime_myr = 10.0 * (100.0 / star_mass.value_in(units.MSun)) ** 0.75
    else:
        return True

    if star_age.value_in(units.Myr) < lifetime_myr * 0.7:
        return False
    elif star_age.value_in(units.Myr) > lifetime_myr * 0.9:
        return True
    else:
        age_fraction = star_age.value_in(units.Myr) / lifetime_myr
        collapse_prob = (age_fraction - 0.7) * 3.33
        return np.random.random() < collapse_prob

def bh_probability_from_q_mass(q, total_mass):
    base_prob = 0.02
    q_factor = 0.05 * q
    mass_factor = 0.0
    if total_mass > 100.0:
        mass_factor = 0.10
    elif total_mass > 80.0:
        mass_factor = 0.06
    elif total_mass > 60.0:
        mass_factor = 0.03
    elif total_mass > 40.0:
        mass_factor = 0.01

    total_prob = base_prob + q_factor + mass_factor
    return max(0.0, min(0.15, total_prob))

def determine_remnant_type_realistic(mass_val, metallicity, stellar_type="",
                                     bh_prob=0.02, current_time=None, star_age=None,
                                     is_merger=False):
    min_bh_single, min_bh_merger = get_bh_thresholds(metallicity)
    bh_mass_threshold = min_bh_merger if is_merger else min_bh_single

    if mass_val < MAX_WD_MASS:
        if is_merger and mass_val > 1.0:
            if np.random.random() < 0.05:
                return "White Dwarf", "WD"
            else:
                return "Merged Star", "Merged"
        return "White Dwarf", "WD"

    elif mass_val <= MAX_NS_MASS:
        if is_merger:
            if np.random.random() < 0.2:
                return "Neutron Star", "NS"
            else:
                return "Merged Star", "Merged"
        return "Neutron Star", "NS"

    elif mass_val <= NS_BH_TRANSITION:
        bh_prob_transition = (mass_val - MAX_NS_MASS) / (NS_BH_TRANSITION - MAX_NS_MASS)
        total_bh_prob = max(bh_prob_transition, bh_prob)

        if np.random.random() < total_bh_prob:
            if can_collapse_to_bh_now(star_age, mass_val | units.MSun, current_time, stellar_type):
                return "Black Hole", "BH"

        if is_merger:
            return "Merged Star", "Merged"
        return "Neutron Star", "NS"

    elif mass_val < bh_mass_threshold:
        enhanced_bh_prob = bh_prob

        if "WN" in stellar_type or "O" in stellar_type:
            enhanced_bh_prob = min(0.25, bh_prob * 2.5)

        if np.random.random() < enhanced_bh_prob:
            if can_collapse_to_bh_now(star_age, mass_val | units.MSun, current_time, stellar_type):
                return "Black Hole", "BH"

        return "Merged Star", "Merged"

    else:
        final_bh_prob = bh_prob

        if is_merger or "WN" in stellar_type or "O" in stellar_type:
            mass_factor = min(0.4, (mass_val - bh_mass_threshold) / 150.0)
            final_bh_prob = min(0.6, bh_prob + mass_factor)

        if np.random.random() < final_bh_prob:
            if is_merger or can_collapse_to_bh_now(star_age, mass_val | units.MSun, current_time, stellar_type):
                return "Black Hole", "BH"

        if mass_val > 80:
            return "Very Massive Star", "VMS"
        return "Merged Star", "Merged"

def retain_frac_from_q(q):
    return max(0.0, min(1.0, 1.0 - C_MASS_LOSS_MS * q / (1.0 + q) ** 2))

def retain_frac_reinoso(q):
    if q < Q_BREAK:
        f_lost = C_LOW_Q * q / (1.0 + q ** 2)
    else:
        f_lost = C_HIGH_Q * q / (1.0 + q ** 2)
    return max(0.0, min(1.0, 1.0 - f_lost))

COLLISION_RECIPES = {
    "O": {
        "type": "O",
        "retain_model": "glebbeek",
        "bh_prob_model": "kremer_mass",
        "min_mass_bh_single": 20.0,
        "min_mass_bh_merger": 40.0,
        "hydrogen_retain": HYDROGEN_RETAIN_O,
        "wd_probability": 0.01,
        "ns_probability": 0.15,
        "radius_exponent": 0.8,
        "description": "O-type stars (main sequence)"
    },

    "WN": {
        "type": "WN",
        "retain_model": "reinoso",
        "bh_prob_model": "kremer_mass",
        "min_mass_bh_single": 25.0,
        "min_mass_bh_merger": 50.0,
        "hydrogen_retain": HYDROGEN_RETAIN_WN,
        "wd_probability": 0.001,
        "ns_probability": 0.10,
        "radius_exponent": 0.6,
        "description": "Wolf-Rayet stars (hydrogen-poor)"
    },

    "Unclassified": {
        "type": "Unclassified",
        "retain_model": "glebbeek",
        "bh_prob_model": "kremer_mass",
        "min_mass_bh_single": MIN_BH_MASS_SINGLE,
        "min_mass_bh_merger": MIN_BH_MASS_MERGER,
        "hydrogen_retain": HYDROGEN_RETAIN_DEFAULT,
        "wd_probability": 0.05,
        "ns_probability": 0.20,
        "radius_exponent": 0.75,
        "description": "Unclassified/unknown spectral types"
    },

    "BH": {
        "type": "BH",
        "retain_fixed": 1.0,
        "bh_fixed": 1.0,
        "min_mass_bh": 0.0,
        "radius_type": "schwarzschild",
        "description": "Black hole"
    },

    "NS": {
        "type": "NS",
        "retain_fixed": 0.9,
        "bh_fixed": 0.05,
        "min_mass_bh": 2.8,
        "radius_fixed_km": 12.0,
        "description": "Neutron star"
    },

    "WD": {
        "type": "WD",
        "retain_fixed": 0.6,
        "bh_fixed": 0.0,
        "min_mass_bh": 100.0,
        "radius_fixed_rsun": 0.01,
        "description": "White dwarf"
    },

    "Merged": {
        "type": "Merged",
        "retain_model": "glebbeek",
        "bh_prob_model": "kremer_mass",
        "min_mass_bh_single": MIN_BH_MASS_SINGLE,
        "min_mass_bh_merger": MIN_BH_MASS_MERGER,
        "hydrogen_retain": HYDROGEN_RETAIN_DEFAULT,
        "wd_probability": 0.02,
        "ns_probability": 0.10,
        "radius_exponent": 0.7,
        "description": "Previous merger product"
    },

    "VMS": {
        "type": "VMS",
        "retain_model": "glebbeek",
        "bh_prob_model": "kremer_mass",
        "min_mass_bh_single": 30.0,
        "min_mass_bh_merger": 60.0,
        "hydrogen_retain": 0.4,
        "wd_probability": 0.0,
        "ns_probability": 0.05,
        "radius_exponent": 0.6,
        "description": "Very massive star (>80 Msun)"
    }
}

COLLISION_RECIPES_DEFAULT = {
    "type": "Default",
    "retain_model": "glebbeek",
    "bh_prob_model": "kremer_mass",
    "min_mass_bh_single": MIN_BH_MASS_SINGLE,
    "min_mass_bh_merger": MIN_BH_MASS_MERGER,
    "hydrogen_retain": HYDROGEN_RETAIN_DEFAULT,
    "wd_probability": 0.10,
    "ns_probability": 0.25,
    "radius_exponent": 0.75,
    "description": "Default recipe for unknown types"
}

SPECTRAL_TABLE = [
    ("O4-5 I-III", 4, 38.867, 39.73, 3.45e-07),
    ("O4-5 la", 19, 27.458, 28.07, 2.45e-07),
    ("O4-5 la+", 3, 34.846, 35.53, 2.74e-07),
    ("O5-6 III-V", 3, 34.36, 35.86, 6.00e-07),
    ("O5-6 V", 4, 28.985, 29.46, 1.90e-07),
    ("O5-6 la+", 1, 50.492, 52.73, 8.95e-07),
    ("O5.5-6 I-III", 3, 12.712, 12.73, 7.20e-09),
    ("O5.5-6 la", 4, 58.984, 72.23, 5.30e-06),
    ("O6-6.5 III", 2, 60.014, 65.73, 2.29e-06),
    ("O6-6.5 la", 2, 10.826, 10.83, 1.60e-09),
    ("O6-7 III-V", 2, 19.804, 19.93, 5.04e-08),
    ("O6-7 la+", 1, 10.016, 13.03, 1.21e-06),
    ("O6-8 V", 7, 43.828, 48.48, 1.86e-06),
    ("O7-8 III-V", 1, 53.79, 56.73, 1.18e-06),
    ("O7-8 la+", 2, 60.014, 63.73, 1.49e-06),
    ("WN7-8h", 2, 34.172, 37.03, 1.14e-06),
    ("WN8-9h", 11, 13.575, 13.63, 2.20e-08),
    (">=O8 V", 13, 12.57, 12.63, 2.40e-08),
    ("Unclassified", 772, 28.593, 30.39, 7.19e-07),
]

class GasDynamicalFriction:
    def __init__(self, gas_density_profile='central_molecular_zone',
                 drag_coefficient=0.5, coulomb_logarithm=3.0):

        self.drag_coefficient = drag_coefficient
        self.coulomb_log = coulomb_logarithm

        self.gas_profiles = {
            'central_molecular_zone': self.cmz_density_profile,
            'constant_density': self.constant_density_profile,
            'power_law': self.power_law_density_profile
        }

        self.density_profile = self.gas_profiles.get(
            gas_density_profile,
            self.cmz_density_profile
        )

        self.cmz_params = {
            'central_density': 1e4 | units.MSun / units.parsec ** 3,
            'scale_radius': 50 | units.parsec,
            'core_radius': 10 | units.parsec,
            'sound_speed': 15 | units.kms
        }

        self.total_friction_accel = 0.0 | units.kms / units.Myr
        self.max_friction_accel = 0.0 | units.kms / units.Myr
        self.friction_events = 0
        self.debug_info = []
        self.avg_friction_accel = 0.0 | units.kms / units.Myr
        self.current_accel_sum = 0.0 | units.kms / units.Myr

    def cmz_density_profile(self, position):
        r = np.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)

        if r < self.cmz_params['core_radius']:
            density = self.cmz_params['central_density']
        else:
            scale_height = 20 | units.parsec
            radial_scale = self.cmz_params['scale_radius']

            radial_term = np.exp(-r / radial_scale)
            vertical_term = np.exp(-np.abs(position[2]) / scale_height)

            density = self.cmz_params['central_density'] * radial_term * vertical_term

        min_density = 1e-3 * self.cmz_params['central_density']
        return max(density, min_density)

    def constant_density_profile(self, position):
        return 5e3 | units.MSun / units.parsec ** 3

    def power_law_density_profile(self, position):
        r = np.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)
        if r < 1 | units.parsec:
            return self.cmz_params['central_density']
        else:
            return self.cmz_params['central_density'] * (r.value_in(units.parsec)) ** (-2.0)

    def calculate_accelerations(self, particles):
        accelerations = []
        current_debug = []

        self.current_accel_sum = 0.0 | units.kms / units.Myr
        accel_count = 0

        G = constants.G.value_in(units.parsec ** 3 / units.MSun / units.Myr ** 2)

        for i, particle in enumerate(particles):
            position = [particle.x, particle.y, particle.z]

            gas_density = self.density_profile(position)
            rho = gas_density.value_in(units.MSun / units.parsec ** 3)

            v_vec = np.array([
                particle.vx.value_in(units.parsec / units.Myr),
                particle.vy.value_in(units.parsec / units.Myr),
                particle.vz.value_in(units.parsec / units.Myr)
            ])

            v_mag = np.sqrt(v_vec[0] ** 2 + v_vec[1] ** 2 + v_vec[2] ** 2)

            v_mag_kms = v_mag * (units.parsec / units.Myr).value_in(units.kms)

            M = particle.mass.value_in(units.MSun)

            cs = self.cmz_params['sound_speed'].value_in(units.parsec / units.Myr)
            cs_kms = self.cmz_params['sound_speed'].value_in(units.kms)

            v_min_kms = 1.0

            if v_mag_kms < v_min_kms or M < 1.0:
                accelerations.append([0.0, 0.0, 0.0] | units.parsec / units.Myr ** 2)
                continue

            X = v_mag / (np.sqrt(2) * cs) if cs > 0 else 1.0

            if X > 3.0:
                f_X = 0.5 * np.log((X ** 2 - 1) / X ** 2) + 1.0
            elif X > 1.0:
                f_X = 0.5 * np.log(X ** 2 - 1) + 1.0
            else:
                f_X = (2 / 3) * X ** 3

            f_X = max(f_X, 1e-10)

            if v_mag > 0:
                accel_mag = -4 * np.pi * G ** 2 * M * rho * self.coulomb_log * f_X / v_mag ** 3
                accel_mag *= 1
            else:
                accel_mag = 0.0

            if M > 5.0:
                R_bondi = 2 * G * M / (cs ** 2 + v_mag ** 2)
                R = min(R_bondi, 0.1)

                A = np.pi * R ** 2
                drag_accel_mag = -0.5 * self.drag_coefficient * rho * A * v_mag
                accel_mag += drag_accel_mag

            if v_mag > 0 and abs(accel_mag) > 1e-20:
                accel_vec = [
                    accel_mag * v_vec[0],
                    accel_mag * v_vec[1],
                    accel_mag * v_vec[2]
                ]

                accel_kms_myr = abs(accel_mag) * (units.parsec / units.Myr ** 2).value_in(units.kms / units.Myr)

                self.current_accel_sum += accel_kms_myr | units.kms / units.Myr
                accel_count += 1

                acceleration = [
                    accel_vec[0] | units.parsec / units.Myr ** 2,
                    accel_vec[1] | units.parsec / units.Myr ** 2,
                    accel_vec[2] | units.parsec / units.Myr ** 2
                ]
            else:
                acceleration = [0.0, 0.0, 0.0] | units.parsec / units.Myr ** 2
                accel_kms_myr = 0.0

            accelerations.append(acceleration)

            if accel_kms_myr > 0:
                self.total_friction_accel += accel_kms_myr | units.kms / units.Myr
                self.max_friction_accel = max(self.max_friction_accel, accel_kms_myr | units.kms / units.Myr)
                self.friction_events += 1

        if accel_count > 0:
            self.avg_friction_accel = self.current_accel_sum / accel_count
        else:
            self.avg_friction_accel = 0.0 | units.kms / units.Myr

        return accelerations

class GasFrictionForce:
    def __init__(self, gas_friction_module):
        self.gas_friction = gas_friction_module
        self.tracked_particles = None
        self.model_time = 0.0 | units.Myr
        self.call_count = 0

    def set_tracked_particles(self, particles):
        self.tracked_particles = particles

    def get_gravity_at_point(self, eps, x, y, z):
        return (0.0, 0.0, 0.0) | units.parsec / units.Myr ** 2

    def get_potential_at_point(self, eps, x, y, z):
        return 0.0 | units.parsec ** 2 / units.Myr ** 2

    def evolve_model(self, t_end):
        self.model_time = t_end

    @property
    def particles(self):
        return self.tracked_particles

    @property
    def mass(self):
        if self.tracked_particles:
            return self.tracked_particles.total_mass()
        return 0.0 | units.MSun

    def kick_particles(self, dt, particles):
        if particles is None or len(particles) == 0:
            return

        self.call_count += 1

        accelerations = self.gas_friction.calculate_accelerations(particles)

        applied_accels = 0
        max_accel = 0.0

        for i, particle in enumerate(particles):
            if i < len(accelerations):
                ax, ay, az = accelerations[i]

                accel_mag = np.sqrt(ax.value_in(units.parsec / units.Myr ** 2) ** 2 +
                                    ay.value_in(units.parsec / units.Myr ** 2) ** 2 +
                                    az.value_in(units.parsec / units.Myr ** 2) ** 2)

                if accel_mag > 1e-10:
                    particle.vx += ax * dt
                    particle.vy += ay * dt
                    particle.vz += az * dt
                    applied_accels += 1
                    max_accel = max(max_accel, accel_mag)

        if self.call_count % 10 == 0:
            print(f"[GAS FRICTION] Call #{self.call_count}: "
                  f"Applied to {applied_accels}/{len(particles)} particles, "
                  f"Max accel={max_accel * (units.parsec / units.Myr ** 2).value_in(units.kms / units.Myr):.3e} km/s/Myr")

    def print_statistics(self):
        print(f"\n=== GAS FRICTION STATISTICS ===")
        print(f"Total kick calls: {self.call_count}")
        print(f"Total friction events: {self.gas_friction.friction_events}")

        if self.gas_friction.friction_events > 0:
            avg_accel = self.gas_friction.total_friction_accel / self.gas_friction.friction_events
            print(f"Average acceleration: {avg_accel.value_in(units.kms / units.Myr):.3e} km/s/Myr")
            print(
                f"Maximum acceleration: {self.gas_friction.max_friction_accel.value_in(units.kms / units.Myr):.3e} km/s/Myr")
        else:
            print(f"No friction events recorded")

        if self.gas_friction.debug_info:
            print(f"\nDebug info (first few particles):")
            for i, info in enumerate(self.gas_friction.debug_info[:5]):
                print(f"  Particle {i}: ρ={info.get('rho', 0):.1f} Msun/pc³, "
                      f"v={info.get('v_mag_kms', 0):.1f} km/s, "
                      f"a={info.get('accel_kms_myr', 0):.3e} km/s/Myr")

    def stop(self):
        """Метод для корректной остановки кода"""
        if not self.is_stopped:
            print(f"[GasFrictionForce] Остановка модуля газового трения")
            self.is_stopped = True
        return 0

def create_result_directory(base_dir="simulation_results"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(result_dir)

    return result_dir

def save_statistical_plots(result_dir, diagnostic_data, animation_positions, animation_times, animation_com_positions):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Эволюция звёздного скопления', fontsize=16)

    ax = axes[0, 0]

    avg_masses = []
    for frame_idx, frame_positions in enumerate(animation_positions):
        if frame_idx < len(animation_times):
            masses = [star['mass'] for star in frame_positions]
            if masses:
                avg_masses.append(np.mean(masses))
            else:
                avg_masses.append(0.0)

    if len(avg_masses) > 0:
        ax.plot(animation_times[:len(avg_masses)], avg_masses, 'b-', linewidth=2, label='Средняя масса')

        max_masses = []
        for frame_positions in animation_positions:
            masses = [star['mass'] for star in frame_positions]
            if masses:
                max_masses.append(np.max(masses))
            else:
                max_masses.append(0.0)

        ax.plot(animation_times[:len(max_masses)], max_masses, 'r--', linewidth=2, label='Максимальная масса')

        ax.set_xlabel('Время (Myr)')
        ax.set_ylabel('Масса (M☉)')
        ax.set_title('Эволюция средней и максимальной массы объектов')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if len(avg_masses) > 1:
            ax.annotate(f'Начало: {avg_masses[0]:.1f} M☉',
                        xy=(animation_times[0], avg_masses[0]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

    else:
        ax.text(0.5, 0.5, 'Нет данных о массах',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title('Эволюция массы объектов')

    ax = axes[0, 1]
    times = diagnostic_data['times']
    ax.plot(times, diagnostic_data['n_bh'], 'k-', linewidth=2, label='Чёрные дыры')
    ax.plot(times, diagnostic_data['n_ns'], 'r-', linewidth=2, label='Нейтронные звёзды')
    ax.plot(times, diagnostic_data['n_wd'], 'g-', linewidth=2, label='Белые карлики')
    ax.plot(times, diagnostic_data['n_merged'], 'm-', linewidth=2, label='Обычные звезды (слияния)')
    ax.plot(times, diagnostic_data['n_vms'], 'c-', linewidth=2, label='VMS')

    ax.set_xlabel('Время (Myr)')
    ax.set_ylabel('Число объектов')
    ax.set_title('Эволюция типов объектов')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=100)

    ax = axes[1, 0]
    ax.plot(diagnostic_data['times'], diagnostic_data['velocity_dispersion'], 'g-', linewidth=2)
    ax.set_xlabel('Время (Myr)')
    ax.set_ylabel('σ_v (km/s)')
    ax.set_title('Дисперсия скоростей')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(diagnostic_data['times'], diagnostic_data['cluster_radius'], 'b-', linewidth=2)
    ax.set_xlabel('Время (Myr)')
    ax.set_ylabel('Радиус (pc)')
    ax.set_title('Полумассовый радиус кластера')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'statistical_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Дополнительная статистика скопления', fontsize=16)

    ax = axes[0, 0]
    if len(animation_positions) > 1:
        initial_masses = [star['mass'] for star in animation_positions[0]]
        final_masses = [star['mass'] for star in animation_positions[-1]]

        if initial_masses and final_masses:
            ax.hist(initial_masses, bins=30, alpha=0.5, label=f'Начало (N={len(initial_masses)})',
                    color='blue', density=True)
            ax.hist(final_masses, bins=30, alpha=0.5, label=f'Конец (N={len(final_masses)})',
                    color='red', density=True)

            ax.set_xlabel('Масса (M☉)')
            ax.set_ylabel('Плотность вероятности')
            ax.set_title('Распределение масс в начальный и конечный моменты')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            ax.axvline(np.mean(initial_masses), color='blue', linestyle='--', alpha=0.7,
                       label=f'Среднее начало: {np.mean(initial_masses):.1f} M☉')
            ax.axvline(np.mean(final_masses), color='red', linestyle='--', alpha=0.7,
                       label=f'Среднее конец: {np.mean(final_masses):.1f} M☉')
            ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Нет данных о распределении масс',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title('Распределение масс')

    ax = axes[0, 1]
    if 'event_log' in globals() and event_log:
        collision_times = [event['time_Myr'] for event in event_log]
        if collision_times:
            times_unique = sorted(set(collision_times))
            cumulative = [len([t for t in collision_times if t <= time]) for time in times_unique]

            ax.plot(times_unique, cumulative, 'r-', linewidth=2)
            ax.set_xlabel('Время (Myr)')
            ax.set_ylabel('Кумулятивное число столкновений')
            ax.set_title('Эволюция числа столкновений')
            ax.grid(True, alpha=0.3)

            total_collisions = len(event_log)
            ax.annotate(f'Всего: {total_collisions} столкновений',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'Нет данных о столкновениях',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title('Эволюция числа столкновений')

    ax = axes[1, 0]
    ax.plot(diagnostic_data['times'], diagnostic_data['velocity_mag'], 'purple', linewidth=2)
    ax.set_xlabel('Время (Myr)')
    ax.set_ylabel('Скорость (km/s)')
    ax.set_title('Скорость центра масс кластера')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(diagnostic_data['times'], diagnostic_data['orbital_radius'], 'orange', linewidth=2)
    ax.set_xlabel('Время (Myr)')
    ax.set_ylabel('Радиус (pc)')
    ax.set_title('Орбитальный радиус центра масс')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'additional_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    if animation_com_positions and len(animation_com_positions) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        com_x = [pos['x'] for pos in animation_com_positions]
        com_y = [pos['y'] for pos in animation_com_positions]
        com_z = [pos['z'] for pos in animation_com_positions]

        ax.plot(com_x, com_y, com_z, 'b-', alpha=0.6, linewidth=2, label='Траектория ЦМ')

        ax.scatter(com_x[0], com_y[0], com_z[0], c='g', s=100, marker='o', label='Начало')
        ax.scatter(com_x[-1], com_y[-1], com_z[-1], c='r', s=100, marker='s', label='Конец')

        ax.scatter([0], [0], [0], c='k', s=200, marker='*', label='Галакт. центр')

        ax.set_xlabel('X (pc)')
        ax.set_ylabel('Y (pc)')
        ax.set_zlabel('Z (pc)')
        ax.set_title('Орбита центра масс кластера')
        ax.legend(loc='best')

        max_range = max(max(com_x) - min(com_x), max(com_y) - min(com_y), max(com_z) - min(com_z)) * 0.5
        mid_x = (max(com_x) + min(com_x)) * 0.5
        mid_y = (max(com_y) + min(com_y)) * 0.5
        mid_z = (max(com_z) + min(com_z)) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'cluster_orbit_3d.png'), dpi=150, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(com_x, com_y, 'b-', alpha=0.6, linewidth=2)
        axes[0].scatter(com_x[0], com_y[0], c='g', s=50, marker='o')
        axes[0].scatter(com_x[-1], com_y[-1], c='r', s=50, marker='s')
        axes[0].scatter([0], [0], c='k', s=100, marker='*')
        axes[0].set_xlabel('X (pc)')
        axes[0].set_ylabel('Y (pc)')
        axes[0].set_title('XY проекция')
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')

        axes[1].plot(com_x, com_z, 'b-', alpha=0.6, linewidth=2)
        axes[1].scatter(com_x[0], com_z[0], c='g', s=50, marker='o')
        axes[1].scatter(com_x[-1], com_z[-1], c='r', s=50, marker='s')
        axes[1].scatter([0], [0], c='k', s=100, marker='*')
        axes[1].set_xlabel('X (pc)')
        axes[1].set_ylabel('Z (pc)')
        axes[1].set_title('XZ проекция')
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')

        axes[2].plot(com_y, com_z, 'b-', alpha=0.6, linewidth=2)
        axes[2].scatter(com_y[0], com_z[0], c='g', s=50, marker='o')
        axes[2].scatter(com_y[-1], com_z[-1], c='r', s=50, marker='s')
        axes[2].scatter([0], [0], c='k', s=100, marker='*')
        axes[2].set_xlabel('Y (pc)')
        axes[2].set_ylabel('Z (pc)')
        axes[2].set_title('YZ проекция')
        axes[2].grid(True, alpha=0.3)
        axes[2].axis('equal')

        plt.suptitle('Проекции орбиты центра масс кластера')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'cluster_orbit_2d.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Графики сохранены в папку: {result_dir}")


def create_3d_animation(result_dir, animation_positions, animation_times, animation_com_positions, fps=5):
    if not animation_positions or len(animation_positions) < 2:
        print("  Недостаточно данных для создания анимации")
        return

    frames_data = []
    for frame_idx, frame_positions in enumerate(animation_positions):
        if frame_idx >= len(animation_com_positions):
            continue

        com = animation_com_positions[frame_idx]

        frame_data = {
            'x': [], 'y': [], 'z': [],
            'mass': [], 'type': [], 'remnant_type': []
        }

        for star in frame_positions:
            frame_data['x'].append(star['x'] - com['x'])
            frame_data['y'].append(star['y'] - com['y'])
            frame_data['z'].append(star['z'] - com['z'])
            frame_data['mass'].append(star['mass'])
            frame_data['type'].append(star['type'])
            frame_data['remnant_type'].append(star['remnant_type'])

        frames_data.append(frame_data)

    if not frames_data:
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    def get_star_properties(remnant_type, star_type):
        """Возвращает цвет и прозрачность для звезды"""
        remnant_str = str(remnant_type)
        type_str = str(star_type)

        if "Black Hole" in remnant_str or "BH" in remnant_str:
            return 'black', 0.8
        elif "Neutron Star" in remnant_str or "NS" in remnant_str:
            return 'red', 0.8
        elif "White Dwarf" in remnant_str or "WD" in remnant_str:
            return 'blue', 0.8
        elif "VMS" in remnant_str:
            return 'orange', 0.8
        elif "Merged" in remnant_str:
            return 'magenta', 0.8
        elif "WN" in type_str:
            return 'cyan', 0.8
        elif "O" in type_str:
            return 'green', 0.8
        else:
            # Серые звезды - низкая прозрачность
            return 'gray', 0.2  # Изменили с 0.2 на 0.3

    ax.set_xlabel('X (pc)')
    ax.set_ylabel('Y (pc)')
    ax.set_zlabel('Z (pc)')
    ax.set_title('Движение звёзд в системе центра кластера')

    all_x, all_y, all_z = [], [], []
    for frame in frames_data:
        all_x.extend(frame['x'])
        all_y.extend(frame['y'])
        all_z.extend(frame['z'])

    max_range = max(max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)) * 0.2
    mid_x = (max(all_x) + min(all_x)) * 0.2
    mid_y = (max(all_y) + min(all_y)) * 0.2
    mid_z = (max(all_z) + min(all_z)) * 0.2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    scatter_objects = {}
    type_properties = {
        'BH': ('black', 0.8),
        'NS': ('red', 0.8),
        'WD': ('blue', 0.8),
        'VMS': ('orange', 0.8),
        'Merged': ('magenta', 0.8),
        'WN': ('cyan', 0.8),
        'O': ('green', 0.8),
        'Other': ('gray', 0.25)
    }

    for type_name, (color, alpha) in type_properties.items():
        scatter_objects[type_name] = ax.scatter([], [], [],
                                                c=color,
                                                s=0.5,
                                                alpha=alpha,
                                                label=type_name)

    from matplotlib.lines import Line2D
    legend_elements = []
    for type_name, (color, alpha) in type_properties.items():
        legend_elements.append(
            Line2D([0], [0],
                   marker='o',
                   color='w',
                   label=type_name,
                   markerfacecolor=color,
                   markersize=10,
                   alpha=alpha)  # Используем ту же прозрачность в легенде
        )
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1.0))

    def update_frame(frame_idx):
        for scatter_obj in scatter_objects.values():
            scatter_obj._offsets3d = ([], [], [])

        frame = frames_data[frame_idx]

        stars_by_type = {type_name: {'x': [], 'y': [], 'z': [], 'mass': []}
                         for type_name in type_properties.keys()}

        for i in range(len(frame['x'])):
            remnant_type = str(frame['remnant_type'][i])
            star_type = str(frame['type'][i])

            if "Black Hole" in remnant_type or "BH" in remnant_type:
                type_key = 'BH'
            elif "Neutron Star" in remnant_type or "NS" in remnant_type:
                type_key = 'NS'
            elif "White Dwarf" in remnant_type or "WD" in remnant_type:
                type_key = 'WD'
            elif "VMS" in remnant_type:
                type_key = 'VMS'
            elif "Merged" in remnant_type:
                type_key = 'Merged'
            elif "WN" in star_type:
                type_key = 'WN'
            elif "O" in star_type:
                type_key = 'O'
            else:
                type_key = 'Other'

            stars_by_type[type_key]['x'].append(frame['x'][i])
            stars_by_type[type_key]['y'].append(frame['y'][i])
            stars_by_type[type_key]['z'].append(frame['z'][i])
            stars_by_type[type_key]['mass'].append(frame['mass'][i])

        for type_key, scatter_obj in scatter_objects.items():
            if stars_by_type[type_key]['x']:
                sizes = [0.5 + np.log10(max(1.0, m)) * 4 for m in stars_by_type[type_key]['mass']]
                scatter_obj._sizes = sizes
                scatter_obj._offsets3d = (
                    stars_by_type[type_key]['x'],
                    stars_by_type[type_key]['y'],
                    stars_by_type[type_key]['z']
                )

        ax.set_title(f'Движение звёзд (t={animation_times[frame_idx]:.3f} Myr, N={len(frame["x"])})')

        return list(scatter_objects.values())

    num_frames = len(frames_data)
    ani = animation.FuncAnimation(
        fig, update_frame, frames=num_frames,
        interval=1000 / fps, blit=True, repeat=True
    )

    gif_path = os.path.join(result_dir, 'cluster_3d_animation.gif')
    print(f"  Создание 3D анимации ({num_frames} кадров)...")

    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=fps, metadata=dict(artist='AMUSE Simulation'), bitrate=1800)
    ani.save(gif_path, writer=writer, dpi=100)

    plt.close(fig)
    print(f"  3D анимация сохранена: {gif_path}")

    fig_last = plt.figure(figsize=(10, 8))
    ax_last = fig_last.add_subplot(111, projection='3d')

    last_frame = frames_data[-1]
    colors = []
    alphas = []
    sizes = []

    for i in range(len(last_frame['x'])):
        remnant_type = str(last_frame['remnant_type'][i])
        star_type = str(last_frame['type'][i])

        # Используем единую функцию для получения свойств
        color, alpha = get_star_properties(remnant_type, star_type)
        colors.append(color)
        alphas.append(alpha)
        sizes.append(0.5 + np.log10(max(1.0, last_frame['mass'][i])) * 8)

    scatter = ax_last.scatter(last_frame['x'], last_frame['y'], last_frame['z'],
                              c=colors, s=sizes, alpha=alphas)

    ax_last.set_xlabel('X (pc)')
    ax_last.set_ylabel('Y (pc)')
    ax_last.set_zlabel('Z (pc)')
    ax_last.set_title(f'Финальное состояние кластера (t={animation_times[-1]:.3f} Myr, N={len(last_frame["x"])})')

    from matplotlib.lines import Line2D
    legend_elements = []
    for type_name, (color, alpha) in type_properties.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=type_name,
                   markerfacecolor=color, markersize=10, alpha=alpha)
        )

    ax_last.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1.0))

    ax_last.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_last.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_last.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'final_cluster_state_3d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_last)

    print(f"  Финальное состояние сохранено как изображение")

def run_simulation_cycle(cycle_num, total_cycles, base_params=None):
    print(f"\n{'=' * 70}")
    print(f"ЦИКЛ СИМУЛЯЦИИ {cycle_num}/{total_cycles}")
    print(f"{'=' * 70}")

    result_dir = create_result_directory()
    print(f"Результаты будут сохранены в: {result_dir}")

    if base_params is None:
        base_params = {
            'SIMULATION_END': SIMULATION_END,
            'CLUSTER_RADIUS': CLUSTER_RADIUS,
            'GAS_FRICTION_ENABLED': GAS_FRICTION_ENABLED,
            'GAS_DENSITY_PROFILE': GAS_DENSITY_PROFILE,
            'DRAG_COEFFICIENT': DRAG_COEFFICIENT,
            'COULOMB_LOGARITHM': COULOMB_LOGARITHM
        }

    global animation_positions, animation_times, animation_com_positions
    global diagnostic_data, event_log, collision_statistics

    animation_positions = []
    animation_times = []
    animation_com_positions = []

    diagnostic_data = {
        'times': [], 'n_stars': [], 'com_x': [], 'com_y': [], 'com_z': [],
        'orbital_radius': [], 'cluster_radius': [], 'velocity_mag': [],
        'n_bh': [], 'n_ns': [], 'n_wd': [], 'n_merged': [], 'n_vms': [],
        'velocity_dispersion': [], 'avg_q': [],
        'avg_friction_accel': [], 'avg_gas_density': []
    }

    collision_statistics = {
        'times': [], 'mass_distribution': [], 'velocity_dispersion': [],
        'energy': [], 'virial_ratio': [], 'escape_velocity': [],
        'q_values': [], 'retain_fractions': [], 'bh_probabilities': []
    }

    event_log = []

    try:
        print("Запуск симуляции...")

        print("\n1. BUILDING STAR POPULATION...")
        stars = build_star_population(SPECTRAL_TABLE)
        N_init = len(stars)
        total_mass = stars.mass.sum()

        print(f"   Initial stars: {N_init}")
        print(f"   Total mass: {total_mass.value_in(units.MSun):.0f} Msun")

        print("\n2. INITIALIZING CLUSTER STRUCTURE...")
        converter = nbody_system.nbody_to_si(total_mass, CLUSTER_RADIUS)
        plummer = new_plummer_model(N_init, convert_nbody=converter, virial_ratio=0.3)
        stars.position = plummer.position
        stars.velocity = plummer.velocity

        print(f"   Cluster radius: {CLUSTER_RADIUS.value_in(units.parsec):.3f} pc")

        print("\n3. SETTING ORBITAL MOTION AROUND GALACTIC CENTER...")
        inclination_rad = np.radians(INCLINATION)

        for star in stars:
            star.x += DISTANCE_TO_GALACTIC_CENTER

        orbital_velocity_y = MEASURED_VELOCITY * np.cos(inclination_rad)
        orbital_velocity_z = MEASURED_VELOCITY * np.sin(inclination_rad)

        for star in stars:
            star.vy += orbital_velocity_y
            star.vz += orbital_velocity_z


        print("\n4. INITIALIZING PH4 FOR CLUSTER GRAVITY...")
        gravity = Ph4(converter, number_of_workers=PH4_WORKERS)
        gravity.parameters.timestep_parameter = 0.01
        gravity.parameters.epsilon_squared = (0 | units.parsec) ** 2
        gravity.parameters.use_gpu = False
        gravity.particles.add_particles(stars)

        for i, star in enumerate(stars):
            if i < len(gravity.particles):
                gravity.particles[i].radius = star.radius * COLLISION_RADIUS_FACTOR

        gravity.stopping_conditions.collision_detection.enable()
        print(f"   PH4 initialized with {PH4_WORKERS} workers")

        print("\n5. INITIALIZING STELLAR EVOLUTION...")
        stellar = Seba()
        stellar_particles = Particles()
        for star in stars:
            p = Particle()
            p.mass = star.mass
            p.age = star.age
            stellar_particles.add_particle(p)
        stellar.particles.add_particles(stellar_particles)
        print(f"   Stellar evolution initialized for {len(stars)} stars")

        print("\n6. INITIALIZING GALACTIC POTENTIAL...")
        galactic_potential = MWpotentialBovy2015()

        print("\n6b. INITIALIZING GAS DYNAMICAL FRICTION...")
        gas_friction = GasDynamicalFriction(
            gas_density_profile=GAS_DENSITY_PROFILE,
            drag_coefficient=DRAG_COEFFICIENT,
            coulomb_logarithm=COULOMB_LOGARITHM
        )

        gas_friction_force = GasFrictionForce(gas_friction)
        gas_friction_force.set_tracked_particles(stars)

        print(f"   Gas friction enabled: {GAS_FRICTION_ENABLED}")
        print(f"   Gas density profile: {GAS_DENSITY_PROFILE}")
        print(f"   Drag coefficient: {DRAG_COEFFICIENT}")
        print(f"   Coulomb logarithm: {COULOMB_LOGARITHM}")

        print("\n7. SETTING UP BRIDGE INTEGRATOR WITH GAS FRICTION...")
        system_bridge = bridge.Bridge(use_threading=True)

        if GAS_FRICTION_ENABLED:
            system_bridge.add_system(gravity, (galactic_potential, gas_friction_force))
            system_bridge.add_system(gas_friction_force, ())
        else:
            system_bridge.add_system(gravity, (galactic_potential,))

        system_bridge.timestep = BRIDGE_TIMESTEP
        print(f"   Bridge timestep: {BRIDGE_TIMESTEP.value_in(units.Myr):.6f} Myr")
        print(f"   Gas friction integrated: {GAS_FRICTION_ENABLED}")

        print("\n8. INITIALIZING REALISTIC COLLISION HANDLER...")
        collision_handler = RealisticCollisionHandler(stars, gravity, stellar, event_log)

        print("\n9. SETTING UP DIAGNOSTICS SYSTEM...")

        def record_diagnostics(current_time):
            diagnostic_data['times'].append(current_time.value_in(units.Myr))
            diagnostic_data['n_stars'].append(len(stars))

            n_collapsed = 0
            for star in stars:
                if hasattr(star, 'is_remnant') and star.is_remnant:

                    n_collapsed += 1
# Добавляем в диагностические данные
            if 'n_collapsed' not in diagnostic_data:
                diagnostic_data['n_collapsed'] = []
            diagnostic_data['n_collapsed'].append(n_collapsed)

            n_bh = n_ns = n_wd = n_merged = n_vms = 0
            for star in stars:
                if star.remnant_type == "Black Hole" or star.spectral_label == "BH":
                    n_bh += 1
                elif star.remnant_type == "Neutron Star" or star.spectral_label == "NS":
                    n_ns += 1
                elif star.remnant_type == "White Dwarf" or star.spectral_label == "WD":
                    n_wd += 1
                elif star.remnant_type == "Very Massive Star" or star.spectral_label == "VMS":
                    n_vms += 1
                elif "Merged" in str(star.spectral_label) or "Merged" in str(star.remnant_type):
                    n_merged += 1

            diagnostic_data['n_bh'].append(n_bh)
            diagnostic_data['n_ns'].append(n_ns)
            diagnostic_data['n_wd'].append(n_wd)
            diagnostic_data['n_merged'].append(n_merged)
            diagnostic_data['n_vms'].append(n_vms)

            if len(stars) > 0:
                com = stars.center_of_mass()
                vel = stars.center_of_mass_velocity()

                diagnostic_data['com_x'].append(com[0].value_in(units.parsec))
                diagnostic_data['com_y'].append(com[1].value_in(units.parsec))
                diagnostic_data['com_z'].append(com[2].value_in(units.parsec))

                r = np.sqrt(com[0] ** 2 + com[1] ** 2 + com[2] ** 2)
                diagnostic_data['orbital_radius'].append(r.value_in(units.parsec))

                v = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
                diagnostic_data['velocity_mag'].append(v.value_in(units.kms))

                half_mass_r = calculate_half_mass_radius(stars)
                diagnostic_data['cluster_radius'].append(half_mass_r.value_in(units.parsec))

                velocities = []
                for star in stars:
                    vx = star.vx - vel[0]
                    vy = star.vy - vel[1]
                    vz = star.vz - vel[2]
                    v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2).value_in(units.kms)
                    velocities.append(v)

                if len(velocities) > 1:
                    diagnostic_data['velocity_dispersion'].append(np.std(velocities))
                else:
                    diagnostic_data['velocity_dispersion'].append(0.0)

                if collision_handler.q_distribution:
                    recent_q = collision_handler.q_distribution[-10:] if len(
                        collision_handler.q_distribution) >= 10 else collision_handler.q_distribution
                    diagnostic_data['avg_q'].append(np.mean(recent_q))
                else:
                    diagnostic_data['avg_q'].append(0.0)

                diagnostic_data['avg_friction_accel'].append(
                    gas_friction.avg_friction_accel.value_in(units.kms / units.Myr)
                )

                densities = []
                for star in stars:
                    pos = [star.x, star.y, star.z]
                    density = gas_friction.density_profile(pos)
                    densities.append(density.value_in(units.MSun / units.parsec ** 3))
                diagnostic_data['avg_gas_density'].append(np.mean(densities) if densities else 0.0)

        def record_animation_frame(stars, current_time):
            if len(animation_positions) < MAX_FRAMES:
                positions = []
                for star in stars:
                    positions.append({
                        'x': star.x.value_in(units.parsec),
                        'y': star.y.value_in(units.parsec),
                        'z': star.z.value_in(units.parsec),
                        'mass': star.mass.value_in(units.MSun),
                        'type': star.spectral_label,
                        'remnant_type': star.remnant_type
                    })
                animation_positions.append(positions)
                animation_times.append(current_time.value_in(units.Myr))

                if len(stars) > 0:
                    com = stars.center_of_mass()
                    animation_com_positions.append({
                        'x': com[0].value_in(units.parsec),
                        'y': com[1].value_in(units.parsec),
                        'z': com[2].value_in(units.parsec)
                    })
                else:
                    animation_com_positions.append({'x': 0, 'y': 0, 'z': 0})

        print("\n" + "=" * 70)
        print("STARTING MAIN SIMULATION LOOP")
        print("=" * 70)

        start_time = time.time()
        model_time = 0.0 | units.Myr
        next_diagnostic = 0.0 | units.Myr
        next_animation = 0.0 | units.Myr
        animation_target_times = np.linspace(0, SIMULATION_END.value_in(units.Myr), MAX_FRAMES) | units.Myr
        animation_frame_index = 0
        animation_interval = SIMULATION_END / MAX_FRAMES
        step_counter = 0

        print(f"\nInitial setup:")
        print(f"  Number of stars: {len(stars)}")
        print(f"  Total mass: {stars.mass.sum().value_in(units.MSun):.1f} Msun")

        record_diagnostics(model_time)
        record_animation_frame(stars, model_time)

        print("\nSimulation Progress:")
        print(
            "Step | Time [Myr] | N_stars | BH | NS | WD | Merged | VMS | R_cluster [pc] | σ_v [km/s] | F_fric [km/s/Myr] | Collisions")
        print("-" * 130)

        try:
            while model_time < SIMULATION_END:
                dt = BRIDGE_TIMESTEP
                target_time = model_time + dt
                step_counter += 1

                stellar.evolve_model(target_time)

                collision_handler.sync_gravity_with_stars()

                system_bridge.evolve_model(target_time)

                if GAS_FRICTION_ENABLED:
                    dt = BRIDGE_TIMESTEP
                    accelerations = gas_friction.calculate_accelerations(stars)

                    for i, star in enumerate(stars):
                        if i < len(accelerations):
                            ax, ay, az = accelerations[i]
                            star.vx += ax * dt
                            star.vy += ay * dt
                            star.vz += az * dt

                    collision_handler.sync_gravity_with_stars()

                model_time = system_bridge.model_time

                temp_particles = Particles()
                for i in range(len(stars)):
                    if i < len(gravity.particles):
                        p = Particle()
                        p.x = gravity.particles[i].x
                        p.y = gravity.particles[i].y
                        p.z = gravity.particles[i].z
                        p.vx = gravity.particles[i].vx
                        p.vy = gravity.particles[i].vy
                        p.vz = gravity.particles[i].vz
                        temp_particles.add_particle(p)

                if len(temp_particles) == len(stars):
                    for i, p in enumerate(temp_particles):
                        stars[i].x = p.x
                        stars[i].y = p.y
                        stars[i].z = p.z
                        stars[i].vx = p.vx
                        stars[i].vy = p.vy
                        stars[i].vz = p.vz

                collisions = collision_handler.resolve_collisions(model_time)

                if step_counter % COLLAPSE_CHECK_INTERVAL == 0:
                    collapsed = simulate_stellar_collapse(stars, model_time, gravity)
                    if collapsed > 0:
                        collision_handler.sync_gravity_with_stars()
                        print(f"[ЦИКЛ] На шаге {step_counter} коллапсировало {collapsed} звёзд.")

                if model_time >= next_diagnostic:
                    record_diagnostics(model_time)
                    next_diagnostic += DIAGNOSTIC_DT

                if model_time >= next_diagnostic:
                    record_diagnostics(model_time)
                    next_diagnostic += DIAGNOSTIC_DT

                if animation_frame_index < len(animation_target_times) and model_time >= animation_target_times[
                    animation_frame_index]:
                    record_animation_frame(stars, model_time)
                    animation_frame_index += 1

                if step_counter % 10 == 0 or collisions > 0:
                    n_bh = diagnostic_data['n_bh'][-1] if diagnostic_data['n_bh'] else 0
                    n_ns = diagnostic_data['n_ns'][-1] if diagnostic_data['n_ns'] else 0
                    n_wd = diagnostic_data['n_wd'][-1] if diagnostic_data['n_wd'] else 0
                    n_merged = diagnostic_data['n_merged'][-1] if diagnostic_data['n_merged'] else 0
                    n_vms = diagnostic_data['n_vms'][-1] if diagnostic_data['n_vms'] else 0

                    cluster_radius = diagnostic_data['cluster_radius'][-1] if diagnostic_data['cluster_radius'] else 0

                    if diagnostic_data['velocity_dispersion'] and len(diagnostic_data['velocity_dispersion']) > 0:
                        velocity_disp = diagnostic_data['velocity_dispersion'][-1]
                    else:
                        velocity_disp = 0.0

                    avg_friction_accel = diagnostic_data['avg_friction_accel'][-1] if diagnostic_data[
                        'avg_friction_accel'] else 0.0

                    print(f"  {step_counter:4d} | {model_time.value_in(units.Myr):7.4f} | {len(stars):7d} | "
                          f"{n_bh:2d} | {n_ns:2d} | {n_wd:2d} | {n_merged:6d} | {n_vms:3d} | "
                          f"{cluster_radius:8.4f} | {velocity_disp:5.1f} | {avg_friction_accel:10.3e} | {collision_handler.collision_counter:4d}")

        except Exception as e:
            print(f"\nERROR during simulation: {str(e)}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 70)
        print("SIMULATION COMPLETED")
        print("=" * 70)

        for code, name in [(stellar, "stellar evolution"), (gravity, "PH4"), (system_bridge, "Bridge")]:
            try:
                code.stop()
                print(f"{name} code stopped")
            except Exception as e:
                print(f"Error stopping {name}: {e}")

        end_time = time.time()
        simulation_time = end_time - start_time

        print(f"\nPERFORMANCE STATISTICS:")
        print(f"  Total simulation time: {simulation_time:.2f} seconds")
        print(f"  Total steps: {step_counter}")
        if simulation_time > 0:
            print(f"  Steps per second: {step_counter / simulation_time:.1f}")

        if GAS_FRICTION_ENABLED:
            gas_friction_force.print_statistics()

        print("\n" + "=" * 70)
        print("REALISTIC COLLISION PHYSICS STATISTICS")
        print("=" * 70)

        if collision_handler.collision_counter > 0:
            collision_handler.print_q_statistics()

            remnant_types = {}
            for event in event_log:
                remnant = event['remnant_type']
                remnant_types[remnant] = remnant_types.get(remnant, 0) + 1

            print(f"\nREMNANT DISTRIBUTION:")
            for remnant, count in sorted(remnant_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / collision_handler.collision_counter) * 100
                print(f"  {remnant}: {count} ({percentage:.1f}%)")

            if collision_handler.retain_fractions:
                retain_fracs = np.array(collision_handler.retain_fractions)
                print(f"\n=== RETAIN FRACTIONS ===")
                print(f"Mean retain_frac: {np.mean(retain_fracs):.3f}")
                print(f"Min retain_frac: {np.min(retain_fracs):.3f}")
                print(f"Max retain_frac: {np.max(retain_fracs):.3f}")

            if collision_handler.bh_probabilities:
                bh_probs = np.array(collision_handler.bh_probabilities)
                print(f"\n=== BH PROBABILITIES ===")
                print(f"Mean bh_prob: {np.mean(bh_probs):.3f}")
                print(f"Min bh_prob: {np.min(bh_probs):.3f}")
                print(f"Max bh_prob: {np.max(bh_probs):.3f}")

                bh_formed = sum(1 for event in event_log if event['remnant_type'] == 'Black Hole')
                print(
                    f"Actual BH formations: {bh_formed} ({bh_formed / collision_handler.collision_counter * 100:.1f}% of collisions)")

        print(f"\nDETAILED SIMULATION STATISTICS:")
        print(f"  Initial stars: {N_init}")
        print(f"  Final stars: {len(stars)}")
        print(f"  Total collisions: {collision_handler.collision_counter}")
        if GAS_FRICTION_ENABLED:
            print(
                f"  Average gas density: {np.mean(diagnostic_data['avg_gas_density']) if diagnostic_data['avg_gas_density'] else 0:.1f} Msun/pc³")
            print(
                f"  Average friction acceleration: {np.mean(diagnostic_data['avg_friction_accel']) if diagnostic_data['avg_friction_accel'] else 0:.3e} km/s/Myr")

        print("\nSaving results...")

        if event_log:
            df_events = pd.DataFrame(event_log)
            df_events.to_csv(os.path.join(result_dir, 'collision_events.csv'), index=False)
            print(f"  Saved {len(event_log)} collision events")

        if diagnostic_data['times']:
            df_diag = pd.DataFrame({
                'time_Myr': diagnostic_data['times'],
                'N_stars': diagnostic_data['n_stars'],
                'N_BH': diagnostic_data['n_bh'],
                'N_NS': diagnostic_data['n_ns'],
                'N_WD': diagnostic_data['n_wd'],
                'N_Merged': diagnostic_data['n_merged'],
                'N_VMS': diagnostic_data['n_vms'],
                'COM_x_pc': diagnostic_data['com_x'],
                'COM_y_pc': diagnostic_data['com_y'],
                'COM_z_pc': diagnostic_data['com_z'],
                'orbital_radius_pc': diagnostic_data['orbital_radius'],
                'velocity_kms': diagnostic_data['velocity_mag'],
                'cluster_radius_pc': diagnostic_data['cluster_radius'],
                'velocity_dispersion_kms': diagnostic_data['velocity_dispersion'],
                'avg_q': diagnostic_data['avg_q'],
                'avg_friction_accel_km_s_Myr': diagnostic_data['avg_friction_accel'],
                'avg_gas_density_Msun_pc3': diagnostic_data['avg_gas_density']
            })
            df_diag.to_csv(os.path.join(result_dir, 'simulation_diagnostics.csv'), index=False)
            print(f"  Saved diagnostic data (including gas friction)")

        if collision_statistics['q_values']:
            stats_df = pd.DataFrame({
                'collision_id': range(1, len(collision_statistics['q_values']) + 1),
                'q_value': collision_statistics['q_values'],
                'retain_fraction': collision_statistics['retain_fractions'],
                'bh_probability': collision_statistics['bh_probabilities']
            })
            stats_df.to_csv(os.path.join(result_dir, 'q_statistics.csv'), index=False)
            print(f"  Saved q statistics")

        if GAS_FRICTION_ENABLED and gas_friction.friction_events > 0:
            gas_stats = {
                'total_events': gas_friction.friction_events,
                'total_accel_kms_Myr': gas_friction.total_friction_accel.value_in(units.kms / units.Myr),
                'max_accel_kms_Myr': gas_friction.max_friction_accel.value_in(units.kms / units.Myr),
                'avg_accel_kms_Myr': (gas_friction.total_friction_accel / gas_friction.friction_events).value_in(
                    units.kms / units.Myr),
                'drag_coefficient': DRAG_COEFFICIENT,
                'coulomb_log': COULOMB_LOGARITHM,
                'density_profile': GAS_DENSITY_PROFILE,
                'avg_friction_accel_km_s_Myr': np.mean(diagnostic_data['avg_friction_accel']) if diagnostic_data[
                    'avg_friction_accel'] else 0.0
            }
            import json

            with open(os.path.join(result_dir, 'gas_friction_statistics.json'), 'w') as f:
                json.dump(gas_stats, f, indent=2)
            print(f"  Saved gas friction statistics")

        save_statistical_plots(result_dir, diagnostic_data, animation_positions,
                               animation_times, animation_com_positions)

        create_3d_animation(result_dir, animation_positions, animation_times,
                            animation_com_positions, fps=10)

        params_file = os.path.join(result_dir, 'simulation_parameters.txt')
        with open(params_file, 'w') as f:
            f.write(f"ПАРАМЕТРЫ СИМУЛЯЦИИ (ЦИКЛ {cycle_num})\n")
            f.write("=" * 50 + "\n")
            f.write(f"Время симуляции: {SIMULATION_END}\n")
            f.write(f"Радиус кластера: {CLUSTER_RADIUS}\n")
            f.write(f"Число начальных звёзд: {N_init}\n")
            f.write(f"Газовое трение: {GAS_FRICTION_ENABLED}\n")
            f.write(f"Профиль плотности газа: {GAS_DENSITY_PROFILE}\n")
            f.write(f"Коэффициент сопротивления: {DRAG_COEFFICIENT}\n")
            f.write(f"Кулоновский логарифм: {COULOMB_LOGARITHM}\n")
            f.write(f"Шаг диагностики: {DIAGNOSTIC_DT}\n")
            f.write(f"Шаг Bridge: {BRIDGE_TIMESTEP}\n")

        print(f"\nВсе результаты цикла {cycle_num} сохранены в: {result_dir}")
        print(f"Общее время выполнения цикла: {time.time() - start_time:.1f} секунд")

        return result_dir

    except Exception as e:
        print(f"Ошибка в цикле {cycle_num}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_multiple_simulations(num_cycles=2):
    print(f"\n{'=' * 70}")
    print(f"ЗАПУСК МНОГОЦИКЛИЧНОЙ СИМУЛЯЦИИ ({num_cycles} циклов)")
    print(f"{'=' * 70}")

    base_params = {
        'SIMULATION_END': SIMULATION_END,
        'CLUSTER_RADIUS': CLUSTER_RADIUS,
        'GAS_FRICTION_ENABLED': GAS_FRICTION_ENABLED,
        'GAS_DENSITY_PROFILE': GAS_DENSITY_PROFILE,
        'DRAG_COEFFICIENT': DRAG_COEFFICIENT,
        'COULOMB_LOGARITHM': COULOMB_LOGARITHM,
        'DIAGNOSTIC_DT': DIAGNOSTIC_DT,
        'BRIDGE_TIMESTEP': BRIDGE_TIMESTEP
    }

    results_dirs = []
    total_start_time = time.time()

    for cycle in range(1, num_cycles + 1):
        cycle_params = base_params.copy()

        result_dir = run_simulation_cycle(cycle, num_cycles, cycle_params)
        if result_dir:
            results_dirs.append(result_dir)

    print(f"\n{'=' * 70}")
    print(f"СВОДНЫЙ ОТЧЕТ ({len(results_dirs)} успешных циклов)")
    print(f"{'=' * 70}")

    for i, dir_path in enumerate(results_dirs, 1):
        print(f"Цикл {i}: {dir_path}")

    if len(results_dirs) > 1:
        create_summary_comparison(results_dirs)

    total_time = time.time() - total_start_time
    print(f"\nОбщее время выполнения всех циклов: {total_time:.1f} секунд")
    print(f"Среднее время на цикл: {total_time / num_cycles:.1f} секунд")

    return results_dirs

def create_summary_comparison(results_dirs):
    print("\nСоздание сводного графика сравнения циклов...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Сравнение результатов всех циклов симуляции', fontsize=16)

    all_cycle_data = []

    for i, dir_path in enumerate(results_dirs, 1):
        stats_file = os.path.join(dir_path, 'simulation_diagnostics.csv')
        if os.path.exists(stats_file):
            df = pd.read_csv(stats_file)
            all_cycle_data.append((i, df))

    if len(all_cycle_data) < 2:
        print("  Недостаточно данных для сравнения")
        return

    ax = axes[0, 0]
    for cycle_num, df in all_cycle_data:
        ax.plot(df['time_Myr'], df['N_stars'], '-', linewidth=2, alpha=0.7, label=f'Цикл {cycle_num}')
    ax.set_xlabel('Время (Myr)')
    ax.set_ylabel('Число звёзд')
    ax.set_title('Эволюция числа объектов')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for cycle_num, df in all_cycle_data:
        ax.plot(df['time_Myr'], df['N_BH'], '-', linewidth=2, alpha=0.7, label=f'Цикл {cycle_num}')
    ax.set_xlabel('Время (Myr)')
    ax.set_ylabel('Число чёрных дыр')
    ax.set_title('Эволюция числа ЧД')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for cycle_num, df in all_cycle_data:
        ax.plot(df['time_Myr'], df['velocity_dispersion_kms'], '-', linewidth=2, alpha=0.7, label=f'Цикл {cycle_num}')
    ax.set_xlabel('Время (Myr)')
    ax.set_ylabel('σ_v (km/s)')
    ax.set_title('Дисперсия скоростей')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for cycle_num, df in all_cycle_data:
        ax.plot(df['time_Myr'], df['cluster_radius_pc'], '-', linewidth=2, alpha=0.7, label=f'Цикл {cycle_num}')
    ax.set_xlabel('Время (Myr)')
    ax.set_ylabel('Радиус (pc)')
    ax.set_title('Полумассовый радиус')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    main_dir = os.path.dirname(results_dirs[0])
    summary_path = os.path.join(main_dir, 'summary_comparison.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Сводный график сохранен: {summary_path}")

def build_star_population(table):
    stars = Particles()

    CLUSTER_METALLICITY = 0.008

    for label, count, ref_mass, init_mass, mloss in table:
        for i in range(count):
            p = Particle()
            p.mass = init_mass | units.MSun
            p.metallicity = CLUSTER_METALLICITY

            if "WN" in label:
                p.radius = ((init_mass) ** 0.5) | units.RSun
            elif "O" in label and "I" in label:
                p.radius = ((init_mass) ** 0.6) | units.RSun
            elif "O" in label and "V" in label:
                p.radius = ((init_mass) ** 0.8) | units.RSun
            elif "la" in label or "lb" in label:
                p.radius = ((init_mass) ** 0.7) | units.RSun
            else:
                p.radius = ((init_mass) ** 0.75) | units.RSun

            p.age = np.random.uniform(0, 2.0) | units.Myr
            p.spectral_label = label
            p.remnant_type = ""
            p.collision_count = 0
            p.accreted_mass = 0.0 | units.MSun

            stars.add_particle(p)

    return stars

def calculate_half_mass_radius(particles):
    if len(particles) == 0:
        return 0 | units.parsec

    com = particles.center_of_mass()
    distances = []
    for p in particles:
        dx = p.x - com[0]
        dy = p.y - com[1]
        dz = p.z - com[2]
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        distances.append((dist.value_in(units.parsec), p.mass.value_in(units.MSun)))

    distances.sort(key=lambda x: x[0])
    total_mass = sum(m for _, m in distances)
    half_mass = total_mass / 2

    cumulative_mass = 0
    for dist, mass in distances:
        cumulative_mass += mass
        if cumulative_mass >= half_mass:
            return dist | units.parsec

    return distances[-1][0] | units.parsec

def calculate_cluster_statistics(stars, current_time):
    if len(stars) == 0:
        return

    masses = [star.mass.value_in(units.MSun) for star in stars]
    collision_statistics['mass_distribution'].append(masses)

    velocities = []
    com_vel = stars.center_of_mass_velocity()
    for star in stars:
        vx = star.vx - com_vel[0]
        vy = star.vy - com_vel[1]
        vz = star.vz - com_vel[2]
        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2).value_in(units.kms)
        velocities.append(v)

    if len(velocities) > 1:
        velocity_dispersion = np.std(velocities)
    else:
        velocity_dispersion = 0.0
    collision_statistics['velocity_dispersion'].append(velocity_dispersion)

    total_mass = stars.mass.sum().value_in(units.MSun)
    half_mass_r = calculate_half_mass_radius(stars).value_in(units.parsec)
    if half_mass_r > 0:
        escape_velocity = np.sqrt(2 * constants.G.value_in(units.parsec ** 3 / units.MSun / units.Myr ** 2) *
                                  total_mass / half_mass_r) * (units.parsec / units.Myr).value_in(units.kms)
    else:
        escape_velocity = 0
    collision_statistics['escape_velocity'].append(escape_velocity)

    collision_statistics['times'].append(current_time.value_in(units.Myr))

class RealisticCollisionHandler:
    def __init__(self, stars, gravity, stellar, event_log):
        self.stars = stars
        self.gravity = gravity
        self.stellar = stellar
        self.event_log = event_log
        self.collision_counter = 0
        self.last_collision_time = -1.0 | units.Myr

        self.q_distribution = []
        self.retain_fractions = []
        self.bh_probabilities = []

        self.channel_stars_to_gravity = stars.new_channel_to(gravity.particles)
        self.channel_gravity_to_stars = gravity.particles.new_channel_to(stars)

    def get_collision_recipe(self, spectral_label):
        label_str = str(spectral_label)

        if "BH" in label_str or "Black Hole" in label_str:
            return COLLISION_RECIPES["BH"]
        elif "NS" in label_str or "Neutron Star" in label_str:
            return COLLISION_RECIPES["NS"]
        elif "WD" in label_str or "White Dwarf" in label_str:
            return COLLISION_RECIPES["WD"]
        elif "VMS" in label_str or "Very Massive Star" in label_str:
            return COLLISION_RECIPES["VMS"]

        if "Merged" in label_str:
            return COLLISION_RECIPES["Merged"]

        if "WN" in label_str:
            return COLLISION_RECIPES["WN"]
        elif "O" in label_str:
            return COLLISION_RECIPES["O"]
        elif "Unclassified" in label_str:
            return COLLISION_RECIPES["Unclassified"]

        return COLLISION_RECIPES_DEFAULT

    def find_star_by_gravity_particle(self, gravity_particle):
        try:
            gravity_particles = list(self.gravity.particles)
            idx = gravity_particles.index(gravity_particle)
            if idx < len(self.stars):
                return self.stars[idx]
        except (ValueError, IndexError):
            pass

        for star in self.stars:
            try:
                if (abs(star.x - gravity_particle.x) < 0.01 | units.parsec and
                        abs(star.y - gravity_particle.y) < 0.01 | units.parsec and
                        abs(star.z - gravity_particle.z) < 0.01 | units.parsec):
                    return star
            except:
                continue
        return None

    def calculate_remnant_radius(self, remnant_type, mass, recipe):
        mass_val = mass.value_in(units.MSun)

        if remnant_type == "Black Hole":
            return 2 * constants.G * mass / constants.c ** 2
        elif remnant_type == "Neutron Star":
            return 12 | units.km
        elif remnant_type == "White Dwarf":
            return 0.01 | units.RSun
        elif remnant_type == "Very Massive Star":
            return (mass_val ** 0.6) | units.RSun
        else:
            exponent = recipe.get("radius_exponent", 0.75)
            return (mass_val ** exponent) | units.RSun

    def handle_collision_pair(self, p1, p2, current_time):
        star1 = self.find_star_by_gravity_particle(p1)
        star2 = self.find_star_by_gravity_particle(p2)

        if star1 is None or star2 is None or star1 is star2:
            return None

        try:
            star1_is_bh = ("BH" in str(star1.spectral_label) or
                           "Black Hole" in str(star1.remnant_type))
            star2_is_bh = ("BH" in str(star2.spectral_label) or
                           "Black Hole" in str(star2.remnant_type))

            if star1_is_bh or star2_is_bh:
                if star1_is_bh and star2_is_bh:
                    primary = star1 if star1.mass >= star2.mass else star2
                    secondary = star2 if star1.mass >= star2.mass else star1
                    recipe = COLLISION_RECIPES["BH"]
                elif star1_is_bh:
                    primary = star1
                    secondary = star2
                    recipe = COLLISION_RECIPES["BH"]
                else:
                    primary = star2
                    secondary = star1
                    recipe = COLLISION_RECIPES["BH"]

                total_mass = primary.mass + secondary.mass
                total_mass_val = total_mass.value_in(units.MSun)

                retain_frac = 1.0
                new_mass = total_mass
                new_mass_val = new_mass.value_in(units.MSun)

                remnant_type = "Black Hole"
                spectral_label = "BH"

                q = secondary.mass / primary.mass

                self.q_distribution.append(q)
                collision_statistics['q_values'].append(q)
                self.retain_fractions.append(retain_frac)
                collision_statistics['retain_fractions'].append(retain_frac)
                bh_prob = 1.0
                self.bh_probabilities.append(bh_prob)
                collision_statistics['bh_probabilities'].append(bh_prob)

                new_star = Particle()
                new_star.mass = new_mass

                new_star.x = (primary.x * primary.mass + secondary.x * secondary.mass) / total_mass
                new_star.y = (primary.y * primary.mass + secondary.y * secondary.mass) / total_mass
                new_star.z = (primary.z * primary.mass + secondary.z * secondary.mass) / total_mass
                new_star.vx = (primary.vx * primary.mass + secondary.vx * secondary.mass) / total_mass
                new_star.vy = (primary.vy * primary.mass + secondary.vy * secondary.mass) / total_mass
                new_star.vz = (primary.vz * primary.mass + secondary.vz * secondary.mass) / total_mass

                new_star.age = 0.0 | units.Myr

                new_star.spectral_label = spectral_label
                new_star.remnant_type = remnant_type
                new_star.metallicity = getattr(primary, 'metallicity', 0.008)
                new_star.collision_count = getattr(primary, 'collision_count', 0) + getattr(secondary,
                                                                                            'collision_count', 0) + 1
                new_star.accreted_mass = getattr(primary, 'accreted_mass', 0.0 | units.MSun) + getattr(secondary,
                                                                                                       'accreted_mass',
                                                                                                       0.0 | units.MSun) + secondary.mass

                new_star.radius = 2 * constants.G * new_mass / constants.c ** 2

                secondary_type = str(secondary.spectral_label)
                if "BH" in secondary_type or "Black Hole" in secondary_type:
                    event_type = "BH-BH merger"
                else:
                    event_type = "BH accretion"

                event = {
                    "time_Myr": current_time.value_in(units.Myr),
                    "collision_id": self.collision_counter + 1,
                    "primary_type": str(primary.spectral_label),
                    "secondary_type": secondary_type,
                    "primary_mass": primary.mass.value_in(units.MSun),
                    "secondary_mass": secondary.mass.value_in(units.MSun),
                    "q_value": float(q),
                    "total_mass": total_mass_val,
                    "retain_frac": float(retain_frac),
                    "hydrogen_factor": 1.0,
                    "new_mass": new_mass_val,
                    "bh_probability": float(bh_prob),
                    "remnant_type": remnant_type,
                    "spectral_label": spectral_label,
                    "mass_loss_percent": 0.0,
                    "metallicity": float(getattr(primary, 'metallicity', 0.008)),
                    "recipe_type": "BH_accretion",
                    "collision_count": new_star.collision_count,
                    "is_merger": True,
                    "event_type": event_type
                }
                self.event_log.append(event)

                self.collision_counter += 1

                print(f"[BH COLLISION #{self.collision_counter}] "
                      f"BH({primary.mass.value_in(units.MSun):.1f}M☉) + "
                      f"{secondary_type}({secondary.mass.value_in(units.MSun):.1f}M☉) → "
                      f"BH({new_mass.value_in(units.MSun):.1f}M☉) | "
                      f"retain=1.000, bh_prob=1.000")

                return (primary, secondary, new_star)

            else:
                if star1.mass > star2.mass:
                    primary, secondary = star1, star2
                else:
                    primary, secondary = star2, star1

                q = secondary.mass / primary.mass

                self.q_distribution.append(q)
                collision_statistics['q_values'].append(q)

                recipe = self.get_collision_recipe(primary.spectral_label)

                total_mass = primary.mass + secondary.mass
                total_mass_val = total_mass.value_in(units.MSun)

                if "retain_fixed" in recipe:
                    retain_frac = recipe["retain_fixed"]
                elif recipe.get("retain_model") == "reinoso":
                    retain_frac = retain_frac_reinoso(q)
                else:
                    retain_frac = retain_frac_from_q(q)

                hydrogen_factor = recipe.get("hydrogen_retain", HYDROGEN_RETAIN_DEFAULT)
                retain_frac *= hydrogen_factor

                self.retain_fractions.append(retain_frac)
                collision_statistics['retain_fractions'].append(retain_frac)

                new_mass = total_mass * retain_frac
                new_mass_val = new_mass.value_in(units.MSun)

                if "bh_fixed" in recipe:
                    bh_prob = recipe["bh_fixed"]
                else:
                    bh_prob = bh_probability_from_q_mass(q, total_mass_val)

                self.bh_probabilities.append(bh_prob)
                collision_statistics['bh_probabilities'].append(bh_prob)

                stellar_type = str(primary.spectral_label)
                metallicity = getattr(primary, 'metallicity', 0.008)
                is_merger = ("Merged" in stellar_type or
                             getattr(primary, 'collision_count', 0) > 0 or
                             getattr(secondary, 'collision_count', 0) > 0)

                remnant_type, spectral_label = determine_remnant_type_realistic(
                    mass_val=new_mass_val,
                    metallicity=metallicity,
                    stellar_type=stellar_type,
                    bh_prob=bh_prob,
                    current_time=current_time,
                    star_age=min(primary.age, secondary.age),
                    is_merger=is_merger
                )

                new_star = Particle()
                new_star.mass = new_mass

                new_star.x = (primary.x * primary.mass + secondary.x * secondary.mass) / total_mass
                new_star.y = (primary.y * primary.mass + secondary.y * secondary.mass) / total_mass
                new_star.z = (primary.z * primary.mass + secondary.z * secondary.mass) / total_mass
                new_star.vx = (primary.vx * primary.mass + secondary.vx * secondary.mass) / total_mass
                new_star.vy = (primary.vy * primary.mass + secondary.vy * secondary.mass) / total_mass
                new_star.vz = (primary.vz * primary.mass + secondary.vz * secondary.mass) / total_mass

                new_star.age = min(primary.age, secondary.age)
                if remnant_type in ["Black Hole", "Neutron Star", "White Dwarf"]:
                    new_star.age = 0.0 | units.Myr

                new_star.spectral_label = spectral_label
                new_star.remnant_type = remnant_type
                new_star.metallicity = metallicity
                new_star.collision_count = getattr(primary, 'collision_count', 0) + getattr(secondary,
                                                                                            'collision_count', 0) + 1
                new_star.accreted_mass = getattr(primary, 'accreted_mass', 0.0 | units.MSun) + getattr(secondary,
                                                                                                       'accreted_mass',
                                                                                                       0.0 | units.MSun) + secondary.mass

                new_star.radius = self.calculate_remnant_radius(remnant_type, new_mass, recipe)

                event = {
                    "time_Myr": current_time.value_in(units.Myr),
                    "collision_id": self.collision_counter + 1,
                    "primary_type": str(primary.spectral_label),
                    "secondary_type": str(secondary.spectral_label),
                    "primary_mass": primary.mass.value_in(units.MSun),
                    "secondary_mass": secondary.mass.value_in(units.MSun),
                    "q_value": float(q),
                    "total_mass": total_mass_val,
                    "retain_frac": float(retain_frac),
                    "hydrogen_factor": float(hydrogen_factor),
                    "new_mass": new_mass_val,
                    "bh_probability": float(bh_prob),
                    "remnant_type": remnant_type,
                    "spectral_label": spectral_label,
                    "mass_loss_percent": (1 - retain_frac) * 100,
                    "metallicity": float(metallicity),
                    "recipe_type": recipe.get("type", "Unknown"),
                    "collision_count": new_star.collision_count,
                    "is_merger": is_merger,
                    "event_type": "normal_collision"
                }
                self.event_log.append(event)

                self.collision_counter += 1

                print(f"[COLLISION #{self.collision_counter}] "
                      f"Primary: {primary.spectral_label}({primary.mass.value_in(units.MSun):.1f}M☉) + "
                      f"Secondary: {secondary.spectral_label}({secondary.mass.value_in(units.MSun):.1f}M☉) | "
                      f"q={q:.3f} → "
                      f"{remnant_type}({new_mass.value_in(units.MSun):.1f}M☉) | "
                      f"retain={retain_frac:.3f} (H={hydrogen_factor:.2f}), "
                      f"bh_prob={bh_prob:.3f}")

                return (primary, secondary, new_star)

        except Exception as e:
            print(f"[ERROR] Failed to handle collision: {e}")
            import traceback
            traceback.print_exc()
            return None

    def resolve_collisions(self, current_time):
        stop_cond = self.gravity.stopping_conditions.collision_detection

        if not stop_cond.is_set():
            return 0

        colliding_particles1 = stop_cond.particles(0)
        colliding_particles2 = stop_cond.particles(1)
        collisions_count = len(colliding_particles1)

        print(f"[COLLISION] Detected {collisions_count} collision pairs")

        if collisions_count == 0:
            return 0

        collisions_to_process = []
        for i in range(collisions_count):
            p1 = colliding_particles1[i]
            p2 = colliding_particles2[i]
            result = self.handle_collision_pair(p1, p2, current_time)
            if result is not None:
                collisions_to_process.append(result)

        if collisions_to_process:
            stars_to_remove = Particles()
            new_stars = Particles()
            stellar_particles_to_remove = Particles()
            stellar_particles_to_add = Particles()

            for star1, star2, new_star in collisions_to_process:
                stars_to_remove.add_particle(star1)
                stars_to_remove.add_particle(star2)
                new_stars.add_particle(new_star)

                for particle_in_stellar in self.stellar.particles:
                    if particle_in_stellar.mass == star1.mass and particle_in_stellar.age == star1.age:
                        stellar_particles_to_remove.add_particle(particle_in_stellar)
                        break
                for particle_in_stellar in self.stellar.particles:
                    if particle_in_stellar.mass == star2.mass and particle_in_stellar.age == star2.age:
                        stellar_particles_to_remove.add_particle(particle_in_stellar)
                        break

                new_stellar_particle = Particle()
                new_stellar_particle.mass = new_star.mass
                if new_star.remnant_type in ["Black Hole", "Neutron Star"]:
                    new_stellar_particle.age = 0.0 | units.Myr
                else:
                    new_stellar_particle.age = new_star.age

                stellar_particles_to_add.add_particle(new_stellar_particle)

            if len(stellar_particles_to_remove) > 0:
                self.stellar.particles.remove_particles(stellar_particles_to_remove)

            if len(stellar_particles_to_add) > 0:
                self.stellar.particles.add_particles(stellar_particles_to_add)

            self.stars.remove_particles(stars_to_remove)
            self.stars.add_particles(new_stars)

            self.last_collision_time = current_time
            print(f"[INFO] Processed {len(collisions_to_process)} collisions. Total stars: {len(self.stars)}")

        stop_cond.disable()
        stop_cond.enable()
        return len(collisions_to_process)

    def sync_gravity_with_stars(self):
        if len(self.gravity.particles) != len(self.stars):
            self.gravity.particles.remove_particles(self.gravity.particles)
            for star in self.stars:
                p = Particle()
                p.mass = star.mass
                p.x = star.x
                p.y = star.y
                p.z = star.z
                p.vx = star.vx
                p.vy = star.vy
                p.vz = star.vz
                if star.remnant_type == "Black Hole":
                    p.radius = star.radius * ACCRETION_RADIUS_FACTOR
                else:
                    p.radius = star.radius * COLLISION_RADIUS_FACTOR
                self.gravity.particles.add_particle(p)
        else:
            temp_particles = Particles()
            for i, star in enumerate(self.stars):
                if i < len(self.gravity.particles):
                    p = Particle()
                    p.mass = star.mass
                    p.x = star.x
                    p.y = star.y
                    p.z = star.z
                    p.vx = star.vx
                    p.vy = star.vy
                    p.vz = star.vz
                    if star.remnant_type == "Black Hole":
                        p.radius = star.radius * ACCRETION_RADIUS_FACTOR
                    else:
                        p.radius = star.radius * COLLISION_RADIUS_FACTOR
                    temp_particles.add_particle(p)
            channel = temp_particles.new_channel_to(self.gravity.particles)
            channel.copy()

    def print_q_statistics(self):
        if self.q_distribution:
            q_values = np.array(self.q_distribution)
            print(f"\n=== STATISTICS FOR MASS RATIO q ===")
            print(f"Total collisions: {len(q_values)}")
            print(f"Mean q: {np.mean(q_values):.3f}")
            print(f"Median q: {np.median(q_values):.3f}")
            print(f"Std q: {np.std(q_values):.3f}")
            print(f"Min q: {np.min(q_values):.3f}")
            print(f"Max q: {np.max(q_values):.3f}")

            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            hist, _ = np.histogram(q_values, bins=bins)
            for i in range(len(bins) - 1):
                print(
                    f"q in [{bins[i]:.1f}, {bins[i + 1]:.1f}]: {hist[i]} collisions ({hist[i] / len(q_values) * 100:.1f}%)")

def get_metallicity_factor(metallicity):
    if metallicity < 0.001:
        return 1.3
    elif metallicity < 0.008:
        return 1.1
    elif metallicity < 0.02:
        return 1.0
    else:
        return 0.7

def get_stellar_lifetime_enhanced(mass_msun, metallicity=0.008):
    if mass_msun <= 0:
        return 0.0
    if mass_msun < 10:
        lifetime = 10000 * (mass_msun ** -2.5)
    elif mass_msun < 30:
        lifetime = 1000 * (mass_msun ** -1.5)
    elif mass_msun < 60:
        lifetime = 300 * (mass_msun ** -1.0)
    elif mass_msun < 100:
        lifetime = max(1.0, 50 * (100 / mass_msun))
    else:
        lifetime = max(0.5, 30 * (150 / mass_msun))
    z_factor = 1.0 + 0.3 * np.log10(max(0.001, metallicity) / 0.02)
    return max(0.1, lifetime * z_factor)

def determine_remnant_for_massive_star(initial_mass_msun, metallicity=0.008, stellar_type=""):
    z_factor = get_metallicity_factor(metallicity)
    adjusted_mass = initial_mass_msun * z_factor
    if initial_mass_msun < 8.0:
        remnant_mass = min(0.1 + initial_mass_msun * 0.08, CHANDRASEKHAR_LIMIT * 0.9)
        remnant_mass = max(0.5, remnant_mass)
        return ("White Dwarf", remnant_mass, "WD", f"Белый карлик. Исходная масса {initial_mass_msun:.1f} M☉, Z={metallicity:.4f}.")
    elif 8.0 <= initial_mass_msun < NS_UPPER_LIMIT:
        remnant_mass = 1.4 + np.random.normal(0, 0.15)
        remnant_mass = max(1.1, min(remnant_mass, OPPENHEIMER_VOLKOFF_LIMIT * 0.9))
        return ("Neutron Star", remnant_mass, "NS", f"Нейтронная звезда. Исходная масса {initial_mass_msun:.1f} M☉.")
    elif (metallicity < 0.001 and PISN_LOWER_THRESHOLD <= adjusted_mass <= PISN_UPPER_THRESHOLD):
        return ("NO_REMNANT", 0.0, "PISN", f"Парно-нестабильная сверхновая. Полное разрушение {initial_mass_msun:.0f} M☉ звезды (Z={metallicity:.4f}).")
    elif 100 <= adjusted_mass < PISN_LOWER_THRESHOLD:
        mass_loss_factor = np.random.uniform(0.4, 0.7)
        remnant_mass = initial_mass_msun * mass_loss_factor
        return ("Black Hole", remnant_mass, "BH", f"Чёрная дыра после пульсаций и сброса оболочки. Исходная масса {initial_mass_msun:.1f} M☉.")
    elif initial_mass_msun >= DIRECT_COLLAPSE_THRESHOLD:
        if "WN" in stellar_type or "WC" in stellar_type:
            mass_fraction = np.random.uniform(0.75, 0.95)
        else:
            mass_fraction = np.random.uniform(0.65, 0.85)
        remnant_mass = initial_mass_msun * mass_fraction * z_factor
        remnant_mass = max(NS_UPPER_LIMIT * 1.5, remnant_mass)
        return ("Black Hole", remnant_mass, "BH", f"Прямой коллапс в чёрную дыру. Исходная масса {initial_mass_msun:.1f} M☉, сохранено {mass_fraction*100:.0f}%.")
    else:
        mass_fraction = np.random.uniform(0.2, 0.35)
        remnant_mass = initial_mass_msun * mass_fraction * z_factor
        remnant_mass = max(OPPENHEIMER_VOLKOFF_LIMIT * 1.2, remnant_mass)
        return ("Black Hole", remnant_mass, "BH", f"Чёрная дыра (через сверхновую). Исходная масса {initial_mass_msun:.1f} M☉.")

def simulate_stellar_collapse(stars, current_time, gravity_code=None):
    collapsed_count = 0
    stars_to_remove = Particles()
    remnants_to_add = Particles()
    for star in stars:
        if hasattr(star, 'is_remnant') and star.is_remnant:
            continue
        if not hasattr(star, 'mass') or star.mass.value_in(units.MSun) <= 0:
            continue
        if not hasattr(star, 'initial_mass'):
            star.initial_mass = star.mass.value_in(units.MSun)
        if not hasattr(star, 'metallicity'):
            star.metallicity = getattr(star, 'metallicity', 0.008)
        lifetime_myr = get_stellar_lifetime_enhanced(star.initial_mass, star.metallicity)
        current_age_myr = star.age.value_in(units.Myr) if hasattr(star, 'age') else 0.0
        if lifetime_myr > 0 and current_age_myr >= lifetime_myr * MAX_AGE_FACTOR:
            stellar_type = str(getattr(star, 'spectral_label', ''))
            remnant_type, remnant_mass_val, spectral_label, description = determine_remnant_for_massive_star(
                star.initial_mass,
                star.metallicity,
                stellar_type
            )
            if remnant_type == "NO_REMNANT":
                print(f"[ПАРНО-НЕСТАБИЛЬНАЯ СВЕРХНОВАЯ] Время {current_time.value_in(units.Myr):.3f} Myr: Звезда {star.initial_mass:.1f} M☉ ({stellar_type}) полностью разрушена без остатка.")
                stars_to_remove.add_particle(star)
                collapsed_count += 1
                continue
            remnant = Particle()
            remnant.mass = remnant_mass_val | units.MSun
            remnant.position = star.position
            remnant.velocity = star.velocity
            remnant.is_remnant = True
            remnant.remnant_type = remnant_type
            remnant.spectral_label = spectral_label
            remnant.initial_mass = star.initial_mass
            remnant.metallicity = star.metallicity
            remnant.collision_count = getattr(star, 'collision_count', 0)
            remnant.accreted_mass = getattr(star, 'accreted_mass', 0.0 | units.MSun)
            if remnant_type == "Black Hole":
                remnant.radius = 2 * constants.G * remnant.mass / constants.c ** 2
                remnant.radius *= ACCRETION_RADIUS_FACTOR
            elif remnant_type == "Neutron Star":
                remnant.radius = 12 | units.km
            elif remnant_type == "White Dwarf":
                remnant.radius = 0.01 | units.RSun
            else:
                remnant.radius = (remnant_mass_val ** 0.8) | units.RSun
            if remnant_type == "White Dwarf":
                remnant.age = current_age_myr | units.Myr
            else:
                remnant.age = 0.0 | units.Myr
            mass_loss_percent = (1 - remnant_mass_val / star.initial_mass) * 100
            print(f"[КОЛЛАПС] Время {current_time.value_in(units.Myr):.3f} Myr: Звезда {star.initial_mass:.1f} M☉ ({stellar_type}, возраст {current_age_myr:.1f}/{lifetime_myr:.1f} Myr) -> {remnant_type} {remnant_mass_val:.2f} M☉ (потеря {mass_loss_percent:.1f}% массы)")
            stars_to_remove.add_particle(star)
            remnants_to_add.add_particle(remnant)
            collapsed_count += 1
    if collapsed_count > 0:
        stars.remove_particles(stars_to_remove)
        stars.add_particles(remnants_to_add)
        if gravity_code is not None:
            gravity_code.particles.remove_particles(stars_to_remove)
            for remnant in remnants_to_add:
                p = Particle()
                p.mass = remnant.mass
                p.position = remnant.position
                p.velocity = remnant.velocity
                if remnant.remnant_type == "Black Hole":
                    p.radius = remnant.radius * ACCRETION_RADIUS_FACTOR
                else:
                    p.radius = remnant.radius * COLLISION_RADIUS_FACTOR
                gravity_code.particles.add_particle(p)
        print(f"[КОЛЛАПС] Обновлено: {collapsed_count} звёзд достигли конца жизни и коллапсировали.")
    return collapsed_count

if __name__ == "__main__":
    print("=" * 70)
    print("ARCHES CLUSTER SIMULATION WITH REALISTIC COLLISION PHYSICS AND GAS FRICTION")
    print("=" * 70)

    print("\nВыберите режим работы:")
    print("1. Запустить однократную симуляцию")
    print("2. Запустить многоцикличную симуляцию")

    choice = int(2)

    if choice == "2":
        num_cycles = input("Введите количество циклов (по умолчанию 2): ").strip()
        num_cycles = int(12) if num_cycles.isdigit() else 2
        results = run_multiple_simulations(num_cycles)
    else:
        print(f"\n{'=' * 70}")
        print(f"ЗАПУСК ОДНОКРАТНОЙ СИМУЛЯЦИИ")
        print(f"{'=' * 70}")
        result_dir = run_simulation_cycle(1, 1, None)
        results = [result_dir] if result_dir else []

    print(f"\nВсе симуляции завершены!")
    if results:
        print(f"Результаты сохранены в следующих папках:")
        for i, dir_path in enumerate(results, 1):
            print(f"  {i}. {dir_path}")
    else:
        print("Нет сохраненных результатов.")
