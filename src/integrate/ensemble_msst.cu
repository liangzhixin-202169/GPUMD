/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The NVE ensemble integrator.
------------------------------------------------------------------------------*/

#include "ensemble_msst.cuh"
#include "utilities/common.cuh"
#define DIM 3

Ensemble_MSST::Ensemble_MSST(int t, int fg,int sd,double shock_velocity, double Qmass,double Mu,double P0,double V0,double E0,double Tscale,double Beta)
{
  type = t;
  fixed_group = fg;
  shock_direction = sd;
  shockvel = shock_velocity;
  qmass = Qmass;
  mu = Mu;
  p0 = P0;
  v0 = V0;
  e0 = E0;
  tscale = Tscale;
  beta = Beta;

  cpu_omega.resize(3);
  omega.resize(3);
  cpu_total_mass.resize(1);
  cpu_etotal.resize(1);
  velocity_sum.resize(1);

  cpu_omega[0] = cpu_omega[1] = cpu_omega[2] = 0.0;
  omega.copy_from_host(cpu_omega.data());
  dilation[0] = dilation[1] = dilation[2] = 1.0;
}

Ensemble_MSST::~Ensemble_MSST(void)
{
  // nothing now
}

void Ensemble_MSST::msst_initial_param(Box& box, GPU_Vector<double>& thermo)
{
 v0 = box.get_volume();
 std::vector<double>thermo_cpu;
 thermo_cpu.resize(thermo.size());
 thermo.copy_to_host(thermo_cpu.data());
 e0 = thermo_cpu[1] + thermo_cpu[2];
 p0 = thermo_cpu[shock_direction + 4];
 printf("    MSST V0: %g A^3, E0: %g eV, P0: %g GPa\n", v0, e0, p0* PRESSURE_UNIT_CONVERSION);
}

static __global__ void gpu_velocity_upgrate_msst(
  const bool is_step1,
  const int number_of_particles,
  const double g_time_step,
  const double* g_mass,
  double* g_x,
  double* g_y,
  double* g_z,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    const double time_step = g_time_step;
      g_x[i] += g_vx[i] * time_step;
      g_y[i] += g_vy[i] * time_step;
      g_z[i] += g_vz[i] * time_step;
  }
}

void Ensemble_MSST::velocity_upgrade_msst(
  const bool is_step1,
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();

  gpu_velocity_upgrate_msst<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    is_step1, number_of_atoms, time_step, mass.data(), position_per_atom.data(),
    position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2,
    velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms, force_per_atom.data(),
    force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL
}

// Find some thermodynamic properties:
// g_thermo[0-7] = T, U, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
static __global__ void gpu_find_thermo_instant_temperature(
  const int N,
  const int N_temperature,
  const double T,
  const double volume,
  const double* g_mass,
  const double* g_potential,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const double* g_sxx,
  const double* g_syy,
  const double* g_szz,
  const double* g_sxy,
  const double* g_sxz,
  const double* g_syz,
  double* g_thermo)
{
  //<<<8, MAX_THREAD>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int patch, n;
  int number_of_patches = (N - 1) / 1024 + 1;
  double mass, vx, vy, vz;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  switch (bid) {
    // temperature
    case 0:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[n];
          vx = g_vx[n];
          vy = g_vy[n];
          vz = g_vz[n];
          s_data[tid] += (vx * vx + vy * vy + vz * vz) * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[0] = s_data[0] / (DIM * N_temperature * K_B);
      }
      break;
      // potential energy
    case 1:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          s_data[tid] += g_potential[n];
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0)
        g_thermo[1] = s_data[0];
      break;
      // sxx
    case 2:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[n];
          vx = g_vx[n];
          s_data[tid] += g_sxx[n] + vx * vx * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[2] = s_data[0] / volume;
      }
      break;
      // syy
    case 3:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[n];
          vy = g_vy[n];
          s_data[tid] += g_syy[n] + vy * vy * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[3] = s_data[0] / volume;
      }
      break;
      // szz
    case 4:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[n];
          vz = g_vz[n];
          s_data[tid] += g_szz[n] + vz * vz * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[4] = s_data[0] / volume;
      }
      break;
      // sxy
    case 5:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          s_data[tid] += g_sxy[n];
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[5] = s_data[0] / volume;
      }
      break;
      // sxz
    case 6:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          s_data[tid] += g_sxz[n];
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[6] = s_data[0] / volume;
      }
      break;
      // syz
    case 7:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          s_data[tid] += g_syz[n];
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[7] = s_data[0] / volume;
      }
      break;
  }
}

static __global__ void gpu_sum_1d(const int N, const double* vector,double* scalar) {
   //<<<1,1024>>>
  int tid = threadIdx.x;
   int patch, n;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (patch = 0; patch < number_of_patches; patch++) {
    n = tid + patch * 1024;
    if (n < N) {
      s_data[tid] += vector[n];
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    scalar[0] = s_data[0];
  }
}

static __global__ void gpu_sum_3d(
  const int N, const double* g_vector1, const double* g_vector2, const double* g_vector3, double* g_scalar)
{
  //<<<1, 1024>>>
  int tid = threadIdx.x;
  int patch, n;
  int number_of_patches = (N - 1) / 1024 + 1;
  double vector1, vector2, vector3;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (patch = 0; patch < number_of_patches; patch++) {
    n = tid + patch * 1024;
    if (n < N) {
      vector1 = g_vector1[n];
      vector2 = g_vector2[n];
      vector3 = g_vector3[n];
      s_data[tid] += vector1 * vector1 + vector2 * vector2 + vector3 * vector3;
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_scalar[0] = s_data[0];
  }
}

void Ensemble_MSST::msst_setup(
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  const double volume)
{
  lagrangian_position = 0.0;

  const int number_of_atoms = mass.size();
  int num_atoms_for_temperature = number_of_atoms;

  // compute temperature
  temperature = 2022.1025;
  //double temperature = 2022.1025;
  std::vector<double> cpu_thermo(8);
  GPU_Vector<double> thermo;
  thermo.resize(8);
  gpu_find_thermo_instant_temperature<<<8, 1024>>>(
    number_of_atoms, num_atoms_for_temperature, temperature, volume, mass.data(),
    potential_per_atom.data(), velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms, virial_per_atom.data(),
    virial_per_atom.data() + number_of_atoms, virial_per_atom.data() + number_of_atoms * 2,
    virial_per_atom.data() + number_of_atoms * 3, virial_per_atom.data() + number_of_atoms * 4,
    virial_per_atom.data() + number_of_atoms * 5, thermo.data());
  CUDA_CHECK_KERNEL
  //temperature potential sxx syy szz sxy sxz syz
  thermo.copy_to_host(cpu_thermo.data());

  // compute total mass
  GPU_Vector<double> total_mass;
  total_mass.resize(1);
  gpu_sum_1d<<<1, 1024>>>(number_of_atoms, mass.data(), total_mass.data());
  CUDA_CHECK_KERNEL
  total_mass.copy_to_host(cpu_total_mass.data());

  double sqrt_initial_temperature_scaling = sqrt(1.0 - tscale);
  double fac1 = tscale * cpu_total_mass[0] / qmass * cpu_thermo[0];
  cpu_omega[shock_direction] = -1 * sqrt(fac1);
  double fac2 = cpu_omega[shock_direction] / v0;

  scale_velocity_global(sqrt_initial_temperature_scaling, velocity_per_atom);
  /*gpu_scale_velocity<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, fac2, velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL*/ 
}

static __global__ void gpu_find_energy(
  const int N,
  const double* g_mass, 
  const double* g_vx, 
  const double* g_vy, 
  const double* g_vz, 
  const double* g_potential,
  double* g_energy)
{
  //<<<1,1024>>>
  int tid = threadIdx.x;
  int patch, n;
  int number_of_patches = (N - 1) / 1024 + 1;
  double mass, vx, vy, vz;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (patch = 0; patch < number_of_patches; patch++) {
    n = tid + patch * 1024;
    if (n < N) {
      mass = g_mass[n];
      vx = g_vx[n];
      vy = g_vy[n];
      vz = g_vz[n];
      s_data[tid] += 0.5 * (vx * vx + vy * vy + vz * vz) * mass + g_potential[n];
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_energy[0] = s_data[0];
  }
}

static __global__ void gpu_fun_1(
  const int N,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz,
  const double* g_mass,
  const double mu,
  const int shock_direction,
  const double* g_omega,
  const double* velocity_sum,
  const double volume,
  double* g_velocity,
  double dthalf)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    double mass = g_mass[n];
    double mass_inv = 1.0 / mass;
    double C[3] = {g_fx[n] * mass_inv, g_fy[n] * mass_inv, g_fz[n] * mass_inv};
    const double tmp =
      g_omega[shock_direction] * g_omega[shock_direction] * mu / (velocity_sum[0] * mass * volume);
    double D[3] = {tmp, tmp, tmp};
    D[shock_direction] -= 2.0 * g_omega[shock_direction] / volume;
    for (int i = 0; i < 3; i++) {
      if (fabs(dthalf * D[i]) > 1.0e-06) {
        const double expd = exp(D[i] * dthalf);
        g_velocity[n + i * N] = expd * (C[i] + D[i] * g_velocity[n + i * N] - C[i] / expd) / D[i];
      } else {
        g_velocity[n + i * N] =
          g_velocity[n + i * N] + (C[i] + D[i] * g_velocity[n + i * N]) * dthalf +
          0.5 * (D[i] * D[i] * g_velocity[n + i * N] + C[i] * D[i]) * dthalf * dthalf;
      }
    }
  }
}

static __global__ void gpu_fun_2(
  const int N,
  const int shock_direction,
  const double dilation,
  double* g_position,
  double* g_velocity,
  const double ctr)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    //g_position[n + shock_direction * N] =
    //  (g_position[n + shock_direction * N] - ctr) * dilation + ctr;
    g_position[n + shock_direction * N] = g_position[n + shock_direction * N] * dilation;
    g_velocity[n + N * shock_direction] = g_velocity[n + N * shock_direction] * dilation;
  }
}

void Ensemble_MSST::remap(
  const int N,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom) {
  box.cpu_h[shock_direction] *= dilation[shock_direction];
  box.cpu_h[shock_direction + 3] = 0.5 * box.cpu_h[shock_direction];
  gpu_fun_2<<<(N - 1) / 128 + 1, 128>>>(
    N, shock_direction, dilation[shock_direction], position_per_atom.data(),
    velocity_per_atom.data(), box.cpu_h[shock_direction + 3]);
  CUDA_CHECK_KERNEL
}

void Ensemble_MSST::compute1(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  int N = mass.size();
  std::vector<double> cpu_thermo(8);
  find_thermo(
    false, box.get_volume(), group, mass, potential_per_atom, velocity_per_atom, virial_per_atom,
    thermo);
  thermo.copy_to_host(cpu_thermo.data(), 8);
  double volume = box.get_volume();

  // compute total energy
  GPU_Vector<double> etotal;
  etotal.resize(1);
  gpu_find_energy<<<1, 1024>>>(
    N, mass.data(), velocity_per_atom.data(), velocity_per_atom.data() + N,
    velocity_per_atom.data() + 2 * N, potential_per_atom.data(), etotal.data());
  CUDA_CHECK_KERNEL
  etotal.copy_to_host(cpu_etotal.data());

  // compute scalar
  double scalar =
    qmass * cpu_omega[shock_direction] * cpu_omega[shock_direction] / (2.0 * cpu_total_mass[0]) -
    0.5 * cpu_total_mass[0] * shockvel * shockvel * (1.0 - volume / v0) * (1.0 - volume / v0) -
    p0 * (v0 - volume);

  // conserved quantity
  double e_scale = cpu_etotal[0] + scalar;

  // propagate the time derivative of the volume 1/2 step at fixed vol, r, rdot
  double p_msst = shockvel * shockvel * cpu_total_mass[0] * (v0 - volume) / (v0 * v0);
  double A = cpu_total_mass[0] * (cpu_thermo[shock_direction + 2] - p0 - p_msst) / qmass;
  double B = cpu_total_mass[0] * mu / (qmass * volume);

  // prevent blow-up of the volume
  if (volume > v0 && A > 0.0) {
    A = -A;
  }
    
  // use Taylor expansion to avoid singularity at B = 0
  double dthalf = 0.5 * time_step;
  if (B * dthalf > 1.0e-06) {
    cpu_omega[shock_direction] =
      (cpu_omega[shock_direction] + A * (exp(B * dthalf) - 1.0) / B) * exp(-B * dthalf);
  } else {
    cpu_omega[shock_direction] = cpu_omega[shock_direction] + (A - B * cpu_omega[shock_direction]) * dthalf +
                             0.5 * (B * B * cpu_omega[shock_direction] - A * B) * dthalf * dthalf;
  }
  omega.copy_from_host(cpu_omega.data());

  // propagate velocity sum 1/2 step by temporarily propagating the velocities
  gpu_sum_3d<<<1, 1024>>>(
    N, velocity_per_atom.data(), velocity_per_atom.data() + N, velocity_per_atom.data() + 2 * N,
    velocity_sum.data());
  CUDA_CHECK_KERNEL

  std::vector<double> cpu_old_velocity(velocity_per_atom.size());
  velocity_per_atom.copy_to_host(cpu_old_velocity.data());
  gpu_fun_1<<<(N - 1) / 128 + 1, 128>>>(
    N, force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + 2 * N, mass.data(),
    mu, shock_direction, omega.data(), velocity_sum.data(), volume, velocity_per_atom.data(),
    dthalf);
  CUDA_CHECK_KERNEL
  gpu_sum_3d<<<1, 1024>>>(
    N, velocity_per_atom.data(), velocity_per_atom.data() + N, velocity_per_atom.data() + 2 * N,
    velocity_sum.data());
  CUDA_CHECK_KERNEL

  // reset the velocities
  velocity_per_atom.copy_from_host(cpu_old_velocity.data());

  // propagate velocities 1/2 step using the new velocity sum
  gpu_fun_1<<<(N - 1) / 128 + 1, 128>>>(
    N, force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + 2 * N, mass.data(),
    mu, shock_direction, omega.data(), velocity_sum.data(), volume, velocity_per_atom.data(),
    dthalf);
  CUDA_CHECK_KERNEL

  // propagate the volume 1/2 step
  double vol1 = volume + cpu_omega[shock_direction] * dthalf;

  // rescale positions and change box size
  dilation[shock_direction] = vol1 / volume;
  remap(N, box, position_per_atom, velocity_per_atom);


  velocity_upgrade_msst(
    true, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  // propagate the volume 1/2 step
  double vol2 = vol1 + cpu_omega[shock_direction] * dthalf;

  // rescale positions and change box size
  dilation[shock_direction] = vol2 / vol1;
  remap(N, box, position_per_atom, velocity_per_atom);
}

void Ensemble_MSST::compute2(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  int N = mass.size();
  std::vector<double> cpu_thermo(8);
  thermo.copy_to_host(cpu_thermo.data(), 8);
  double volume = box.get_volume();

  // compute total energy
  GPU_Vector<double> etotal;
  etotal.resize(1);
  gpu_find_energy<<<1, 1024>>>(
    N, mass.data(), velocity_per_atom.data(), velocity_per_atom.data() + N,
    velocity_per_atom.data() + 2 * N, potential_per_atom.data(), etotal.data());
  CUDA_CHECK_KERNEL
  etotal.copy_to_host(cpu_etotal.data());

  // compute scalar
  double scalar =
    qmass * omega[shock_direction] * omega[shock_direction] / (2.0 * cpu_total_mass[0]) -
    0.5 * cpu_total_mass[0] * shockvel * shockvel * (1.0 - volume / v0) * (1.0 - volume / v0) -
    p0 * (v0 - volume);

  // conserved quantity
  double e_scale = cpu_etotal[0] + scalar;

  // propagate particle velocities 1/2 step
  double dthalf = 0.5 * time_step;
  gpu_fun_1<<<(N - 1) / 128 + 1, 128>>>(
    N, force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + 2 * N, mass.data(),
    mu, shock_direction, omega.data(), velocity_sum.data(), volume, velocity_per_atom.data(),
    dthalf);
  CUDA_CHECK_KERNEL

  // compute new pressure and volume
  find_thermo(
    false, box.get_volume(), group, mass, potential_per_atom, velocity_per_atom, virial_per_atom,
    thermo);
  thermo.copy_to_host(cpu_thermo.data(), 8);
  //GPU_Vector<double> velocity_sum;
  //velocity_sum.resize(1);
  gpu_sum_3d<<<1, 1024>>>(
    N, velocity_per_atom.data(), velocity_per_atom.data() + N, velocity_per_atom.data() + 2 * N,
    velocity_sum.data());
  CUDA_CHECK_KERNEL
  volume = box.get_volume();

  // propagate the time derivative of the volume 1/2 step at fixed V, r, rdot
  double p_msst = shockvel * shockvel * cpu_total_mass[0] * (v0 - volume) / (v0 * v0);
  double A = cpu_total_mass[0] * (cpu_thermo[shock_direction + 2] - p0 - p_msst)/qmass;
  double B = cpu_total_mass[0] * mu / (qmass * volume);

  // prevent blow-up of the volume
  if (volume > v0 && A > 0.0) {
    A = -A;
  }
  
  // use Taylor expansion to avoid singularity at B = 0
  //double dthalf = 0.5 * time_step;
  if (B * dthalf > 1.0e-06) {
    cpu_omega[shock_direction] =
      (cpu_omega[shock_direction] + A * (exp(B * dthalf) - 1.0) / B) * exp(-B * dthalf);
  } else {
    cpu_omega[shock_direction] =
      cpu_omega[shock_direction] + (A - B * cpu_omega[shock_direction]) * dthalf +
      0.5 * (B * B * cpu_omega[shock_direction] - A * B) * dthalf * dthalf;
  }
  omega.copy_from_host(cpu_omega.data());

  // calculate Lagrangian position of computational cell
  lagrangian_position -= shockvel * volume / v0 * time_step;

  //velocity_verlet(
  //  false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  find_thermo(
    false, box.get_volume(), group, mass, potential_per_atom, velocity_per_atom, virial_per_atom,
    thermo);
}
