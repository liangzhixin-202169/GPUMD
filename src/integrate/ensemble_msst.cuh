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

#pragma once
#include "ensemble.cuh"

class Ensemble_MSST : public Ensemble
{
public:
  Ensemble_MSST(int, int, int, double, double, double, double, double, double, double, double);
  virtual ~Ensemble_MSST(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo);

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo);

  virtual void msst_initial_param(Box& box, GPU_Vector<double>& thermo);

  virtual void msst_setup(
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    const double volume);

  void remap(
    const int N,
    Box& box, 
    GPU_Vector<double>& position_per_atom, 
    GPU_Vector<double>& velocity_per_atom);

  void velocity_upgrade_msst(
    const bool is_step1,
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom);

  int shock_direction;
  double shockvel; 
  double qmass;
  double mu;
  double p0;
  double v0;
  double e0;
  double tscale;
  double beta;
  double lagrangian_position;
  std::vector<double> cpu_omega;
  GPU_Vector<double> omega;
  std::vector<double> cpu_total_mass;
  std::vector<double> cpu_etotal;
  double dilation[3];
  GPU_Vector<double> velocity_sum;
};
