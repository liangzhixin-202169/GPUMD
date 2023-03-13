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
#include <curand_kernel.h>
#include <vector>

class Ensemble_PIMD : public Ensemble
{
public:
  Ensemble_PIMD(
    int number_of_atoms_input,
    int number_of_beads_input,
    double temperature_input,
    double temperature_coupling_input,
    double temperature_coupling_beads_input);

  virtual ~Ensemble_PIMD(void);

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

  struct Beads {
    const double* velocity[128]; // at most 128 beads
    const double* position[128]; // at most 128 beads
    const double* force[128];    // at most 128 beads
  };

protected:
  int number_of_atoms = 0;
  int number_of_beads = 0;
  double temperature_coupling_beads;
  double c1, c2;
  GPU_Vector<curandState> curand_states;
  std::vector<GPU_Vector<double>> position;
  std::vector<GPU_Vector<double>> velocity;
  std::vector<GPU_Vector<double>> force;
  GPU_Vector<double> transformation_matrix;
  Beads beads;

  void
  integrate_nvt_lan_half(const GPU_Vector<double>& mass, GPU_Vector<double>& velocity_per_atom);
};
