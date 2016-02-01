// Copyright (C) 2014 by Thomas Moulard, AIST, CNRS.
//
// This file is part of the roboptim.
//
// roboptim is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// roboptim is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with roboptim.  If not, see <http://www.gnu.org/licenses/>.

#include "roboptim/core/plugin/knitro/solver.hh"

#include <roboptim/core/function.hh>

namespace roboptim
{
  template <>
  int KNITROSolver<EigenMatrixDense>::getSparsityPattern (
    Eigen::VectorXi& jacIndexVars, Eigen::VectorXi& jacIndexCons, int n,
    int m) const
  {
    const int nnz = n * m;
    jacIndexVars.resize (nnz);
    jacIndexCons.resize (nnz);

    int k = 0;

    // If dense RobOptim jacobian matrices are column-major:
    if (GenericFunctionTraits<traits_t>::StorageOrder == Eigen::ColMajor)
    {
      for (int j = 0; j < n; j++) // col
        for (int i = 0; i < m; i++) // row
        {
          jacIndexCons[k] = i; // row
          jacIndexVars[k] = j; // col
          k++;
        }
    }
    else // row-major
    {
      for (int i = 0; i < m; i++) // row
        for (int j = 0; j < n; j++) // col
        {
          jacIndexCons[k] = i; // row
          jacIndexVars[k] = j; // col
          k++;
        }
    }

    return nnz;
  }
} // end of namespace roboptim

extern "C" {
using namespace roboptim;
typedef KNITROSolver<EigenMatrixDense> knitroSolver_t;
typedef knitroSolver_t::solver_t solver_t;

ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ();
ROBOPTIM_DLLEXPORT solver_t* create (const solver_t::problem_t&);
ROBOPTIM_DLLEXPORT void destroy (solver_t*);

ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ()
{
  return sizeof (solver_t::problem_t);
}

ROBOPTIM_DLLEXPORT const char* getTypeIdOfConstraintsList ()
{
  return typeid (solver_t::problem_t::constraintsList_t).name ();
}

ROBOPTIM_DLLEXPORT solver_t* create (const solver_t::problem_t& pb)
{
  return new knitroSolver_t (pb);
}

ROBOPTIM_DLLEXPORT void destroy (solver_t* p)
{
  delete p;
}
}
