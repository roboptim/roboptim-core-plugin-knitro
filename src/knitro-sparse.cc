// Copyright (C) 2016 by Benjamin Chrétien, CNRS-AIST JRL.
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

#include <roboptim/core/function.hh>

#include "roboptim/core/plugin/knitro/solver.hh"

namespace roboptim
{
  template <>
  inline void evalGradObj (double* const objGrad,
                           typename GenericDifferentiableFunction<
                             EigenMatrixSparse>::const_argument_ref x,
                           const KNITROSolver<EigenMatrixSparse>* solver)
  {
    typedef KNITROSolver<EigenMatrixSparse> solver_t;
    typedef typename solver_t::problem_t problem_t;
    typedef typename solver_t::vector_t vector_t;
    typedef typename solver_t::differentiableFunction_t differentiableFunction_t;

    const problem_t& pb = solver->problem ();

    const differentiableFunction_t* df = 0;
    if (pb.function ().asType<differentiableFunction_t> ())
    {
      df = pb.function ().castInto<differentiableFunction_t> ();
      df->gradient (solver->objGrad (), x);
    }

    // Objective gradient is dense in KNITRO
    Eigen::Map<vector_t> objGrad_ (objGrad, solver->inputSize ());
    objGrad_ = solver->objGrad ();
  }

  template <>
  inline void evalJacobian (
    double* const jac,
    typename GenericDifferentiableFunction<EigenMatrixSparse>::const_argument_ref /* x */,
    const KNITROSolver<EigenMatrixSparse>* /* solver */)
  {
    // df->jacobian (jac.block (idx, 0, df->outputSize (), df->inputSize ()), x);
    jac[0] = 42.;
  }

  template <>
  int KNITROSolver<EigenMatrixSparse>::getSparsityPattern (
    Eigen::VectorXi& jacIndexVars, Eigen::VectorXi& jacIndexCons, int n, int m,
    const argument_t& x) const
  {
    const problem_t& pb = this->problem ();
    jacobian_t jac = pb.jacobian (x);
    size_type idx = 0;
    size_type nnz = jac.nonZeros ();
    jacIndexVars.resize (nnz);
    jacIndexCons.resize (nnz);

    if (StorageOrder == Eigen::ColMajor)
    {
      for (int k = 0; k < n; ++k)
        for (typename jacobian_t::InnerIterator it (jac, k);
            it; ++it)
        {
          jacIndexCons[idx] = it.row (); // row
          jacIndexVars[idx] = it.col (); // col
          idx++;
        }
    }
    else // row-major
    {
      for (int k = 0; k < m; ++k)
        for (typename jacobian_t::InnerIterator it (jac, k);
            it; ++it)
        {
          jacIndexCons[idx] = it.row (); // row
          jacIndexVars[idx] = it.col (); // col
          idx++;
        }
    }

    return static_cast<int> (nnz);
  }
} // end of namespace roboptim

extern "C" {
using namespace roboptim;
typedef KNITROSolver<EigenMatrixSparse> knitroSolver_t;
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
