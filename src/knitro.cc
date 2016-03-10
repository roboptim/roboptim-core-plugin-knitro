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
#include "roboptim/core/plugin/knitro/config.hh"

#include <roboptim/core/function.hh>
#include <roboptim/core/differentiable-function.hh>

namespace roboptim
{
  template <>
  inline void evalGradObj (
    double* const objGrad,
    typename GenericDifferentiableFunction<EigenMatrixDense>::const_argument_ref x,
    const KNITROSolver<EigenMatrixDense>* solver)
  {
    typedef KNITROSolver<EigenMatrixDense> solver_t;
    typedef typename solver_t::problem_t problem_t;
    typedef typename solver_t::differentiableFunction_t differentiableFunction_t;
    typedef typename solver_t::gradient_t gradient_t;

    const problem_t& pb = solver->problem ();
    Eigen::Map<gradient_t> objGrad_ (objGrad, solver->inputSize ());

    const differentiableFunction_t* df = 0;
    if (pb.function ().asType<differentiableFunction_t> ())
    {
      df = pb.function ().castInto<differentiableFunction_t> ();
      df->gradient (objGrad_, x);
    }
  }

  template <>
  inline void evalJacobian (
    double* const jac,
    typename GenericDifferentiableFunction<EigenMatrixDense>::const_argument_ref x,
    const KNITROSolver<EigenMatrixDense>* solver)
  {
    typedef KNITROSolver<EigenMatrixDense> solver_t;
    typedef typename solver_t::problem_t problem_t;
    typedef typename solver_t::jacobian_t jacobian_t;
    typedef typename solver_t::differentiableFunction_t differentiableFunction_t;
    typedef typename problem_t::constraints_t::const_iterator iterator_t;

    const problem_t& pb = solver->problem ();
    int n = static_cast<int> (solver->inputSize ());
    int m = static_cast<int> (solver->outputSize ());

    ptrdiff_t idx = 0;
    const differentiableFunction_t* df = 0;
    Eigen::Map<jacobian_t> jacobianBuf (jac, m, n);
    for (iterator_t it = pb.constraints ().begin ();
         it != pb.constraints ().end (); ++it)
    {
      if ((*it)->asType<differentiableFunction_t> ())
      {
        df = (*it)->castInto<differentiableFunction_t> ();
        df->jacobian (jacobianBuf.block (idx, 0, df->outputSize (),
              df->inputSize ()), x);
        idx += (*it)->outputSize ();
      }
    }
  }

  template <>
  int KNITROSolver<EigenMatrixDense>::getSparsityPattern (
    Eigen::VectorXi& jacIndexVars, Eigen::VectorXi& jacIndexCons, int n, int m,
    const argument_t&, jacobian_t&) const
  {
    const int nnz = n * m;
    jacIndexVars.resize (nnz);
    jacIndexCons.resize (nnz);

    int k = 0;

    // If dense RobOptim jacobian matrices are column-major:
    if (GenericFunctionTraits<EigenMatrixDense>::StorageOrder ==
        Eigen::ColMajor)
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

ROBOPTIM_CORE_PLUGIN_KNITRO_DLLEXPORT unsigned getSizeOfProblem ();
ROBOPTIM_CORE_PLUGIN_KNITRO_DLLEXPORT const char* getTypeIdOfConstraintsList ();
ROBOPTIM_CORE_PLUGIN_KNITRO_DLLEXPORT solver_t* create (const solver_t::problem_t&);
ROBOPTIM_CORE_PLUGIN_KNITRO_DLLEXPORT void destroy (solver_t*);

unsigned getSizeOfProblem ()
{
  return sizeof (solver_t::problem_t);
}

const char* getTypeIdOfConstraintsList ()
{
  return typeid (solver_t::problem_t::constraintsList_t).name ();
}

solver_t* create (const solver_t::problem_t& pb)
{
  return new knitroSolver_t (pb);
}

void destroy (solver_t* p)
{
  delete p;
}
}
