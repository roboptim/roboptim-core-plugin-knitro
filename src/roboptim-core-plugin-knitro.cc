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

#include <roboptim/core/function.hh>
#include <roboptim/core/indent.hh>
#include <roboptim/core/result.hh>
#include <roboptim/core/result-with-warnings.hh>
#include <roboptim/core/util.hh>

#include <knitro.h>

#include "roboptim-core-plugin-knitro.hh"

namespace roboptim
{

  template <typename U>
  class EvaluateConstraint : public boost::static_visitor<>
  {
  public:
    EvaluateConstraint (double* const c, std::ptrdiff_t& offset, const U& x)
      : boost::static_visitor<> (),
	c (c),
	offset (offset),
	x (x)
    {}

    template<typename T>
    void
    operator () (const boost::shared_ptr<T>& constraint) const
    {
      Eigen::Map<Eigen::Matrix<typename T::value_type,
			       Eigen::Dynamic, 1> >
	constraintValue (c + offset, constraint->outputSize ());
    constraintValue = (*constraint) (x);

    offset += constraint->outputSize ();
    }

private:
double* const c;
std::ptrdiff_t& offset;
const U& x;
  };

  static int callback (const int evalRequestCode,
		       const int n,
		       const int ROBOPTIM_DEBUG_ONLY (m),
		       const int /* nnzJ */,
		       const int /* nnzH */,
		       const double* const x,
		       const double* const /* lambda */,
		       double* const obj,
		       double* const c,
		       double* const /* objGrad */,
		       double* const /* jac */,
		       double* const /* hessian */,
		       double* const /* hessVector */,
		       void* userParams)
  {
    typedef KNITROSolver::vector_t vector_t;
    typedef KNITROSolver::value_type value_type;

    if (!userParams)
      {
	std::cerr << "bad user params\n";
	return -1;
      }
    KNITROSolver* solver = static_cast<KNITROSolver*> (userParams);

    if (evalRequestCode == KTR_RC_EVALFC)
      {
	Eigen::Map<const Eigen::Matrix<value_type, Eigen::Dynamic, 1> > x_ (x, n);
	Eigen::Map<Eigen::Matrix<value_type, Eigen::Dynamic, 1> > obj_ (obj, 1);

	// objective
	obj_ = solver->problem ().function () (x_);

	// constraints
	typedef KNITROSolver::problem_t::constraints_t::const_iterator iterator_t;
	std::ptrdiff_t offset = 0;

	for (iterator_t it = solver->problem ().constraints ().begin ();
	     it != solver->problem ().constraints ().end (); ++it)
	  {
	    assert (offset < m);
	    boost::apply_visitor
	      (EvaluateConstraint<Eigen::Map<const Eigen::Matrix<value_type,
								 Eigen::Dynamic, 1> > >
		(c, offset, x_), *it);
	  }

	return(0);
      }
    else
      {
	std::cerr << "bad evalRequestCode\n";
	return -1;
      }
    return 0;
  }

  class ConstraintsSizeVisitor : public boost::static_visitor<int>
  {
  public:
    template<typename T>
    int operator () (const boost::shared_ptr<T>& constraint) const
    {
      return static_cast<int> (constraint->outputSize ());
    }
  };

  KNITROSolver::KNITROSolver (const problem_t& problem) throw ()
    : parent_t (problem),
      solverState_ (problem),
      knitro_ ()
  {}

  KNITROSolver::~KNITROSolver () throw ()
  {
    KTR_free (&knitro_);
  }

  int
  KNITROSolver::computeConstraintsSize () throw ()
  {
    typedef problem_t::constraints_t::const_iterator iterator_t;
    int result = 0;

    for (iterator_t it = this->problem ().constraints ().begin ();
	 it != this->problem ().constraints ().end (); ++it)
	result += boost::apply_visitor (ConstraintsSizeVisitor (), *it);

    return result;
  }

  void
  KNITROSolver::solve () throw ()
  {
    int nStatus = 0;


    if (!knitro_)
      knitro_ = KTR_new ();
    if (!knitro_)
      {
	// failure
	return;
      }

    if (KTR_set_int_param_by_name (knitro_, "gradopt", KTR_GRADOPT_FORWARD))
      {
	// failure
	return;
      }
    if (KTR_set_int_param_by_name (knitro_, "hessopt", KTR_HESSOPT_BFGS))
      {
	// failure
	return;
      }
    if (KTR_set_int_param_by_name (knitro_, "outlev", 1))
      {
	// failure
	return;
      }

    if (KTR_set_func_callback (knitro_, &callback))
      {
	// failure
	return;
      }

    // Problem definition

    // number of variables
    int n = static_cast<int> (this->problem ().function ().inputSize ());

    // number of constraints
    int m = computeConstraintsSize ();
    int nnzJ = n * m;
    int nnzH = 0;

    int objType = KTR_OBJTYPE_GENERAL;
    int objGoal = KTR_OBJGOAL_MINIMIZE;


    // bounds and constraints type
    std::vector<double> xLoBnds (n);
    std::vector<double> xUpBnds (n);
    for (int i = 0; i < n; i++)
      xLoBnds[i] = 0.0, xUpBnds[i] = KTR_INFBOUND;

    std::vector<int> cType (m);
    std::vector<double> cLoBnds (m);
    std::vector<double> cUpBnds (m);
    for (int j = 0; j < m; j++)
      {
	cType[j] = KTR_CONTYPE_GENERAL;
	cLoBnds[j] = 0.0;
	cUpBnds[j] = (j == 0 ? 0.0 : KTR_INFBOUND);
      }

    // initial point
    Eigen::Matrix<value_type, Eigen::Dynamic, 1> xInitial;
    if (this->problem ().startingPoint ())
      xInitial = *this->problem ().startingPoint ();
    else
      xInitial.resize
	(this->problem ().function ().inputSize (), value_type ());

    // sparsity pattern (dense here)
    std::vector<int> jacIndexVars (nnzJ);
    std::vector<int> jacIndexCons (nnzJ);

    int k = 0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++)
	{
	  jacIndexCons[k] = j;
	  jacIndexVars[k] = i;
	  k++;
	}

    nStatus = KTR_init_problem (knitro_, n, objGoal, objType,
				&xLoBnds[0], &xUpBnds[0],
				m, &cType[0], &cLoBnds[0], &cUpBnds[0],
				nnzJ, &jacIndexVars[0], &jacIndexCons[0],
				nnzH, 0, 0, &xInitial[0], 0);

    vector_t x (this->problem ().function ().inputSize ());
    std::vector<double> lambda (m + n);
    double obj = 0.;
    nStatus = KTR_solve (knitro_, &x[0], &lambda[0], 0, &obj, 0, 0, 0, 0, 0, 0);

    if (nStatus != 0)
      {
	// failure
      }

    // ok

  }

  std::ostream&
  KNITROSolver::print (std::ostream& o) const throw ()
  {
    return o;
  }

  void
  KNITROSolver::setIterationCallback (callback_t callback)
    throw (std::runtime_error)
  {
    callback_ = callback;
  }


} // end of namespace roboptim.

extern "C"
{
  using namespace roboptim;
  typedef KNITROSolver::parent_t solver_t;

  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ();
  ROBOPTIM_DLLEXPORT solver_t* create (const KNITROSolver::problem_t&);
  ROBOPTIM_DLLEXPORT void destroy (solver_t*);


  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ()
  {
    return sizeof (KNITROSolver::problem_t);
  }

  ROBOPTIM_DLLEXPORT const char* getTypeIdOfConstraintsList ()
  {
    return typeid (KNITROSolver::problem_t::constraintsList_t).name ();
  }

  ROBOPTIM_DLLEXPORT solver_t* create (const KNITROSolver::problem_t& pb)
  {
    return new KNITROSolver (pb);
  }

  ROBOPTIM_DLLEXPORT void destroy (solver_t* p)
  {
    delete p;
  }
}
