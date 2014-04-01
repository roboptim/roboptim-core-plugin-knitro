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

  template <typename U>
  class EvaluateConstraintJacobian : public boost::static_visitor<>
  {
  public:
    EvaluateConstraintJacobian (int n,
				int m,
				double* const jac,
				std::ptrdiff_t& offset,
				const U& x)
      : boost::static_visitor<> (),
	n (n),
	m (m),
	jac (jac),
	offset (offset),
	x (x)
    {}

    template<typename T>
    void
    operator () (const boost::shared_ptr<T>& constraint) const
    {
      Eigen::Map<KNITROSolver::matrix_t> jac_ (jac, m, n);
      jac_.block (offset, 0, constraint->outputSize (), n) =
	constraint->jacobian (x);
      offset += constraint->outputSize ();
    }

  private:
    int n;
    int m;
    double* const jac;
    std::ptrdiff_t& offset;
    const U& x;
  };

  static int KNITRO_callback (const int evalRequestCode,
			      const int n,
			      const int ROBOPTIM_DEBUG_ONLY (m),
			      const int /* nnzJ */,
			      const int /* nnzH */,
			      const double* const x,
			      const double* const /* lambda */,
			      double* const obj,
			      double* const c,
			      double* const objGrad ,
			      double* const jac,
			      double* const /* hessian */,
			      double* const /* hessVector */,
			      void* userParams)
  {
    typedef KNITROSolver::matrix_t matrix_t;
    typedef KNITROSolver::vector_t vector_t;
    typedef KNITROSolver::value_type value_type;
    typedef KNITROSolver::problem_t::constraints_t::const_iterator iterator_t;

    if (!userParams)
      {
	std::cerr << "bad user params\n";
	return -1;
      }
    KNITROSolver* solver = static_cast<KNITROSolver*> (userParams);

    Eigen::Map<const Eigen::Matrix<value_type, Eigen::Dynamic, 1> > x_ (x, n);
    // ask to evaluate objective and constraints
    if (evalRequestCode == KTR_RC_EVALFC)
      {
	Eigen::Map<Eigen::Matrix<value_type, Eigen::Dynamic, 1> > obj_ (obj, 1);

	// objective
	obj_ = solver->problem ().function () (x_);

	// constraints
	std::ptrdiff_t offset = 0;

	for (iterator_t it = solver->problem ().constraints ().begin ();
	     it != solver->problem ().constraints ().end (); ++it)
	  {
	    assert (offset < m);
	    boost::apply_visitor
	      (EvaluateConstraint<Eigen::Map<const vector_t> >
		(c, offset, x_), *it);
	  }
      }
    // evaluate cost gradient and constraints jacobian
    else if (evalRequestCode == KTR_RC_EVALGA)
      {
	Eigen::Map<vector_t> objGrad_ (objGrad, n);

	// cost gradient
	objGrad_ = solver->problem ().function ().gradient (x_);

	// constraints jacobian
	std::ptrdiff_t offset = 0;

	for (iterator_t it = solver->problem ().constraints ().begin ();
	     it != solver->problem ().constraints ().end (); ++it)
	  {
	    assert (offset < m);
	    boost::apply_visitor
	      (EvaluateConstraintJacobian<Eigen::Map<const vector_t> >
	       (n, m, jac, offset, x_), *it);
	  }
      }
    else
      {
	std::cerr << "bad evalRequestCode\n";
	return -1;
      }
    return 0;
  }

  static int KNITRO_newpt_callback
  (KTR_context_ptr,
   const int,
   const int,
   const int,
   const double* const x,
   const double* const,
   const double,
   const double* const,
   const double* const,
   const double* const,
   void*  userParams)
  {
    typedef KNITROSolver::matrix_t matrix_t;
    typedef KNITROSolver::vector_t vector_t;
    typedef KNITROSolver::value_type value_type;
    typedef KNITROSolver::problem_t::constraints_t::const_iterator iterator_t;

    if (!userParams)
      {
	std::cerr << "bad user params\n";
	return -1;
      }
    KNITROSolver* solver = static_cast<KNITROSolver*> (userParams);
    if (!solver->callback ())
      return 0;

    vector_t x_ = Eigen::Map<const Eigen::VectorXd>
      (x, solver->problem ().function ().inputSize ());
    solver->solverState ().x () = x_;
    solver->callback () (solver->problem (), solver->solverState ());
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

  class StaticCastConstraint
    : public boost::static_visitor<boost::shared_ptr<Function> >
  {
  public:
    template<typename T>
    boost::shared_ptr<Function>
    operator () (const boost::shared_ptr<T>& constraint) const
    {
      return boost::static_pointer_cast<Function> (constraint);
    }
  };

#define DEFINE_PARAMETER(KEY, DESCRIPTION, VALUE)	\
  do {							\
    parameters_[KEY].description = DESCRIPTION;		\
    parameters_[KEY].value = VALUE;			\
  } while (0)

  KNITROSolver::KNITROSolver (const problem_t& problem) throw ()
    : parent_t (problem),
      solverState_ (problem),
      knitro_ ()
  {
    DEFINE_PARAMETER ("knitro.outlev", "output level", 0);
  }

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
	result_ = SolverError ("failed to initialize KNITRO");
	return;
      }

    if (KTR_set_int_param_by_name (knitro_, "gradopt", KTR_GRADOPT_EXACT))
      {
	result_ = SolverError ("failed to set parameter gradopt");
	return;
      }
    if (KTR_set_int_param_by_name (knitro_, "hessopt", KTR_HESSOPT_BFGS))
      {
	result_ = SolverError ("failed to set parameter hessopt");
	return;
      }
    if (KTR_set_int_param_by_name
	(knitro_, "outlev", getParameter<int> ("knitro.outlev")))
      {
	result_ = SolverError ("failed to set parameter outlev");
	return;
      }

    if (KTR_set_func_callback (knitro_, &KNITRO_callback))
      {
	result_ = SolverError ("failed to set function callback");
	return;
      }
    if (KTR_set_grad_callback (knitro_, &KNITRO_callback))
      {
	result_ = SolverError ("failed to set gradient callback");
	return;
      }
    if (KTR_set_newpt_callback (knitro_, &KNITRO_newpt_callback))
      {
	result_ = SolverError ("failed to set gradient callback");
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
      {
	if (this->problem ().argumentBounds ()[i].first == -Function::infinity ())
	  xLoBnds[i] = -KTR_INFBOUND;
	else
	  xLoBnds[i] = this->problem ().argumentBounds ()[i].first;
	if (this->problem ().argumentBounds ()[i].second == Function::infinity ())
	  xUpBnds[i] = KTR_INFBOUND;
	else
	  xUpBnds[i] = this->problem ().argumentBounds ()[i].second;
      }

    std::vector<int> cType (m);
    std::vector<double> cLoBnds (m);
    std::vector<double> cUpBnds (m);

    typedef problem_t::constraints_t::const_iterator iterator_t;
    std::ptrdiff_t offset = 0;
    for (iterator_t it = this->problem ().constraints ().begin ();
	 it != this->problem ().constraints ().end (); ++it)
      {
	assert (offset < m);
	std::ptrdiff_t i = it - this->problem ().constraints ().begin ();

	const boost::shared_ptr<Function> constraint =
	  boost::apply_visitor (StaticCastConstraint (), *it);

	for (int j = 0; j < constraint->outputSize (); j++)
	  {
	    //FIXME: dispatch linear constraints here
	    cType[offset + j] = KTR_CONTYPE_GENERAL;
	    if (this->problem ().boundsVector ()[i][j].first == -Function::infinity ())
	      cLoBnds[offset + j] = -KTR_INFBOUND;
	    else
	      cLoBnds[offset + j] = this->problem ().boundsVector ()[i][j].first;

	    if (this->problem ().boundsVector ()[i][j].second == Function::infinity ())
	      cUpBnds[offset + j] = KTR_INFBOUND;
	    else
	      cUpBnds[offset + j] = this->problem ().boundsVector ()[i][j].second;
	  }

	offset += constraint->outputSize ();
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
    vector_t lambda (m + n);
    vector_t obj (1);
    nStatus = KTR_solve
      (knitro_, &x[0], &lambda[0], 0, &obj[0], 0, 0, 0, 0, 0, this);

    if (nStatus != 0)
      {
	result_ = SolverError ("failed to solve problem");
	return;
      }

    Result res (n, 1);
    res.x = x;
    res.value = obj;
    res.constraints.resize (m); //FIXME: fill me.
    res.lambda = lambda;
    result_ = res;
  }

  std::ostream&
  KNITROSolver::print (std::ostream& o) const throw ()
  {
    parent_t::print (o);
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
