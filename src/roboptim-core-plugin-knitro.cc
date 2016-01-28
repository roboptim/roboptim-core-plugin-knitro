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

using namespace std;

namespace roboptim
{

  static int KNITRO_callback (const int evalRequestCode,
			      const int n,
			      const int (m),
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
    typedef KNITROSolver::vector_t vector_t;
    typedef KNITROSolver::value_type value_type;
    typedef KNITROSolver::problem_t::constraints_t::const_iterator iterator_t;

    if (!userParams)
    {
        cerr << "bad user params\n";
        return -1;
    }
    KNITROSolver* solver = static_cast<KNITROSolver*> (userParams);

    //Eigen::Map<const typename function_t::vector_t> x_ (x, n);
    Eigen::Map<const Eigen::Matrix<value_type, Eigen::Dynamic, 1> > x_ (x, n);
    // ask to evaluate objective and constraints
    if (evalRequestCode == KTR_RC_EVALFC)
    {
        Eigen::Map<Eigen::Matrix<value_type, Eigen::Dynamic, 1> > obj_ (obj, 1);

	    // objective
	    obj_ = solver->problem ().function () (x_);

	    // constraints
	    ptrdiff_t idx = 0;
        Eigen::Map<Eigen::Matrix<value_type, Eigen::Dynamic, 1> > constraintsBuf (c, m, 1);
	    for (iterator_t it = solver->problem ().constraints ().begin ();
	         it != solver->problem ().constraints ().end (); ++it)
	    {
            constraintsBuf.segment (idx, (*it)->outputSize ()) = (*(*it)) (x_);
            idx += (*it)->outputSize ();
	    }
    }
    // evaluate cost gradient and constraints jacobian
    else if (evalRequestCode == KTR_RC_EVALGA)
    {
	    Eigen::Map<vector_t> objGrad_ (objGrad, n);

	    // cost gradient
        const DifferentiableFunction* df = 0;
        if(solver->problem ().function ().asType<DifferentiableFunction>())
        {
            df = solver->problem ().function ().castInto<DifferentiableFunction>();
            objGrad_ = df->gradient (x_);
        }

        // constraints jacobian
	    ptrdiff_t idx = 0;
        df = 0;
        Eigen::Map<KNITROSolver::matrix_t> jacobianBuf (jac, m, n);
	    for (iterator_t it = solver->problem ().constraints ().begin ();
	         it != solver->problem ().constraints ().end (); ++it)
	    {
            if((*it)->asType<DifferentiableFunction>())
            {
                df = (*it)->castInto<DifferentiableFunction>();
                jacobianBuf.block (idx, 0, (*it)->outputSize (), n) = df->jacobian (x_); 
                idx += (*it)->outputSize ();
            }
	    }
    }
    else
    {
        cerr << "bad evalRequestCode\n";
        return -1;
    }
    return 0;
  }

  /*static int KNITRO_newpt_callback
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
    typedef KNITROSolver::vector_t vector_t;

    if (!userParams)
      {
	cerr << "bad user params\n";
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
  }*/

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
        result += static_cast<int> ((*it)->outputSize ());

    return result;
  }

  void
  KNITROSolver::solve () throw ()
  {
    int nStatus = 0;

    if(!setKnitroParams())
        return;

    // Problem definition

    // number of variables
    int n = static_cast<int> (this->problem ().function ().inputSize ());

    // number of constraints
    int cSize = computeConstraintsSize ();
    int nnzJ = n * cSize;
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

    std::vector<int> cType (cSize);
    std::vector<double> cLoBnds (cSize);
    std::vector<double> cUpBnds (cSize);

    typedef KNITROSolver::problem_t::constraints_t::const_iterator iterator_t;
    ptrdiff_t offset = 0;
    for (iterator_t it = this->problem ().constraints ().begin ();
    it != this->problem ().constraints ().end (); ++it)
    {
        assert (offset < cSize);
        ptrdiff_t i = it - this->problem ().constraints ().begin ();

        const boost::shared_ptr<Function> constraint = boost::static_pointer_cast<Function> (*it);
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
        xInitial.resize(this->problem ().function ().inputSize (), value_type ());

    // sparsity pattern (dense here)
    std::vector<int> jacIndexVars (nnzJ);
    std::vector<int> jacIndexCons (nnzJ);

    int k = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < cSize; j++)
        {
            jacIndexCons[k] = j;
            jacIndexVars[k] = i;
            k++;
        }


    nStatus = KTR_init_problem (knitro_, n, objGoal, objType,
                &xLoBnds[0], &xUpBnds[0],
                cSize, &cType[0], &cLoBnds[0], &cUpBnds[0],
                nnzJ, &jacIndexVars[0], &jacIndexCons[0],
                nnzH, 0, 0, &xInitial[0], 0);

    cout << "Start Solver" << endl;
    vector_t x (this->problem ().function ().inputSize ());
    vector_t lambda (cSize + n);
    vector_t obj (1);
    nStatus = KTR_solve (knitro_, &x[0], &lambda[0], 0, &obj[0], 0, 0, 0, 0, 0, this);

    cout << "Solver Resolved" << endl;
    if (nStatus != 0)
    {
        result_ = SolverError ("failed to solve problem");
        HandleErrorCode(nStatus);
        //return;
    }

    Result res (n, 1);
    res.x = x;
    res.value = obj;
    res.constraints.resize (cSize); //FIXME: fill me.
    res.lambda = lambda;
    result_ = res;
    //cout << res << endl;
    cout << res.x << endl;
    cout << res.value << endl;

    KTR_free (&knitro_);
  }

  bool KNITROSolver::setKnitroParams()
  {
    if (!knitro_)
        knitro_ = KTR_new ();

    if (!knitro_)
    {
        result_ = SolverError ("failed to initialize KNITRO");
        return false;
    }
    if (KTR_set_int_param_by_name (knitro_, "algorithm", KTR_ALG_BAR_DIRECT))
    {
        result_ = SolverError ("failed to set parameter algorithm");
        return false;
    }
    /*if (KTR_set_int_param_by_name (knitro_, "scale", KTR_SCALE_ALLOW))
    {
        result_ = SolverError ("failed to set parameter scale");
        return false;
    }*/
    if (KTR_set_int_param_by_name (knitro_, "maxit", 10))
    {
        result_ = SolverError ("failed to set the maximum number of iterations");
        return false;
    }
    if (KTR_set_int_param_by_name (knitro_, "gradopt", KTR_GRADOPT_FORWARD))//FORWARD, EXACT, CENTRAL
    {
        result_ = SolverError ("failed to set parameter gradopt");
        return false;
    }
    if (KTR_set_int_param_by_name (knitro_, "hessopt", KTR_HESSOPT_BFGS))
    {
        result_ = SolverError ("failed to set parameter hessopt");
        return false;
    }
    if (KTR_set_int_param_by_name (knitro_, "outlev", 4))
    {
        result_ = SolverError ("failed to set parameter outlev");
        return false;
    }
    if(KTR_set_int_param_by_name (knitro_, "par_numthreads", 8))
    {
        result_ = SolverError ("failed to set number of threads");
        return false;
    }
    if(KTR_set_int_param_by_name (knitro_, "honorbnds", KTR_HONORBNDS_ALWAYS))//_abs
    {
        result_ = SolverError ("failed to set the initial barriver param");//TolFun
        return false;
    }
    if(KTR_set_double_param_by_name (knitro_, "bar_initmu", 0.01))//_abs
    {
        result_ = SolverError ("failed to set the initial barriver param");//TolFun
        return false;
    }
    /*if (KTR_set_int_param_by_name (knitro_, "bar_murule", KTR_BAR_MURULE_FULLMPC))
    {
        result_ = SolverError ("failed to set parameter barrier mu rule");
        return false;
    }*/
    if(KTR_set_double_param_by_name (knitro_, "opttol", 1))//_abs
    {
        result_ = SolverError ("failed to set the termination tolerance on the function value");//TolFun
        return false;
    }
    if(KTR_set_double_param_by_name (knitro_, "feastol", 1e-2))//_abs, infeastol
    {
        result_ = SolverError ("failed to set the termination tolerance on the constraint violation");//TolCon
        return false;
    }
    if(KTR_set_double_param_by_name (knitro_, "xTol", 1e-5))
    {
        result_ = SolverError ("failed to set the termination on x");//TolX
        return false;
    }
    if (KTR_set_func_callback (knitro_, &KNITRO_callback))
    {
        result_ = SolverError ("failed to set function callback");
        return false;
    }
    if (KTR_set_grad_callback (knitro_, &KNITRO_callback))
    {
        result_ = SolverError ("failed to set gradient callback");
        return false;
    }
    /*if (KTR_set_newpt_callback (knitro_, &KNITRO_newpt_callback))
    {
        result_ = SolverError ("failed to set newpt callback");
        return false;
    }*/
    return true;
  }

  void
  KNITROSolver::HandleErrorCode(int errorCode)
  {
      switch(errorCode)
      {
        case KTR_RC_NEAR_OPT:
            cout << "iPrimal feasible solution estimate cannot be improved. It appears to be optimal, but desired accuracy in dual feasibility could not be achieved. No more progress can be made, but the stopping tests are close to being satisfied (within a factor of 100) and so the current approximate solution is believed to be optimal." << endl;
            break;
        case KTR_RC_FEAS_XTOL:
            cout << "Primal feasible solution; the optimization terminated because the relative change in the solution estimate is less than that specified by the parameter xtol. To try to get more accuracy one may decrease xtol. If xtol is very small already, it is an indication that no more significant progress can be made. It’s possible the approximate feasible solution is optimal, but perhaps the stopping tests cannot be satisfied because of degeneracy, ill-conditioning or bad scaling." << endl;
            break;
        case KTR_RC_FEAS_NO_IMPROVE:
            cout << "Primal feasible solution estimate cannot be improved; desired accuracy in dual feasibility could not be achieved. No further progress can be made. It’s possible the approximate feasible solution is optimal, but perhaps the stopping tests cannot be satisfied because of degeneracy, ill-conditioning or bad scaling." << endl;
            break;
        case KTR_RC_FEAS_FTOL:
            cout << "Primal feasible solution; the optimization terminated because the relative change in the objective function is less than that specified by the parameter ftol for ftol_iters consecutive iterations. To try to get more accuracy one may decrease ftol and/or increase ftol_iters. If ftol is very small already, it is an indication that no more significant progress can be made. It’s possible the approximate feasible solution is optimal, but perhaps the stopping tests cannot be satisfied because of degeneracy, ill-conditioning or bad scaling." << endl;
            break;
        case KTR_RC_INFEASIBLE:
            cout << "Convergence to an infeasible point. Problem may be locally infeasible. If problem is believed to be feasible, try multistart to search for feasible points. The algorithm has converged to an infeasible point from which it cannot further decrease the infeasibility measure. This happens when the problem is infeasible, but may also occur on occasion for feasible problems with nonlinear constraints or badly scaled problems. It is recommended to try various initial points with the multi-start feature. If this occurs for a variety of initial points, it is likely the problem is infeasible." << endl;
            break;
        case KTR_RC_INFEAS_XTOL:
            cout << "Terminate at infeasible point because the relative change in the solution estimate is less than that specified by the parameter xtol. To try to find a feasible point one may decrease xtol. If xtol is very small already, it is an indication that no more significant progress can be made. It is recommended to try various initial points with the multi-start feature. If this occurs for a variety of initial points, it is likely the problem is infeasible." << endl;
            break;
        case KTR_RC_INFEAS_NO_IMPROVE:
            cout << "Current infeasible solution estimate cannot be improved. Problem may be badly scaled or perhaps infeasible. If problem is believed to be feasible, try multistart to search for feasible points. If this occurs for a variety of initial points, it is likely the problem is infeasible." << endl;
            break;
        case KTR_RC_INFEAS_MULTISTART:
            cout << "Multistart: no primal feasible point found. The multi-start feature was unable to find a feasible point. If the problem is believed to be feasible, then increase the number of initial points tried in the multi-start feature and also perhaps increase the range from which random initial points are chosen." << endl;
            break;
        case KTR_RC_INFEAS_CON_BOUNDS:
            cout << "The constraint bounds have been determined to be infeasible." << endl;
            break;
        case KTR_RC_INFEAS_VAR_BOUNDS:
            cout << "The variable bounds have been determined to be infeasible." << endl;
            break;
        case KTR_RC_UNBOUNDED:
            cout << "Problem appears to be unbounded. Iterate is feasible and objective magnitude is greater than objrange. The objective function appears to be decreasing without bound, while satisfying the constraints. If the problem really is bounded, increase the size of the parameter objrange to avoid terminating with this message." << endl;
            break;
        case KTR_RC_ITER_LIMIT:
            cout << "The iteration limit was reached before being able to satisfy the required stopping criteria. A feasible point was found. The iteration limit can be increased through the user option maxit." << endl;
            break;
        case KTR_RC_TIME_LIMIT:
            cout << "The time limit was reached before being able to satisfy the required stopping criteria. A feasible point was found. The time limit can be increased through the user options maxtime_cpu and maxtime_real." << endl;
            break;
        case KTR_RC_FEVAL_LIMIT:
            cout << "The function evaluation limit was reached before being able to satisfy the required stopping criteria. A feasible point was found. The function evaluation limit can be increased through the user option maxfevals." << endl;
            break;
        case KTR_RC_CALLBACK_ERR:
            cout << "Callback function error. This termination value indicates that an error (i.e., negative return value) occurred in a user provided callback routine." << endl;
            break;
        case KTR_RC_LP_SOLVER_ERR:
            cout << "LP solver error. This termination value indicates that an unrecoverable error occurred in the LP solver used in the active-set algorithm preventing the optimization from continuing." << endl;
            break;
        case KTR_RC_EVAL_ERR:
            cout << "Evaluation error. This termination value indicates that an evaluation error occurred (e.g., divide by 0, taking the square root of a negative number), preventing the optimization from continuing." << endl;
            break;
        case KTR_RC_OUT_OF_MEMORY:
            cout << "Not enough memory available to solve problem. This termination value indicates that there was not enough memory available to solve the problem." << endl;
            break;
        case KTR_RC_USER_TERMINATION:
            cout << "Knitro has been terminated by the user." << endl;
            break;
        default:
            cout << "Unhandled error code of type: " << errorCode << endl;
            break;
      }
  }


  ostream&
  KNITROSolver::print (ostream& o) const throw ()
  {
    parent_t::print (o);
    return o;
  }

  void
  KNITROSolver::setIterationCallback (callback_t callback)
    throw (runtime_error)
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
