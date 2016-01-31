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

#include <stdexcept>

#include <boost/static_assert.hpp>
#include <boost/filesystem.hpp>
#include <boost/variant.hpp>

#include <knitro.h>

#include <roboptim/core/function.hh>
#include <roboptim/core/indent.hh>
#include <roboptim/core/result.hh>
#include <roboptim/core/result-with-warnings.hh>
#include <roboptim/core/util.hh>

#include "roboptim/core/plugin/knitro/util.hh"
#include "roboptim/core/plugin/knitro/knitro-solver.hh"
#include "roboptim/core/plugin/knitro/knitro-parameters-updater.hh"

namespace roboptim
{
  int iterationCallback (KTR_context_ptr kc, const int n,
                         const int /* m */, const int /* nnzJ */,
                         const double* const x,
                         const double* const /* lambda */, const double obj,
                         const double* const /* c */,
                         const double* const /* objGrad */,
                         const double* const /* jac */, void* userParams)
  {
    typedef KNITROSolver::vector_t vector_t;
    typedef KNITROSolver::solverState_t solverState_t;

    if (!userParams)
    {
      std::cerr << "bad user params\n";
      return -1;
    }
    const KNITROSolver* solver = static_cast<const KNITROSolver*> (userParams);

    if (!solver->callback ()) return 0;

    solverState_t& solverState = solver->solverState ();
    solverState.x () = Eigen::Map<const vector_t> (x, n);
    solverState.cost () = obj;
    solverState.constraintViolation () = KTR_get_abs_feas_error (kc);

    // call user-defined callback
    solver->callback () (solver->problem (), solverState);

    return 0;
  }

  int computeCallback (const int evalRequestCode, const int n, const int m,
                       const int /* nnzJ */, const int /* nnzH */,
                       const double* const x, const double* const /* lambda */,
                       double* const obj, double* const c,
                       double* const objGrad, double* const jac,
                       double* const /* hessian */,
                       double* const /* hessVector */, void* userParams)
  {
    typedef KNITROSolver::argument_t argument_t;
    typedef KNITROSolver::gradient_t gradient_t;
    typedef KNITROSolver::jacobian_t jacobian_t;
    typedef KNITROSolver::result_t result_t;
    typedef KNITROSolver::problem_t::constraints_t::const_iterator iterator_t;

    if (!userParams)
    {
      std::cerr << "bad user params\n";
      return -1;
    }
    KNITROSolver* solver = static_cast<KNITROSolver*> (userParams);

    const Eigen::Map<const argument_t> x_ (x, n);

    // ask to evaluate objective and constraints
    if (evalRequestCode == KTR_RC_EVALFC)
    {
      Eigen::Map<result_t> obj_ (obj, 1);

      // objective
      obj_ = solver->problem ().function () (x_);

      // constraints
      ptrdiff_t idx = 0;
      Eigen::Map<result_t> constraintsBuf (c, m);
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
      Eigen::Map<gradient_t> objGrad_ (objGrad, n);

      // cost gradient
      const DifferentiableFunction* df = 0;
      if (solver->problem ().function ().asType<DifferentiableFunction> ())
      {
        df = solver->problem ().function ().castInto<DifferentiableFunction> ();
        objGrad_ = df->gradient (x_);
      }

      // constraints jacobian
      ptrdiff_t idx = 0;
      df = 0;
      Eigen::Map<jacobian_t> jacobianBuf (jac, m, n);
      for (iterator_t it = solver->problem ().constraints ().begin ();
           it != solver->problem ().constraints ().end (); ++it)
      {
        if ((*it)->asType<DifferentiableFunction> ())
        {
          df = (*it)->castInto<DifferentiableFunction> ();
          df->jacobian (jacobianBuf.block (idx, 0, (*it)->outputSize (), n),
                        x_);
          idx += (*it)->outputSize ();
        }
      }
    }
    else
    {
      std::cerr << "bad evalRequestCode\n";
      return -1;
    }
    return 0;
  }

  KNITROSolver::KNITROSolver (const problem_t& problem)
    : parent_t (problem), solverState_ (problem), knitro_ ()
  {
    initializeParameters ();
  }

  KNITROSolver::~KNITROSolver ()
  {
    KTR_free (&knitro_);
  }

#define FILL_RESULT()                                           \
  res.x = x;                                                    \
  res.lambda.segment (0, n) = lambda.segment (m, m + n);        \
  res.lambda.segment (n, n + m) = lambda.segment (0, m);        \
  res.constraints.resize (m);                                   \
  KTR_get_constraint_values (knitro_, res.constraints.data ()); \
  res.value = obj

#define SWITCH_ERROR(NAME, ERROR)  \
  case NAME:                       \
  {                                \
    Result res (n, 1);             \
    FILL_RESULT ();                \
    result_ = SolverError (ERROR); \
  }                                \
  break

// FIXME Fill with actual warning
#define SWITCH_WARNING(NAME, WARNING)                 \
  case NAME:                                          \
  {                                                   \
    ResultWithWarnings res (n, 1);                    \
    FILL_RESULT ();                                   \
    res.warnings.push_back (SolverWarning (WARNING)); \
    result_ = res;                            \
  }                                                   \
  break

#define MAP_KNITRO_ERRORS(MACRO)                                                \
  MACRO (KTR_RC_FEAS_XTOL,                                                      \
         "Primal feasible solution; the optimization terminated because the "   \
         "relative change in the solution estimate is less than that "          \
         "specified by the parameter xtol. To try to get more accuracy one "    \
         "may decrease xtol. If xtol is very small already, it is an "          \
         "indication that no more significant progress can be made. It's "      \
         "possible the approximate feasible solution is optimal, but perhaps "  \
         "the stopping tests cannot be satisfied because of degeneracy, "       \
         "ill-conditioning or bad scaling.");                                   \
  MACRO (KTR_RC_FEAS_NO_IMPROVE,                                                \
         "Primal feasible solution estimate cannot be improved; desired "       \
         "accuracy in dual feasibility could not be achieved. No further "      \
         "progress can be made. It's possible the approximate feasible "        \
         "solution is optimal, but perhaps the stopping tests cannot be "       \
         "satisfied because of degeneracy, ill-conditioning or bad scaling.");  \
  MACRO (KTR_RC_FEAS_FTOL,                                                      \
         "Primal feasible solution; the optimization terminated because the "   \
         "relative change in the objective function is less than that "         \
         "specified by the parameter ftol for ftol_iters consecutive "          \
         "iterations. To try to get more accuracy one may decrease ftol "       \
         "and/or increase ftol_iters. If ftol is very small already, it is "    \
         "an indication that no more significant progress can be made. It's "   \
         "possible the approximate feasible solution is optimal, but perhaps "  \
         "the stopping tests cannot be satisfied because of degeneracy, "       \
         "ill-conditioning or bad scaling.");                                   \
  MACRO (KTR_RC_INFEASIBLE,                                                     \
         "Convergence to an infeasible point. Problem may be locally "          \
         "infeasible. If problem is believed to be feasible, try multistart "   \
         "to search for feasible points. The algorithm has converged to an "    \
         "infeasible point from which it cannot further decrease the "          \
         "infeasibility measure. This happens when the problem is "             \
         "infeasible, but may also occur on occasion for feasible problems "    \
         "with nonlinear constraints or badly scaled problems. It is "          \
         "recommended to try various initial points with the multi-start "      \
         "feature. If this occurs for a variety of initial points, it is "      \
         "likely the problem is infeasible.");                                  \
  MACRO (KTR_RC_INFEAS_XTOL,                                                    \
         "Terminate at infeasible point because the relative change in the "    \
         "solution estimate is less than that specified by the parameter "      \
         "xtol. To try to find a feasible point one may decrease xtol. If "     \
         "xtol is very small already, it is an indication that no more "        \
         "significant progress can be made. It is recommended to try various "  \
         "initial points with the multi-start feature. If this occurs for a "   \
         "variety of initial points, it is likely the problem is "              \
         "infeasible.");                                                        \
  MACRO (KTR_RC_INFEAS_NO_IMPROVE,                                              \
         "Current infeasible solution estimate cannot be improved. Problem "    \
         "may be badly scaled or perhaps infeasible. If problem is believed "   \
         "to be feasible, try multistart to search for feasible points. If "    \
         "this occurs for a variety of initial points, it is likely the "       \
         "problem is infeasible.");                                             \
  MACRO (KTR_RC_INFEAS_MULTISTART,                                              \
         "Multistart: no primal feasible point found. The multi-start "         \
         "feature was unable to find a feasible point. If the problem is "      \
         "believed to be feasible, then increase the number of initial "        \
         "points tried in the multi-start feature and also perhaps increase "   \
         "the range from which random initial points are chosen.");             \
  MACRO (KTR_RC_INFEAS_CON_BOUNDS,                                              \
         "The constraint bounds have been determined to be infeasible.");       \
  MACRO (KTR_RC_INFEAS_VAR_BOUNDS,                                              \
         "The variable bounds have been determined to be infeasible.");         \
  MACRO (KTR_RC_UNBOUNDED,                                                      \
         "Problem appears to be unbounded. Iterate is feasible and objective "  \
         "magnitude is greater than objrange. The objective function appears "  \
         "to be decreasing without bound, while satisfying the constraints. "   \
         "If the problem really is bounded, increase the size of the "          \
         "parameter objrange to avoid terminating with this message.");         \
  MACRO (KTR_RC_CALLBACK_ERR,                                                   \
         "Callback function error. This termination value indicates that an "   \
         "error (i.e., negative return value) occurred in a user provided "     \
         "callback routine.");                                                  \
  MACRO (KTR_RC_LP_SOLVER_ERR,                                                  \
         "LP solver error. This termination value indicates that an "           \
         "unrecoverable error occurred in the LP solver used in the "           \
         "active-set algorithm preventing the optimization from continuing.");  \
  MACRO (KTR_RC_EVAL_ERR,                                                       \
         "Evaluation error. This termination value indicates that an "          \
         "evaluation error occurred (e.g., divide by 0, taking the square "     \
         "root of a negative number), preventing the optimization from "        \
         "continuing.");                                                        \
  MACRO (KTR_RC_OUT_OF_MEMORY,                                                  \
         "Not enough memory available to solve problem. This termination "      \
         "value indicates that there was not enough memory available to "       \
         "solve the problem.")

#define MAP_KNITRO_WARNINGS(MACRO)                                            \
  MACRO (KTR_RC_NEAR_OPT,                                                     \
         "iPrimal feasible solution estimate cannot be improved. It appears " \
         "to be optimal, but desired accuracy in dual feasibility could not " \
         "be achieved. No more progress can be made, but the stopping tests " \
         "are close to being satisfied (within a factor of 100) and so the "  \
         "current approximate solution is believed to be optimal.");          \
  MACRO (KTR_RC_ITER_LIMIT,                                                   \
         "The iteration limit was reached before being able to satisfy the "  \
         "required stopping criteria. A feasible point was found. The "       \
         "iteration limit can be increased through the user option maxit.");  \
  MACRO (KTR_RC_TIME_LIMIT,                                                   \
         "The time limit was reached before being able to satisfy the "       \
         "required stopping criteria. A feasible point was found. The time "  \
         "limit can be increased through the user options maxtime_cpu and "   \
         "maxtime_real.");                                                    \
  MACRO (KTR_RC_USER_TERMINATION, "Knitro has been terminated by the user.")

  // TODO: investigate the following error code:
  // MACRO (KTR_RC_FEVAL_LIMIT,
  //        "The function evaluation limit was reached before being able to "
  //        "satisfy the required stopping criteria. A feasible point was "
  //        "found. The function evaluation limit can be increased through the "
  //        "user option maxfevals.");

  void KNITROSolver::solve ()
  {
    int nStatus = 0;

    updateParameters ();
    // if(!setKnitroParams())
    //    return;

    // Problem definition

    // number of variables
    int n = static_cast<int> (problem ().function ().inputSize ());

    // number of constraints
    int m = static_cast<int> (problem ().constraintsOutputSize ());
    int nnzJ = n * m;
    int nnzH = 0;

    int objType = KTR_OBJTYPE_GENERAL;
    int objGoal = KTR_OBJGOAL_MINIMIZE;

    // bounds and constraints type
    vector_t xLoBnds (n);
    vector_t xUpBnds (n);
    for (int i = 0; i < n; i++)
    {
      if (problem ().argumentBounds ()[i].first == -Function::infinity ())
        xLoBnds[i] = -KTR_INFBOUND;
      else
        xLoBnds[i] = problem ().argumentBounds ()[i].first;

      if (problem ().argumentBounds ()[i].second == Function::infinity ())
        xUpBnds[i] = KTR_INFBOUND;
      else
        xUpBnds[i] = problem ().argumentBounds ()[i].second;
    }

    Eigen::VectorXi cType (m);
    vector_t cLoBnds (m);
    vector_t cUpBnds (m);

    typedef KNITROSolver::problem_t::constraints_t::const_iterator iterator_t;
    ptrdiff_t offset = 0;
    for (iterator_t it = problem ().constraints ().begin ();
         it != problem ().constraints ().end (); ++it)
    {
      assert (offset < m);
      ptrdiff_t i = it - problem ().constraints ().begin ();

      for (int j = 0; j < (*it)->outputSize (); j++)
      {
        // FIXME: dispatch linear constraints here
        cType[offset + j] = KTR_CONTYPE_GENERAL;
        if (problem ().boundsVector ()[i][j].first ==
            -Function::infinity ())
          cLoBnds[offset + j] = -KTR_INFBOUND;
        else
          cLoBnds[offset + j] = problem ().boundsVector ()[i][j].first;

        if (problem ().boundsVector ()[i][j].second ==
            Function::infinity ())
          cUpBnds[offset + j] = KTR_INFBOUND;
        else
          cUpBnds[offset + j] = problem ().boundsVector ()[i][j].second;
      }

      offset += (*it)->outputSize ();
    }

    // initial point
    argument_t xInitial (n);
    if (problem ().startingPoint ())
      xInitial = *problem ().startingPoint ();
    else
      xInitial.setZero ();

    // sparsity pattern (dense here)
    Eigen::VectorXi jacIndexVars (nnzJ);
    Eigen::VectorXi jacIndexCons (nnzJ);

    // FIXME: this depends on RowMajor/ColMajor
    BOOST_STATIC_ASSERT (Eigen::ROBOPTIM_STORAGE_ORDER == Eigen::ColMajor);
    int k = 0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++)
      {
        jacIndexCons[k] = j;
        jacIndexVars[k] = i;
        k++;
      }

    nStatus = KTR_init_problem (knitro_, n, objGoal, objType, xLoBnds.data(),
                                xUpBnds.data(), m, cType.data(), cLoBnds.data(),
                                cUpBnds.data(), nnzJ, jacIndexVars.data(),
                                jacIndexCons.data(), nnzH, 0, 0, xInitial.data(), 0);

    argument_t x (n);
    vector_t lambda (m + n);
    result_t obj (1);
    nStatus =
      KTR_solve (knitro_, x.data(), lambda.data(), 0, obj.data(), 0, 0, 0, 0, 0, this);

    switch (nStatus)
    {
#ifdef KTR_RC_OPTIMAL_OR_SATISFACTORY
      case KTR_RC_OPTIMAL_OR_SATISFACTORY:
#else //! KTR_RC_OPTIMAL_OR_SATISFACTORY
      case KTR_RC_OPTIMAL:
#endif //! KTR_RC_OPTIMAL_OR_SATISFACTORY
      {

        Result res (n, 1);
        FILL_RESULT ();
        result_ = res;
        break;
      }

      MAP_KNITRO_WARNINGS (SWITCH_WARNING);
      MAP_KNITRO_ERRORS (SWITCH_ERROR);

      default:
        throw std::runtime_error ("unknown KNITRO return code");
    }

    KTR_free (&knitro_);
  }

#undef SWITCH_ERROR
#undef SWITCH_WARNING
#undef MAP_KNITRO_ERRORS
#undef MAP_KNITRO_WARNINGS

#define DEFINE_PARAMETER(KEY, DESCRIPTION, VALUE)     \
  do                                                  \
  {                                                   \
    this->parameters_[KEY].description = DESCRIPTION; \
    this->parameters_[KEY].value = VALUE;             \
  } while (0)

  void KNITROSolver::initializeParameters ()
  {
    this->parameters_.clear ();

    // KNITRO specific.
    // Much more options are available for Knitro see Knitro documentation

    if (!knitro_) knitro_ = KTR_new ();
    if (!knitro_) throw std::runtime_error ("failed to initialize KNITRO");

    if (KTR_set_func_callback (knitro_, &computeCallback))
      throw std::runtime_error ("failed to set evaluation callback");
    if (KTR_set_grad_callback (knitro_, &computeCallback))
      throw std::runtime_error ("failed to set gradient callback");

    //  Output
    DEFINE_PARAMETER ("knitro.outlev", "output verbosity level", 4);
    DEFINE_PARAMETER ("knitro.outmode",
                      "output style (\"screen\", \"file\" or \"both\")",
                      std::string ("file"));
    DEFINE_PARAMETER ("knitro.outdir", "output directory", std::string ("."));
    DEFINE_PARAMETER ("knitro.outappend", "output append mode", false);

    // Gradient and hessian used
    DEFINE_PARAMETER ("knitro.gradopt", "type of gradient method used",
                      std::string ("GRADOPT_FORWARD"));
    DEFINE_PARAMETER ("knitro.hessopt", "type of hessian method used",
                      std::string ("HESSOPT_BFGS"));

    //  Termination
    DEFINE_PARAMETER ("knitro.maxit", "maximum number of iteration permitted",
                      3000);
    DEFINE_PARAMETER ("knitro.opttol",
                      "desired convergence tolerance (relative)", 1e-6);
    DEFINE_PARAMETER ("knitro.feastol", "desired threshold for the feasibility",
                      1e-6);
    DEFINE_PARAMETER ("knitro.xtol",
                      "desired threshold for the constraint violation", 1e-5);

    //  Barrier parameter
    DEFINE_PARAMETER ("knitro.bar_initmu", "barrier initial mu", 1e-1);

    // Algorithm choice.
    DEFINE_PARAMETER ("knitro.algorithm", "type of solver algorithm",
                      std::string ("ALG_BAR_DIRECT"));

    // Miscellaneous
    DEFINE_PARAMETER ("knitro.par_numthreads",
                      "number of parallel threads to use", 1);

    stringToEnum_ = knitroParameterMap ();
  }


  void KNITROSolver::updateParameters ()
  {
    // Create the log directory
    createLogDir ();

    // If a user-defined callback was set
    if (callback ())
    {
      DEFINE_PARAMETER ("knitro.newpoint",
                        "action on a newpoint (e.g. callback)", std::string ("user"));

      if (KTR_set_newpt_callback (knitro_, &iterationCallback))
        throw std::runtime_error ("failed to set solver callback");
    }

    const std::string prefix = "knitro.";
    typedef const std::pair<const std::string, Parameter> const_iterator_t;
    BOOST_FOREACH (const_iterator_t& it, this->parameters_)
    {
      if (it.first.substr (0, prefix.size ()) == prefix)
      {
        boost::apply_visitor (
          KnitroParametersUpdater (knitro_, it.first.substr (prefix.size ()),
                                   stringToEnum_),
          it.second.value);
      }
    }
  }

#undef DEFINE_PARAMETER

  std::ostream& KNITROSolver::print (std::ostream& o) const
  {
    parent_t::print (o);
    return o;
  }

  void KNITROSolver::setIterationCallback (callback_t callback)
  {
    callback_ = callback;
  }

  void KNITROSolver::createLogDir () const
  {
    // If the output level is not 0
    parameters_t::const_iterator it_lvl, it_file, it_name;
    it_lvl = parameters_.find ("knitro.outlev");
    if (it_lvl == parameters_.end ()) return;

    int loglvl = boost::get<int> (it_lvl->second.value);
    if (loglvl <= 0) return;

    // If logging to file is enabled
    it_file = parameters_.find ("knitro.outmode");
    if (it_file == parameters_.end ()) return;

    const std::string& logmode =
      boost::get<std::string> (it_file->second.value);
    if (logmode != "file" and logmode != "both") return;

    it_name = parameters_.find ("knitro.outdir");
    if (it_name != parameters_.end ())
    {
      boost::filesystem::path logdir (
        boost::get<std::string> (it_name->second.value));
      boost::filesystem::create_directory (logdir);
    }
  }
} // end of namespace roboptim.

extern "C" {
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
